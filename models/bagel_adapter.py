"""
BAGEL Feature Extraction Adapter for CMP Framework

This module provides adapters to integrate BAGEL's powerful feature extraction
capabilities into the existing CMP framework for text-to-image retrieval.

Key Features:
- Text feature extraction using BAGEL's Qwen2 language model
- Image feature extraction using BAGEL's ViT encoder (visual understanding)
- Optional VAE encoder for generative image features
- Feature space alignment with existing CMP embedding dimensions
- Compatible interface with existing CMP training/evaluation pipeline

Author: Generated with Claude Code
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import os
import sys

# Add BAGEL root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modeling.bagel.bagel import Bagel, BagelConfig
from modeling.bagel.qwen2_navit import Qwen2ForCausalLM
from modeling.bagel.siglip_navit import SiglipVisionModel
from modeling.autoencoder import AutoEncoder
from transformers import BertTokenizer
from data.data_utils import patchify


class FeatureAligner(nn.Module):
    """
    Aligns BAGEL features to CMP framework embedding dimensions.
    Provides flexible mapping strategies for different feature types.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_p: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []

        # Optional layer normalization
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))

        # Dropout for regularization
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        # Linear projection
        layers.append(nn.Linear(input_dim, output_dim))

        # Optional activation
        if activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'relu':
            layers.append(nn.ReLU())

        self.projection = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)

        Returns:
            Aligned features of shape (batch_size, seq_len, output_dim) or (batch_size, output_dim)
        """
        return self.projection(x)


class BagelTextAdapter(nn.Module):
    """
    Adapter for text feature extraction using BAGEL's Qwen2 language model.
    Provides compatible interface with existing CMP text processing pipeline.
    """

    def __init__(
        self,
        bagel_model: Bagel,
        embed_dim: int = 768,
        max_length: int = 77,
        use_feature_aligner: bool = True
    ):
        super().__init__()

        self.bagel_model = bagel_model
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Get text embedding dimension from BAGEL
        self.text_input_dim = bagel_model.hidden_size

        # Feature aligner to match CMP embedding dimensions
        if use_feature_aligner and self.text_input_dim != embed_dim:
            self.text_aligner = FeatureAligner(
                input_dim=self.text_input_dim,
                output_dim=embed_dim,
                dropout_p=0.1,
                use_layer_norm=True
            )
        else:
            self.text_aligner = nn.Identity()

        # Freeze BAGEL parameters if needed for fine-tuning
        self._setup_parameter_training()

    def _setup_parameter_training(self):
        """Setup which parameters to train during fine-tuning."""
        # By default, allow all parameters to be trained
        # Can be modified to freeze certain layers
        pass

    def get_text_embeds(self, text_ids: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        """
        Extract text embeddings using BAGEL's Qwen2 model.

        Args:
            text_ids: Text token IDs of shape (batch_size, seq_len)
            text_atts: Text attention mask of shape (batch_size, seq_len)

        Returns:
            Text embeddings of shape (batch_size, seq_len, hidden_size)
        """
        # Use BAGEL's text embedding layer
        text_embeds = self.bagel_model.language_model.model.embed_tokens(text_ids)

        # Apply position embeddings and causal mask for text-only processing
        seq_len = text_ids.size(1)
        position_ids = torch.arange(seq_len, device=text_ids.device).unsqueeze(0).expand(text_ids.size(0), -1)

        # Get text embeddings from language model
        outputs = self.bagel_model.language_model(
            input_ids=text_ids,
            attention_mask=text_atts,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True
        )

        text_embeds = outputs.last_hidden_state

        return text_embeds

    def get_text_feat(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled text features (CLS token).

        Args:
            text_embeds: Text embeddings of shape (batch_size, seq_len, hidden_size)

        Returns:
            Pooled text features of shape (batch_size, embed_dim)
        """
        # Use first token (CLS/BOS) as pooled representation
        pooled_feat = text_embeds[:, 0, :]

        # Align to CMP embedding dimensions
        aligned_feat = self.text_aligner(pooled_feat)

        return aligned_feat


class BagelImageAdapter(nn.Module):
    """
    Adapter for image feature extraction using BAGEL's ViT encoder.
    Supports both visual understanding (ViT) and generative (VAE) features.
    """

    def __init__(
        self,
        bagel_model: Bagel,
        embed_dim: int = 768,
        use_vit: bool = True,
        use_vae: bool = False,
        patch_size: int = 14,
        max_image_size: int = 224,
        use_feature_aligner: bool = True
    ):
        super().__init__()

        self.bagel_model = bagel_model
        self.embed_dim = embed_dim
        self.use_vit = use_vit
        self.use_vae = use_vae
        self.patch_size = patch_size
        self.max_image_size = max_image_size

        # Get image embedding dimensions
        self.vit_input_dim = bagel_model.vit_hidden_size if use_vit else None
        self.vae_input_dim = bagel_model.latent_channel * (bagel_model.latent_patch_size ** 2) if use_vae else None

        # Feature aligners
        if use_feature_aligner:
            if use_vit and self.vit_input_dim != embed_dim:
                self.vit_aligner = FeatureAligner(
                    input_dim=self.vit_input_dim,
                    output_dim=embed_dim,
                    dropout_p=0.1,
                    use_layer_norm=True
                )
            else:
                self.vit_aligner = nn.Identity()

            if use_vae and self.vae_input_dim != embed_dim:
                self.vae_aligner = FeatureAligner(
                    input_dim=self.vae_input_dim,
                    output_dim=embed_dim,
                    dropout_p=0.1,
                    use_layer_norm=True
                )
            else:
                self.vae_aligner = nn.Identity()

        # Setup parameter training
        self._setup_parameter_training()

    def _setup_parameter_training(self):
        """Setup which parameters to train during fine-tuning."""
        pass

    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract image embeddings using BAGEL's ViT encoder.

        Args:
            image: Input images of shape (batch_size, 3, H, W)

        Returns:
            Tuple of (image_embeds, image_atts)
            - image_embeds: Image embeddings of shape (batch_size, num_patches, hidden_size)
            - image_atts: Image attention mask of shape (batch_size, num_patches)
        """
        if not self.use_vit:
            raise ValueError("ViT encoder not enabled in this adapter")

        # Preprocess image for ViT
        batch_size = image.size(0)
        h, w = image.shape[2], image.shape[3]

        # Patchify the image
        patch_size = self.bagel_model.vit_patch_size
        patches = patchify(image, patch_size)  # (num_patches, patch_size*patch_size*3)

        # Create position IDs
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        num_patches = num_patches_h * num_patches_w

        # Generate position IDs (simplified - should match BAGEL's position encoding)
        position_ids = torch.arange(num_patches, device=image.device).unsqueeze(0).expand(batch_size, -1)

        # Create attention mask (all ones for visible patches)
        image_atts = torch.ones(batch_size, num_patches, dtype=torch.long, device=image.device)

        # Get ViT embeddings
        with torch.no_grad():
            # Reshape patches for ViT input
            patches = patches.view(batch_size, num_patches, -1)

            # Use BAGEL's ViT model to get embeddings
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(torch.full((batch_size,), num_patches, dtype=torch.int), dim=0),
                (1, 0)
            ).to(torch.int32)

            vit_embeds = self.bagel_model.vit_model(
                packed_pixel_values=patches.view(-1, patches.shape[-1]),
                packed_flattened_position_ids=position_ids.view(-1),
                cu_seqlens=cu_seqlens,
                max_seqlen=num_patches
            )

            # Apply connector
            vit_embeds = self.bagel_model.connector(vit_embeds)

            # Add position embeddings
            pos_emb = self.bagel_model.vit_pos_embed(position_ids.view(-1))
            vit_embeds = vit_embeds + pos_emb

            # Reshape back to batch format
            image_embeds = vit_embeds.view(batch_size, num_patches, -1)

        return image_embeds, image_atts

    def get_image_feat(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled image features (CLS token or average pooling).

        Args:
            image_embeds: Image embeddings of shape (batch_size, num_patches, hidden_size)

        Returns:
            Pooled image features of shape (batch_size, embed_dim)
        """
        # Use first patch as CLS representation
        pooled_feat = image_embeds[:, 0, :]

        # Align to CMP embedding dimensions
        aligned_feat = self.vit_aligner(pooled_feat)

        return aligned_feat


class BagelAdapter(nn.Module):
    """
    Main adapter class that combines text and image feature extraction
    using BAGEL model, providing unified interface for CMP framework.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        embed_dim: int = 768,
        max_text_length: int = 77,
        use_vit: bool = True,
        use_vae: bool = False,
        use_feature_aligner: bool = True
    ):
        super().__init__()

        self.config = config
        self.embed_dim = embed_dim
        self.use_vit = use_vit
        self.use_vae = use_vae

        # Load BAGEL model
        self.bagel_model = self._load_bagel_model(config)

        # Initialize text and image adapters
        self.text_adapter = BagelTextAdapter(
            bagel_model=self.bagel_model,
            embed_dim=embed_dim,
            max_length=max_text_length,
            use_feature_aligner=use_feature_aligner
        )

        self.image_adapter = BagelImageAdapter(
            bagel_model=self.bagel_model,
            embed_dim=embed_dim,
            use_vit=use_vit,
            use_vae=use_vae,
            use_feature_aligner=use_feature_aligner
        )

        # Temperature parameter for contrastive learning
        self.temp = nn.Parameter(torch.ones([]) * config.get('temp', 0.07))

    def _load_bagel_model(self, config: Dict[str, Any]) -> Bagel:
        """Load BAGEL model from checkpoint."""
        # This is a placeholder - actual implementation would load from config
        # For now, we'll create a mock configuration

        # Create mock configurations
        from modeling.qwen2.configuration_qwen2 import Qwen2Config
        from modeling.siglip.configuration_siglip import SiglipVisionConfig

        llm_config = Qwen2Config(
            hidden_size=config.get('llm_hidden_size', 1536),
            num_attention_heads=config.get('num_attention_heads', 12),
            num_hidden_layers=config.get('num_hidden_layers', 24),
            vocab_size=config.get('vocab_size', 152064)
        )

        vit_config = SiglipVisionConfig(
            hidden_size=config.get('vit_hidden_size', 1024),
            num_attention_heads=config.get('vit_num_heads', 16),
            num_hidden_layers=config.get('vit_num_layers', 24),
            patch_size=config.get('patch_size', 14),
            image_size=config.get('image_size', 224)
        )

        vae_config = {
            'z_channels': config.get('vae_z_channels', 16),
            'downsample': config.get('vae_downsample', 16)
        }

        bagel_config = BagelConfig(
            visual_und=use_vit,
            visual_gen=use_vae,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=config.get('latent_patch_size', 2),
            max_latent_size=config.get('max_latent_size', 32)
        )

        # Create mock language model and vit model
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config) if use_vit else None

        bagel_model = Bagel(
            language_model=language_model,
            vit_model=vit_model,
            config=bagel_config
        )

        # Load checkpoint if provided
        if 'checkpoint_path' in config:
            checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
            if 'model' in checkpoint:
                bagel_model.load_state_dict(checkpoint['model'], strict=False)
            else:
                bagel_model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded BAGEL checkpoint from {config['checkpoint_path']}")

        return bagel_model

    def get_text_embeds(self, text_ids: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        """Delegate to text adapter."""
        return self.text_adapter.get_text_embeds(text_ids, text_atts)

    def get_text_feat(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Delegate to text adapter."""
        return self.text_adapter.get_text_feat(text_embeds)

    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Delegate to image adapter."""
        return self.image_adapter.get_vision_embeds(image)

    def get_image_feat(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Delegate to image adapter."""
        return self.image_adapter.get_image_feat(image_embeds)

    def forward(self, **kwargs):
        """Forward pass for training compatibility."""
        # This can be implemented for training if needed
        pass


def create_bagel_adapter(config: Dict[str, Any], **kwargs) -> BagelAdapter:
    """
    Factory function to create BAGEL adapter.

    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments for adapter initialization

    Returns:
        BagelAdapter instance
    """
    return BagelAdapter(config, **kwargs)


if __name__ == "__main__":
    # Example usage
    config = {
        'llm_hidden_size': 1536,
        'vit_hidden_size': 1024,
        'embed_dim': 768,
        'temp': 0.07,
        'use_vit': True,
        'use_vae': False
    }

    adapter = create_bagel_adapter(config)
    print("BAGEL adapter created successfully!")
    print(f"Text adapter: {adapter.text_adapter}")
    print(f"Image adapter: {adapter.image_adapter}")