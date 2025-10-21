"""
Enhanced CMP Model with BAGEL Integration

This module extends the original CMP model to support BAGEL feature extraction
while maintaining backward compatibility with existing training and evaluation pipelines.

Features:
- Seamless integration of BAGEL's powerful multimodal features
- Backward compatibility with existing CMP workflow
- Flexible feature extraction strategies (BAGEL-only, hybrid, original)
- Support for both training and inference modes

Author: Generated with Claude Code
License: Apache-2.0
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod

from models.cmp import CMP, AllGather, allgather, build_vision_encoder, build_text_encoder, feature_mapping
from models.bagel_adapter import BagelAdapter, create_bagel_adapter
from utils import read_json


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def get_text_embeds(self, text_ids: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_text_feat(self, text_embeds: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_image_feat(self, image_embeds: torch.Tensor) -> torch.Tensor:
        pass


class CMPFeatureExtractor(BaseFeatureExtractor):
    """Original CMP feature extractor wrapper."""

    def __init__(self, cmp_model: CMP):
        self.cmp_model = cmp_model

    def get_text_embeds(self, text_ids: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        return self.cmp_model.get_text_embeds(text_ids, text_atts)

    def get_text_feat(self, text_embeds: torch.Tensor) -> torch.Tensor:
        return self.cmp_model.get_text_feat(text_embeds)

    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cmp_model.get_vision_embeds(image)

    def get_image_feat(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.cmp_model.get_image_feat(image_embeds)


class BagelFeatureExtractor(BaseFeatureExtractor):
    """BAGEL feature extractor wrapper."""

    def __init__(self, bagel_adapter: BagelAdapter):
        self.bagel_adapter = bagel_adapter

    def get_text_embeds(self, text_ids: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        return self.bagel_adapter.get_text_embeds(text_ids, text_atts)

    def get_text_feat(self, text_embeds: torch.Tensor) -> torch.Tensor:
        return self.bagel_adapter.get_text_feat(text_embeds)

    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bagel_adapter.get_vision_embeds(image)

    def get_image_feat(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.bagel_adapter.get_image_feat(image_embeds)


class HybridFeatureExtractor(BaseFeatureExtractor):
    """Hybrid feature extractor that combines CMP and BAGEL features."""

    def __init__(
        self,
        cmp_extractor: CMPFeatureExtractor,
        bagel_extractor: BagelFeatureExtractor,
        fusion_strategy: str = 'concat',
        text_weight: float = 0.5,
        image_weight: float = 0.5
    ):
        self.cmp_extractor = cmp_extractor
        self.bagel_extractor = bagel_extractor
        self.fusion_strategy = fusion_strategy
        self.text_weight = text_weight
        self.image_weight = image_weight

    def get_text_embeds(self, text_ids: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        cmp_embeds = self.cmp_extractor.get_text_embeds(text_ids, text_atts)
        bagel_embeds = self.bagel_extractor.get_text_embeds(text_ids, text_atts)

        if self.fusion_strategy == 'concat':
            return torch.cat([cmp_embeds, bagel_embeds], dim=-1)
        elif self.fusion_strategy == 'weighted_avg':
            return self.text_weight * cmp_embeds + (1 - self.text_weight) * bagel_embeds
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def get_text_feat(self, text_embeds: torch.Tensor) -> torch.Tensor:
        # Split fused embeddings if necessary
        if self.fusion_strategy == 'concat':
            mid = text_embeds.shape[-1] // 2
            cmp_embeds = text_embeds[..., :mid]
            bagel_embeds = text_embeds[..., mid:]
        else:
            cmp_embeds = text_embeds
            bagel_embeds = text_embeds

        cmp_feat = self.cmp_extractor.get_text_feat(cmp_embeds)
        bagel_feat = self.bagel_extractor.get_text_feat(bagel_embeds)

        if self.fusion_strategy == 'concat':
            return torch.cat([cmp_feat, bagel_feat], dim=-1)
        elif self.fusion_strategy == 'weighted_avg':
            return self.text_weight * cmp_feat + (1 - self.text_weight) * bagel_feat

    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cmp_embeds, cmp_atts = self.cmp_extractor.get_vision_embeds(image)
        bagel_embeds, bagel_atts = self.bagel_extractor.get_vision_embeds(image)

        if self.fusion_strategy == 'concat':
            return torch.cat([cmp_embeds, bagel_embeds], dim=-1), torch.cat([cmp_atts, bagel_atts], dim=-1)
        elif self.fusion_strategy == 'weighted_avg':
            return (self.image_weight * cmp_embeds + (1 - self.image_weight) * bagel_embeds,
                   torch.max(cmp_atts, bagel_atts))  # Use max for attention masks

    def get_image_feat(self, image_embeds: torch.Tensor) -> torch.Tensor:
        # Split fused embeddings if necessary
        if self.fusion_strategy == 'concat':
            mid = image_embeds.shape[-1] // 2
            cmp_embeds = image_embeds[..., :mid]
            bagel_embeds = image_embeds[..., mid:]
        else:
            cmp_embeds = image_embeds
            bagel_embeds = image_embeds

        cmp_feat = self.cmp_extractor.get_image_feat(cmp_embeds)
        bagel_feat = self.bagel_extractor.get_image_feat(bagel_embeds)

        if self.fusion_strategy == 'concat':
            return torch.cat([cmp_feat, bagel_feat], dim=-1)
        elif self.fusion_strategy == 'weighted_avg':
            return self.image_weight * cmp_feat + (1 - self.image_weight) * bagel_feat


class CMPBagel(CMP):
    """
    Enhanced CMP model with BAGEL integration support.

    This class extends the original CMP model to support multiple feature extraction
    strategies while maintaining full backward compatibility.
    """

    def __init__(self, config=None):
        super().__init__(config)

        # Feature extraction mode
        self.feature_mode = config.get('feature_mode', 'cmp')  # 'cmp', 'bagel', 'hybrid'

        # Initialize feature extractors
        self.cmp_extractor = CMPFeatureExtractor(self)
        self.feature_extractor = self.cmp_extractor  # Default to CMP

        # BAGEL configuration
        self.use_bagel = config.get('use_bagel', False)
        self.bagel_config = config.get('bagel_config', {})

        # Initialize BAGEL adapter if needed
        if self.use_bagel and self.feature_mode in ['bagel', 'hybrid']:
            self._init_bagel_adapter()

        # Initialize hybrid extractor if needed
        if self.feature_mode == 'hybrid':
            bagel_extractor = BagelFeatureExtractor(self.bagel_adapter)
            self.feature_extractor = HybridFeatureExtractor(
                cmp_extractor=self.cmp_extractor,
                bagel_extractor=bagel_extractor,
                fusion_strategy=config.get('fusion_strategy', 'weighted_avg'),
                text_weight=config.get('text_weight', 0.5),
                image_weight=config.get('image_weight', 0.5)
            )

            # Adjust projection layers for fused features
            self._adjust_projections_for_hybrid(config)

        print(f"Initialized CMP-BAGEL model with feature mode: {self.feature_mode}")

    def _init_bagel_adapter(self):
        """Initialize BAGEL adapter."""
        # Create BAGEL configuration
        bagel_config = {
            'llm_hidden_size': self.bagel_config.get('llm_hidden_size', 1536),
            'vit_hidden_size': self.bagel_config.get('vit_hidden_size', 1024),
            'embed_dim': self.embed_dim,
            'temp': self.temp.item() if hasattr(self.temp, 'item') else self.temp,
            'use_vit': self.bagel_config.get('use_vit', True),
            'use_vae': self.bagel_config.get('use_vae', False),
            'checkpoint_path': self.bagel_config.get('checkpoint_path', None)
        }

        # Add BAGEL model parameters
        if 'bagel_model_params' in self.bagel_config:
            bagel_config.update(self.bagel_config['bagel_model_params'])

        self.bagel_adapter = create_bagel_adapter(bagel_config)

        if self.feature_mode == 'bagel':
            self.feature_extractor = BagelFeatureExtractor(self.bagel_adapter)

    def _adjust_projections_for_hybrid(self, config):
        """Adjust projection layers for hybrid features."""
        if config.get('fusion_strategy') == 'concat':
            # Double the input dimension for projections
            original_vision_proj = self.vision_proj
            original_text_proj = self.text_proj

            self.vision_proj = nn.Linear(
                original_vision_proj.in_features * 2,
                self.embed_dim
            )
            self.text_proj = nn.Linear(
                original_text_proj.in_features * 2,
                self.embed_dim
            )

            # Initialize new weights
            nn.init.normal_(self.vision_proj.weight, std=0.02)
            nn.init.constant_(self.vision_proj.bias, 0.0)
            nn.init.normal_(self.text_proj.weight, std=0.02)
            nn.init.constant_(self.text_proj.bias, 0.0)

    def set_feature_mode(self, mode: str):
        """
        Dynamically change feature extraction mode.

        Args:
            mode: 'cmp', 'bagel', or 'hybrid'
        """
        if mode not in ['cmp', 'bagel', 'hybrid']:
            raise ValueError(f"Invalid feature mode: {mode}. Must be 'cmp', 'bagel', or 'hybrid'")

        if mode == 'hybrid' and not hasattr(self, 'bagel_adapter'):
            raise ValueError("BAGEL adapter not initialized. Set use_bagel=True in config.")

        if mode == 'bagel' and not hasattr(self, 'bagel_adapter'):
            raise ValueError("BAGEL adapter not initialized. Set use_bagel=True in config.")

        self.feature_mode = mode

        if mode == 'cmp':
            self.feature_extractor = self.cmp_extractor
        elif mode == 'bagel':
            self.feature_extractor = BagelFeatureExtractor(self.bagel_adapter)
        elif mode == 'hybrid':
            bagel_extractor = BagelFeatureExtractor(self.bagel_adapter)
            self.feature_extractor = HybridFeatureExtractor(
                cmp_extractor=self.cmp_extractor,
                bagel_extractor=bagel_extractor
            )

        print(f"Switched to feature mode: {mode}")

    # Override feature extraction methods to use the selected extractor
    def get_text_embeds(self, text_ids: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        """Extract text embeddings using the selected feature extractor."""
        return self.feature_extractor.get_text_embeds(text_ids, text_atts)

    def get_text_feat(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Extract text features using the selected feature extractor."""
        return self.feature_extractor.get_text_feat(text_embeds)

    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract vision embeddings using the selected feature extractor."""
        return self.feature_extractor.get_vision_embeds(image)

    def get_image_feat(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Extract image features using the selected feature extractor."""
        return self.feature_extractor.get_image_feat(image_embeds)

    def get_cross_embeds(self, image_embeds: torch.Tensor, image_atts: torch.Tensor,
                        text_embeds: torch.Tensor, text_atts: torch.Tensor) -> torch.Tensor:
        """
        Get cross-modal embeddings.
        Note: This method uses the original CMP cross-encoder.
        BAGEL's cross-modal reasoning is handled differently.
        """
        # For now, use CMP's cross-encoder
        # Future enhancement could implement BAGEL-based cross-modal reasoning
        return super().get_cross_embeds(image_embeds, image_atts, text_embeds, text_atts)

    def load_pretrained(self, ckpt_rpath: str):
        """Load pretrained checkpoint."""
        checkpoint = torch.load(ckpt_rpath, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

        # Load CMP components
        msg = super().load_pretrained(ckpt_rpath)

        # Load BAGEL adapter if exists
        if hasattr(self, 'bagel_adapter') and 'bagel_adapter' in state_dict:
            try:
                self.bagel_adapter.load_state_dict(state_dict['bagel_adapter'], strict=False)
                print("Loaded BAGEL adapter state dict")
            except Exception as e:
                print(f"Failed to load BAGEL adapter: {e}")

        return msg

    def save_checkpoint(self, path: str):
        """Save checkpoint including BAGEL adapter."""
        checkpoint = {
            'model': self.state_dict(),
            'config': self.config if hasattr(self, 'config') else {},
            'feature_mode': self.feature_mode
        }

        if hasattr(self, 'bagel_adapter'):
            checkpoint['bagel_adapter'] = self.bagel_adapter.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def create_cmp_bagel(config: Dict[str, Any]) -> CMPBagel:
    """
    Factory function to create CMP-BAGEL model.

    Args:
        config: Configuration dictionary

    Returns:
        CMPBagel instance
    """
    return CMPBagel(config)


if __name__ == "__main__":
    # Example usage
    config = {
        'feature_mode': 'hybrid',
        'use_bagel': True,
        'bagel_config': {
            'llm_hidden_size': 1536,
            'vit_hidden_size': 1024,
            'use_vit': True,
            'use_vae': False,
            'checkpoint_path': None  # Add path to BAGEL checkpoint
        },
        'fusion_strategy': 'weighted_avg',
        'text_weight': 0.6,
        'image_weight': 0.6,
        'embed_dim': 768,
        'temp': 0.07,
        'vision_config': 'path/to/vision_config.json',
        'text_config': 'path/to/text_config.json',
        'text_encoder': 'bert-base-uncased',
        'load_params': True
    }

    # Create model
    model = create_cmp_bagel(config)
    print("CMP-BAGEL model created successfully!")
    print(f"Feature mode: {model.feature_mode}")
    print(f"Using BAGEL: {model.use_bagel}")

    # Test mode switching
    model.set_feature_mode('cmp')
    model.set_feature_mode('bagel')
    model.set_feature_mode('hybrid')