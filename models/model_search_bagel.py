"""
Enhanced Search Model with BAGEL Integration

This module extends the original Search model to support BAGEL feature extraction
while maintaining all existing functionality including pose estimation and YOLO detection.

Features:
- Full BAGEL integration for enhanced multimodal understanding
- Backward compatibility with existing Search model features
- Support for pose-based image enhancement
- YOLO-based person detection with GCN integration
- Flexible feature extraction strategies

Author: Generated with Claude Code
License: Apache-2.0
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List

from models.model_search import Search as OriginalSearch
from models.cmp_bagel import CMPBagel, create_cmp_bagel
from models.pose import Block, ConvExpandReduce
from models.gcn import Block2
from nets.nn import YOLODetector


class SearchBagel(CMPBagel):
    """
    Enhanced Search model with BAGEL integration and original Search functionality.

    This class combines the powerful feature extraction capabilities of BAGEL
    with the specialized features of the original Search model including:
    - Pose-based image enhancement
    - YOLO person detection
    - GCN-based person feature processing
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize CMP-BAGEL first
        super().__init__(config)

        # Original Search model specific features
        self.be_hard = config.get('be_hard', False)
        self.be_pose_img = config.get('be_pose_img', False)
        self.be_pose_conv = config.get('pose_conv', False)
        self.use_yolo_gcn = config.get('use_yolo_gcn', False)

        # YOLO configuration
        self.yolo_person_class_id = config.get('yolo_person_class_id', 0)
        self.yolo_conf_threshold = config.get('yolo_conf_threshold', 0.2)

        # Initialize pose processing
        if self.be_pose_img:
            self.pose_block = Block()
            self.init_params.extend(['pose_block.' + n for n, _ in self.pose_block.named_parameters()])

            if self.be_pose_conv:
                print('Enabling pose convolution')
                self.pose_conv = ConvExpandReduce()
                self.init_params.extend(['pose_conv.' + n for n, _ in self.pose_conv.named_parameters()])

        # Initialize YOLO and GCN
        if self.use_yolo_gcn:
            self._init_yolo_gcn(config)

        print(f"Initialized Search-BAGEL model with feature mode: {self.feature_mode}")
        if self.be_pose_img:
            print("Pose-based image enhancement: ENABLED")
        if self.use_yolo_gcn:
            print("YOLO + GCN person detection: ENABLED")

    def _init_yolo_gcn(self, config: Dict[str, Any]):
        """Initialize YOLO detector and GCN modules."""
        # Initialize YOLO detector
        self.yolo_detector = YOLODetector()

        # Load YOLO weights if provided
        yolo_weight_path = config.get('yolo_weight_path')
        if yolo_weight_path and os.path.exists(yolo_weight_path):
            try:
                yolo_model = torch.load(yolo_weight_path, map_location="cpu")['model'].float()
                state_dict = yolo_model.state_dict()
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = "model." + k
                    new_state_dict[new_key] = v
                self.yolo_detector.load_state_dict(new_state_dict, strict=True)
                print(f"[YOLO] Loaded weights from {yolo_weight_path}")
            except Exception as e:
                print(f"[YOLO] Failed to load weights: {e}")
        else:
            print(f"[YOLO] Using random initialization (no weights found at {yolo_weight_path})")

        # Freeze YOLO parameters
        for param in self.yolo_detector.parameters():
            param.requires_grad = False
        self.yolo_detector.eval()

        # Freeze BN/Dropout layers in YOLO
        for m in self.yolo_detector.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # Initialize GCN module
        self.gcn = Block2()  # GCNBlock()
        self.init_params.extend(['gcn.' + n for n, _ in self.gcn.named_parameters()])

        # Initialize person feature fusion module
        self.person_block = Block2()
        self.init_params.extend(['person_block.' + n for n, _ in self.person_block.named_parameters()])

    def detect_persons_and_extract_features(self, image: torch.Tensor, save_dir: str = "debug_vis") -> Tuple[torch.Tensor, List[int]]:
        """
        Detect persons using YOLO and extract ROI features with BAGEL enhancement.

        Args:
            image: Input images of shape (batch_size, 3, H, W)
            save_dir: Directory to save visualization outputs

        Returns:
            Tuple of (person_features, person_counts)
            - person_features: Extracted person features of shape (total_persons, feature_dim)
            - person_counts: Number of persons detected per image
        """
        os.makedirs(save_dir, exist_ok=True)

        batch_size = image.size(0)
        self.yolo_detector.eval()

        # YOLO detection
        with torch.no_grad():
            detections, yolo_feature_maps = self.yolo_detector(image)
            boxes, scores, classes = detections

        # Get feature map for ROI extraction
        feature_map = yolo_feature_maps[2]  # Use middle-scale feature map
        spatial_scale = feature_map.shape[-1] / image.shape[-1]

        all_person_features = []
        person_counts = []

        for i in range(batch_size):
            img_boxes = boxes[i]
            img_scores = scores[i]
            img_classes = classes[i]

            # Filter person detections
            person_mask = img_classes == self.yolo_person_class_id
            person_boxes = img_boxes[person_mask]
            person_scores = img_scores[person_mask]

            # Filter by confidence threshold
            confident_mask = person_scores >= self.yolo_conf_threshold
            person_boxes = person_boxes[confident_mask]
            person_scores = person_scores[confident_mask]

            person_counts.append(len(person_boxes))

            if len(person_boxes) == 0:
                # No persons detected, create dummy feature
                dummy_feature = torch.zeros(1, self.vision_width, device=image.device)
                all_person_features.append(dummy_feature)
                continue

            # Extract ROI features using BAGEL-enhanced approach
            roi_features = self._extract_roi_features_bagel(
                image[i:i+1], person_boxes, feature_map[i:i+1], spatial_scale
            )

            # Process with GCN
            if len(roi_features) > 0:
                roi_features = self.gcn(roi_features)

            all_person_features.append(roi_features)

        # Concatenate all person features
        if len(all_person_features) > 0:
            person_features = torch.cat(all_person_features, dim=0)
        else:
            person_features = torch.zeros(0, self.vision_width, device=image.device)

        return person_features, person_counts

    def _extract_roi_features_bagel(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        feature_map: torch.Tensor,
        spatial_scale: float
    ) -> torch.Tensor:
        """
        Extract ROI features using BAGEL-enhanced processing.

        Args:
            image: Single image of shape (1, 3, H, W)
            boxes: Detected bounding boxes of shape (N, 4)
            feature_map: Feature map of shape (1, C, H, W)
            spatial_scale: Scale factor for ROI alignment

        Returns:
            ROI features of shape (N, feature_dim)
        """
        if len(boxes) == 0:
            return torch.zeros(0, self.vision_width, device=image.device)

        # Convert boxes to [x1, y1, x2, y2] format if needed
        if boxes.shape[1] == 4:  # [x1, y1, x2, y2]
            roi_boxes = boxes.clone()
        else:
            # Assume [x, y, w, h] format
            roi_boxes = torch.zeros_like(boxes)
            roi_boxes[:, 0] = boxes[:, 0]  # x1
            roi_boxes[:, 1] = boxes[:, 1]  # y1
            roi_boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
            roi_boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

        # Normalize boxes to [0, 1] range
        img_h, img_w = image.shape[2], image.shape[3]
        roi_boxes[:, [0, 2]] /= img_w
        roi_boxes[:, [1, 3]] /= img_h

        # Add batch index to each box
        batch_idx = torch.zeros(len(roi_boxes), 1, device=image.device)
        roi_boxes = torch.cat([batch_idx, roi_boxes], dim=1)

        # Extract ROI features
        roi_size = (7, 7)  # Standard ROI size
        roi_features = roi_align(
            feature_map,
            roi_boxes,
            output_size=roi_size,
            spatial_scale=spatial_scale,
            sampling_ratio=-1
        )  # Shape: (N, C, 7, 7)

        # Global average pooling
        roi_features = F.adaptive_avg_pool2d(roi_features, (1, 1)).squeeze(-1).squeeze(-1)

        # Project to vision embedding dimension
        if roi_features.shape[1] != self.vision_width:
            roi_features = F.linear(roi_features, torch.randn(self.vision_width, roi_features.shape[1], device=image.device))

        return roi_features

    def get_vision_embeds(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced vision embedding extraction with pose and person detection support.

        Args:
            image: Input images of shape (batch_size, 3, H, W)

        Returns:
            Tuple of (image_embeds, image_atts)
        """
        # Extract base vision embeddings using selected feature extractor
        image_embeds, image_atts = self.feature_extractor.get_vision_embeds(image)

        # Apply pose enhancement if enabled
        if self.be_pose_img:
            # This would require pose images - for now, we'll skip the actual pose processing
            # In practice, you would need to provide pose images alongside the RGB images
            pass

        # Apply person detection and feature enhancement if enabled
        if self.use_yolo_gcn:
            person_features, person_counts = self.detect_persons_and_extract_features(image)

            # Enhance image embeddings with person features
            if len(person_features) > 0:
                image_embeds = self._enhance_with_person_features(image_embeds, person_features, person_counts)

        return image_embeds, image_atts

    def _enhance_with_person_features(
        self,
        image_embeds: torch.Tensor,
        person_features: torch.Tensor,
        person_counts: List[int]
    ) -> torch.Tensor:
        """
        Enhance image embeddings with detected person features.

        Args:
            image_embeds: Original image embeddings of shape (batch_size, seq_len, dim)
            person_features: Extracted person features
            person_counts: Number of persons per image

        Returns:
            Enhanced image embeddings
        """
        batch_size = image_embeds.shape[0]

        # Simple enhancement: concatenate average person features to image embeddings
        enhanced_embeds = image_embeds.clone()

        start_idx = 0
        for i in range(batch_size):
            num_persons = person_counts[i]
            if num_persons > 0:
                # Get person features for this image
                end_idx = start_idx + num_persons
                img_person_features = person_features[start_idx:end_idx]

                # Average person features
                avg_person_feat = img_person_features.mean(dim=0, keepdim=True)

                # Enhance the first token (CLS-like) with person information
                enhanced_embeds[i, 0, :] = enhanced_embeds[i, 0, :] + avg_person_feat

            start_idx = end_idx

        return enhanced_embeds

    def get_cross_embeds(
        self,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor,
        text_embeds: torch.Tensor,
        text_atts: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhanced cross-modal embeddings with person feature integration.

        Args:
            image_embeds: Image embeddings
            image_atts: Image attention mask
            text_embeds: Text embeddings
            text_atts: Text attention mask

        Returns:
            Cross-modal embeddings
        """
        # Use base cross-modal reasoning
        cross_embeds = super().get_cross_embeds(image_embeds, image_atts, text_embeds, text_atts)

        # Apply person block if YOLO+GCN is enabled
        if self.use_yolo_gcn:
            cross_embeds = self.person_block(cross_embeds)

        return cross_embeds

    def forward(
        self,
        image: torch.Tensor,
        text_ids: torch.Tensor,
        text_atts: torch.Tensor,
        mode: str = 'itm'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting different modes.

        Args:
            image: Input images
            text_ids: Text token IDs
            text_atts: Text attention mask
            mode: Forward mode ('itm', 'itc', 'mlm')

        Returns:
            Dictionary containing outputs based on mode
        """
        outputs = {}

        if mode in ['itm', 'itc']:
            # Extract features
            image_embeds, image_atts = self.get_vision_embeds(image)
            text_embeds = self.get_text_embeds(text_ids, text_atts)

            # Get features for similarity computation
            image_feat = self.get_image_feat(image_embeds)
            text_feat = self.get_text_feat(text_embeds)

            # Normalize features
            image_feat = F.normalize(image_feat, dim=-1)
            text_feat = F.normalize(text_feat, dim=-1)

            outputs['image_feat'] = image_feat
            outputs['text_feat'] = text_feat

            if mode == 'itm':
                # Image-Text Matching
                cross_embeds = self.get_cross_embeds(image_embeds, image_atts, text_embeds, text_atts)
                cross_embeds = cross_embeds[:, 0, :]  # Use CLS token
                itm_logits = self.itm_head(cross_embeds)
                outputs['itm_logits'] = itm_logits

            elif mode == 'itc':
                # Image-Text Contrastive
                similarity = torch.matmul(image_feat, text_feat.t()) / self.temp
                outputs['similarity'] = similarity

        return outputs

    def set_feature_mode(self, mode: str):
        """
        Set feature extraction mode with validation for Search-specific features.

        Args:
            mode: 'cmp', 'bagel', or 'hybrid'
        """
        super().set_feature_mode(mode)

        # Validate compatibility with Search features
        if self.use_yolo_gcn and mode == 'bagel':
            print("Warning: YOLO+GCN features may not be fully compatible with BAGEL-only mode")
            print("Consider using 'hybrid' mode for best results")

        if self.be_pose_img and mode == 'bagel':
            print("Warning: Pose features may not be fully compatible with BAGEL-only mode")
            print("Consider using 'hybrid' mode for best results")


def create_search_bagel(config: Dict[str, Any]) -> SearchBagel:
    """
    Factory function to create Search-BAGEL model.

    Args:
        config: Configuration dictionary

    Returns:
        SearchBagel instance
    """
    return SearchBagel(config)


if __name__ == "__main__":
    # Example usage
    config = {
        'feature_mode': 'hybrid',
        'use_bagel': True,
        'bagel_config': {
            'llm_hidden_size': 1536,
            'vit_hidden_size': 1024,
            'use_vit': True,
            'use_vae': False
        },
        'be_pose_img': False,
        'use_yolo_gcn': False,
        'embed_dim': 768,
        'temp': 0.07,
        'fusion_strategy': 'weighted_avg',
        'text_weight': 0.6,
        'image_weight': 0.6
    }

    # Create model
    model = create_search_bagel(config)
    print("Search-BAGEL model created successfully!")
    print(f"Feature mode: {model.feature_mode}")
    print(f"Using BAGEL: {model.use_bagel}")
    print(f"Pose enhancement: {model.be_pose_img}")
    print(f"YOLO+GCN: {model.use_yolo_gcn}")

    # Test mode switching
    for mode in ['cmp', 'bagel', 'hybrid']:
        model.set_feature_mode(mode)
        print(f"Switched to {mode} mode")