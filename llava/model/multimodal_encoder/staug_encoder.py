import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Sequence, Tuple, Union

from transformers import PretrainedConfig, SiglipImageProcessor, SiglipVisionModel

from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    

class PureAttentionPoolingBlock(nn.Module):
    """
    Just a pure attn_pooling implementation, without ln_post, without projection, no mormalized_final
    """

    def __init__(
            self,
            context_dim: int,
            n_head: int = 8,
            norm_layer: Callable = LayerNorm,
            need_weights: bool = False
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(context_dim, n_head, kdim=context_dim, vdim=context_dim, batch_first=True,
                                          add_zero_attn=True)
        self.ln_q = norm_layer(context_dim)
        self.ln_k = norm_layer(context_dim)
        self.ln_v = norm_layer(context_dim)
        self.need_weights=need_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output_attn_weights=False, average_attn_weights=True):
        batch_size, seg_length, embed_dim = k.size()
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        if self.need_weights or output_attn_weights:
            out, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=average_attn_weights)
            return F.normalize(out, dim=-1), attn_weights
        else:
            out = self.attn(q, k, v, need_weights=False)[0]
            return F.normalize(out, dim=-1)

class STAugVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        model_name_or_path = "google/siglip-so400m-patch14-384" # hard-coded
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            torch_dtype=eval(config.model_dtype),
            state_dict=state_dict,
        )
        self.is_loaded = True

        # Initialize attention pooling components
        hidden_size = self.vision_tower.config.hidden_size
        self.attn_pooling = PureAttentionPoolingBlock(context_dim=hidden_size)

    def forward(self, images):
        """
        Custom forward pass for STAugVisionTower with enhanced processing.
        """
        if type(images) is list:
            image_features = []
            for image in images:
                # Process each image individually
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)

                # Apply additional spatial augmentation processing here
                # This is where you would add your custom processing logic
                image_feature = self._apply_spatial_augmentation(image_feature)

                image_features.append(image_feature)
        else:
            # Process batch of images
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            # Apply additional spatial augmentation processing here
            image_features = self._apply_spatial_augmentation(image_features)

        return image_features

    def _apply_spatial_augmentation(self, features):
        """
        Apply spatial augmentation to the features.
        This is a placeholder for your custom spatial augmentation logic.

        Args:
            features: Image features from the vision tower

        Returns:
            Augmented features
        """
        # This is where you would implement your custom spatial augmentation
        # For now, we'll just return the original features
        # You can replace this with your actual implementation

        # Example of a simple augmentation (identity for now):
        return features

    def attention_pooling(self, image_features, text_features=None, pooling_ratio=0.5):
        if text_features is None:
            return image_features
        else:
            pooled_img_feature = self.attn_pooling(image_features, text_features)
            return pooled_img_feature

        # """
        # Apply attention pooling to reduce the number of image tokens.

        # Args:
        #     image_features: Image features from the vision tower [B, N_img, D]
        #     text_features: Text features to use as queries [B, N_text, D]
        #                   If None, use a learned query vector
        #     pooling_ratio: Ratio of tokens to keep (0.5 means reduce by half)

        # Returns:
        #     Pooled image features with reduced tokens
        # """
        # batch_size, num_img_tokens, hidden_size = image_features.shape
        # num_tokens_to_keep = max(1, int(num_img_tokens * pooling_ratio))

        # # If no text features provided, use a learned query or the first token
        # if text_features is None:
        #     # Use the first token (CLS) as the query
        #     queries = self.query_proj(image_features[:, 0:1, :])
        # else:
        #     # Use text features as queries
        #     queries = self.query_proj(text_features)

        # # Project image features to keys and values
        # keys = self.key_proj(image_features)
        # values = self.value_proj(image_features)

        # # Compute attention scores
        # attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.attention_scale

        # # Get the top-k attention scores for each query
        # # Sum across all queries to get importance of each image token
        # token_importance = attention_scores.sum(dim=1)  # [B, N_img]

        # # Select top-k tokens based on importance
        # _, top_indices = torch.topk(token_importance, num_tokens_to_keep, dim=1)
        # top_indices = top_indices.sort(dim=1)[0]  # Sort indices to maintain spatial order

        # # Gather the top-k tokens for each batch
        # pooled_features = torch.stack([
        #     image_features[b, top_indices[b]] for b in range(batch_size)
        # ])

        # return pooled_features

# TODO @lx support S2
class STAugVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(model_name_or_path, torch_dtype=eval(config.model_dtype))
        
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[-1]

        self.is_loaded = True

        # Initialize attention pooling components
        hidden_size = self.vision_tower.config.hidden_size
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.attention_scale = hidden_size ** -0.5

    @torch.no_grad()
    def forward_feature(self, images):
        """
        Enhanced forward_feature method with spatial augmentation.
        """
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # Apply spatial augmentation
        image_features = self._apply_spatial_augmentation(image_features)

        return image_features

    def _apply_spatial_augmentation(self, features):
        """
        Apply spatial augmentation to the features.
        This is a placeholder for your custom spatial augmentation logic.

        Args:
            features: Image features from the vision tower

        Returns:
            Augmented features
        """
        # Implement your custom spatial augmentation here
        # For now, we'll just return the original features
        return features

    def attention_pooling(self, image_features, text_features=None, pooling_ratio=0.5):
        """
        Apply attention pooling to reduce the number of image tokens.

        Args:
            image_features: Image features from the vision tower [B, N_img, D]
            text_features: Text features to use as queries [B, N_text, D]
                          If None, use a learned query vector
            pooling_ratio: Ratio of tokens to keep (0.5 means reduce by half)

        Returns:
            Pooled image features with reduced tokens
        """
        batch_size, num_img_tokens, hidden_size = image_features.shape
        num_tokens_to_keep = max(1, int(num_img_tokens * pooling_ratio))

        # If no text features provided, use a learned query or the first token
        if text_features is None:
            # Use the first token (CLS) as the query
            queries = self.query_proj(image_features[:, 0:1, :])
        else:
            # Use text features as queries
            queries = self.query_proj(text_features)

        # Project image features to keys and values
        keys = self.key_proj(image_features)
        values = self.value_proj(image_features)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.attention_scale

        # Get the top-k attention scores for each query
        # Sum across all queries to get importance of each image token
        token_importance = attention_scores.sum(dim=1)  # [B, N_img]

        # Select top-k tokens based on importance
        _, top_indices = torch.topk(token_importance, num_tokens_to_keep, dim=1)
        top_indices = top_indices.sort(dim=1)[0]  # Sort indices to maintain spatial order

        # Gather the top-k tokens for each batch
        pooled_features = torch.stack([
            image_features[b, top_indices[b]] for b in range(batch_size)
        ])

        return pooled_features
