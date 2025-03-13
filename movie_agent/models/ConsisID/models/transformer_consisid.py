# Copyright 2024 ConsisID Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def ConsisIDFeedForward(dim, mult=4):
    """
    Creates a consistent ID feedforward block consisting of layer normalization, two linear layers, and a GELU
    activation.

    Args:
        dim (int): The input dimension of the tensor.
        mult (int, optional): Multiplier for the inner dimension. Default is 4.

    Returns:
        nn.Sequential: A sequence of layers comprising LayerNorm, Linear layers, and GELU.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    """
    Reshapes the input tensor for multi-head attention.

    Args:
        x (torch.Tensor): The input tensor with shape (batch_size, length, width).
        heads (int): The number of attention heads.

    Returns:
        torch.Tensor: The reshaped tensor, with shape (batch_size, heads, length, width).
    """
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    """
    Implements the Perceiver attention mechanism with multi-head attention.

    This layer takes two inputs: 'x' (image features) and 'latents' (latent features), applying multi-head attention to
    both and producing an output tensor with the same dimension as the input tensor 'x'.

    Args:
        dim (int): The input dimension.
        dim_head (int, optional): The dimension of each attention head. Default is 64.
        heads (int, optional): The number of attention heads. Default is 8.
        kv_dim (int, optional): The key-value dimension. If None, `dim` is used for both keys and values.
    """

    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Forward pass for Perceiver attention.

        Args:
            x (torch.Tensor): Image features tensor with shape (batch_size, num_pixels, D).
            latents (torch.Tensor): Latent features tensor with shape (batch_size, num_latents, D).

        Returns:
            torch.Tensor: Output tensor after applying attention and transformation.
        """
        # Apply normalization
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape  # Get batch size and sequence length

        # Compute query, key, and value matrices
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # Reshape the tensors for multi-head attention
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        # Reshape and return the final output
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class LocalFacialExtractor(nn.Module):
    def __init__(
        self,
        id_dim=1280,
        vit_dim=1024,
        depth=10,
        dim_head=64,
        heads=16,
        num_id_token=5,
        num_queries=32,
        output_dim=2048,
        ff_mult=4,
    ):
        """
        Initializes the LocalFacialExtractor class.

        Parameters:
        - id_dim (int): The dimensionality of id features.
        - vit_dim (int): The dimensionality of vit features.
        - depth (int): Total number of PerceiverAttention and ConsisIDFeedForward layers.
        - dim_head (int): Dimensionality of each attention head.
        - heads (int): Number of attention heads.
        - num_id_token (int): Number of tokens used for identity features.
        - num_queries (int): Number of query tokens for the latent representation.
        - output_dim (int): Output dimension after projection.
        - ff_mult (int): Multiplier for the feed-forward network hidden dimension.
        """
        super().__init__()

        # Storing identity token and query information
        self.num_id_token = num_id_token
        self.vit_dim = vit_dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = vit_dim**-0.5

        # Learnable latent query embeddings
        self.latents = nn.Parameter(torch.randn(1, num_queries, vit_dim) * scale)
        # Projection layer to map the latent output to the desired dimension
        self.proj_out = nn.Parameter(scale * torch.randn(vit_dim, output_dim))

        # Attention and ConsisIDFeedForward layer stack
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=vit_dim, dim_head=dim_head, heads=heads),  # Perceiver Attention layer
                        ConsisIDFeedForward(dim=vit_dim, mult=ff_mult),  # ConsisIDFeedForward layer
                    ]
                )
            )

        # Mappings for each of the 5 different ViT features
        for i in range(5):
            setattr(
                self,
                f"mapping_{i}",
                nn.Sequential(
                    nn.Linear(vit_dim, vit_dim),
                    nn.LayerNorm(vit_dim),
                    nn.LeakyReLU(),
                    nn.Linear(vit_dim, vit_dim),
                    nn.LayerNorm(vit_dim),
                    nn.LeakyReLU(),
                    nn.Linear(vit_dim, vit_dim),
                ),
            )

        # Mapping for identity embedding vectors
        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(id_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim * num_id_token),
        )

    def forward(self, x, y):
        """
        Forward pass for LocalFacialExtractor.

        Parameters:
        - x (Tensor): The input identity embedding tensor of shape (batch_size, id_dim).
        - y (list of Tensor): A list of 5 visual feature tensors each of shape (batch_size, vit_dim).

        Returns:
        - Tensor: The extracted latent features of shape (batch_size, num_queries, output_dim).
        """

        # Repeat latent queries for the batch size
        latents = self.latents.repeat(x.size(0), 1, 1)

        # Map the identity embedding to tokens
        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token, self.vit_dim)

        # Concatenate identity tokens with the latent queries
        latents = torch.cat((latents, x), dim=1)

        # Process each of the 5 visual feature inputs
        for i in range(5):
            vit_feature = getattr(self, f"mapping_{i}")(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)

            # Pass through the PerceiverAttention and ConsisIDFeedForward layers
            for attn, ff in self.layers[i * self.depth : (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        # Retain only the query latents
        latents = latents[:, : self.num_queries]
        # Project the latents to the output dimension
        latents = latents @ self.proj_out
        return latents


class PerceiverCrossAttention(nn.Module):
    """

    Args:
        dim (int): Dimension of the input latent and output. Default is 3072.
        dim_head (int): Dimension of each attention head. Default is 128.
        heads (int): Number of attention heads. Default is 16.
        kv_dim (int): Dimension of the key/value input, allowing flexible cross-attention. Default is 2048.

    Attributes:
        scale (float): Scaling factor used in dot-product attention for numerical stability.
        norm1 (nn.LayerNorm): Layer normalization applied to the input image features.
        norm2 (nn.LayerNorm): Layer normalization applied to the latent features.
        to_q (nn.Linear): Linear layer for projecting the latent features into queries.
        to_kv (nn.Linear): Linear layer for projecting the input features into keys and values.
        to_out (nn.Linear): Linear layer for outputting the final result after attention.

    """

    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """

        Args:
            x (torch.Tensor): Input image features with shape (batch_size, n1, D), where:
                - batch_size (b): Number of samples in the batch.
                - n1: Sequence length (e.g., number of patches or tokens).
                - D: Feature dimension.

            latents (torch.Tensor): Latent feature representations with shape (batch_size, n2, D), where:
                - n2: Number of latent elements.

        Returns:
            torch.Tensor: Attention-modulated features with shape (batch_size, n2, D).

        """
        # Apply layer normalization to the input image and latent features
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        # Compute queries, keys, and values
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Compute attention weights
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable scaling than post-division
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the output via weighted combination of values
        out = weight @ v

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


@maybe_allow_in_graph
class ConsisIDBlock(nn.Module):
    r"""
    Transformer block used in [ConsisID](https://github.com/PKU-YuanGroup/ConsisID) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class ConsisIDTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [ConsisID](https://github.com/PKU-YuanGroup/ConsisID).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because ConsisID processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
        is_train_face (`bool`, defaults to `False`):
            Whether to use enable the identity-preserving module during the training process. When set to `True`, the
            model will focus on identity-preserving tasks.
        is_kps (`bool`, defaults to `False`):
            Whether to enable keypoint for global facial extractor. If `True`, keypoints will be in the model.
        cross_attn_interval (`int`, defaults to `2`):
            The interval between cross-attention layers in the Transformer architecture. A larger value may reduce the
            frequency of cross-attention computations, which can help reduce computational overhead.
        cross_attn_dim_head (`int`, optional, defaults to `128`):
            The dimensionality of each attention head in the cross-attention layers of the Transformer architecture. A
            larger value increases the capacity to attend to more complex patterns, but also increases memory and
            computation costs.
        cross_attn_num_heads (`int`, optional, defaults to `16`):
            The number of attention heads in the cross-attention layers. More heads allow for more parallel attention
            mechanisms, capturing diverse relationships between different components of the input, but can also
            increase computational requirements.
        LFE_id_dim (`int`, optional, defaults to `1280`):
            The dimensionality of the identity vector used in the Local Facial Extractor (LFE). This vector represents
            the identity features of a face, which are important for tasks like face recognition and identity
            preservation across different frames.
        LFE_vit_dim (`int`, optional, defaults to `1024`):
            The dimension of the vision transformer (ViT) output used in the Local Facial Extractor (LFE). This value
            dictates the size of the transformer-generated feature vectors that will be processed for facial feature
            extraction.
        LFE_depth (`int`, optional, defaults to `10`):
            The number of layers in the Local Facial Extractor (LFE). Increasing the depth allows the model to capture
            more complex representations of facial features, but also increases the computational load.
        LFE_dim_head (`int`, optional, defaults to `64`):
            The dimensionality of each attention head in the Local Facial Extractor (LFE). This parameter affects how
            finely the model can process and focus on different parts of the facial features during the extraction
            process.
        LFE_num_heads (`int`, optional, defaults to `16`):
            The number of attention heads in the Local Facial Extractor (LFE). More heads can improve the model's
            ability to capture diverse facial features, but at the cost of increased computational complexity.
        LFE_num_id_token (`int`, optional, defaults to `5`):
            The number of identity tokens used in the Local Facial Extractor (LFE). This defines how many
            identity-related tokens the model will process to ensure face identity preservation during feature
            extraction.
        LFE_num_querie (`int`, optional, defaults to `32`):
            The number of query tokens used in the Local Facial Extractor (LFE). These tokens are used to capture
            high-frequency face-related information that aids in accurate facial feature extraction.
        LFE_output_dim (`int`, optional, defaults to `2048`):
            The output dimension of the Local Facial Extractor (LFE). This dimension determines the size of the feature
            vectors produced by the LFE module, which will be used for subsequent tasks such as face recognition or
            tracking.
        LFE_ff_mult (`int`, optional, defaults to `4`):
            The multiplication factor applied to the feed-forward network's hidden layer size in the Local Facial
            Extractor (LFE). A higher value increases the model's capacity to learn more complex facial feature
            transformations, but also increases the computation and memory requirements.
        local_face_scale (`float`, defaults to `1.0`):
            A scaling factor used to adjust the importance of local facial features in the model. This can influence
            how strongly the model focuses on high frequency face-related content.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        is_train_face: bool = False,
        is_kps: bool = False,
        cross_attn_interval: int = 2,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
        LFE_id_dim: int = 1280,
        LFE_vit_dim: int = 1024,
        LFE_depth: int = 10,
        LFE_dim_head: int = 64,
        LFE_num_heads: int = 16,
        LFE_num_id_token: int = 5,
        LFE_num_querie: int = 32,
        LFE_output_dim: int = 2048,
        LFE_ff_mult: int = 4,
        local_face_scale: float = 1.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no ConsisID checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                ConsisIDBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        self.is_train_face = is_train_face
        self.is_kps = is_kps

        # 5. Define identity-preserving config
        if is_train_face:
            # LFE configs
            self.LFE_id_dim = LFE_id_dim
            self.LFE_vit_dim = LFE_vit_dim
            self.LFE_depth = LFE_depth
            self.LFE_dim_head = LFE_dim_head
            self.LFE_num_heads = LFE_num_heads
            self.LFE_num_id_token = LFE_num_id_token
            self.LFE_num_querie = LFE_num_querie
            self.LFE_output_dim = LFE_output_dim
            self.LFE_ff_mult = LFE_ff_mult
            # cross configs
            self.inner_dim = inner_dim
            self.cross_attn_interval = cross_attn_interval
            self.num_cross_attn = num_layers // cross_attn_interval
            self.cross_attn_dim_head = cross_attn_dim_head
            self.cross_attn_num_heads = cross_attn_num_heads
            self.cross_attn_kv_dim = int(self.inner_dim / 3 * 2)
            self.local_face_scale = local_face_scale
            # face modules
            self._init_face_inputs()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _init_face_inputs(self):
        device = self.device
        weight_dtype = self.dtype
        self.local_facial_extractor = LocalFacialExtractor(
            id_dim=self.LFE_id_dim,
            vit_dim=self.LFE_vit_dim,
            depth=self.LFE_depth,
            dim_head=self.LFE_dim_head,
            heads=self.LFE_num_heads,
            num_id_token=self.LFE_num_id_token,
            num_queries=self.LFE_num_querie,
            output_dim=self.LFE_output_dim,
            ff_mult=self.LFE_ff_mult,
        )
        self.local_facial_extractor.to(device, dtype=weight_dtype)
        self.perceiver_cross_attention = nn.ModuleList(
            [
                PerceiverCrossAttention(
                    dim=self.inner_dim,
                    dim_head=self.cross_attn_dim_head,
                    heads=self.cross_attn_num_heads,
                    kv_dim=self.cross_attn_kv_dim,
                ).to(device, dtype=weight_dtype)
                for _ in range(self.num_cross_attn)
            ]
        )

    def save_face_modules(self, path: str):
        save_dict = {
            "local_facial_extractor": self.local_facial_extractor.state_dict(),
            "perceiver_cross_attention": [ca.state_dict() for ca in self.perceiver_cross_attention],
        }
        torch.save(save_dict, path)

    def load_face_modules(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.local_facial_extractor.load_state_dict(checkpoint["local_facial_extractor"])
        for ca, state_dict in zip(self.perceiver_cross_attention, checkpoint["perceiver_cross_attention"]):
            ca.load_state_dict(state_dict)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        id_cond: Optional[torch.Tensor] = None,
        id_vit_hidden: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # fuse clip and insightface
        if self.is_train_face:
            assert id_cond is not None and id_vit_hidden is not None
            valid_face_emb = self.local_facial_extractor(
                id_cond, id_vit_hidden
            )  # torch.Size([1, 1280]), list[5](torch.Size([1, 577, 1024]))  ->  torch.Size([1, 32, 2048])

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        # torch.Size([1, 226, 4096])   torch.Size([1, 13, 32, 60, 90])
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)  # torch.Size([1, 17776, 3072])
        hidden_states = self.embedding_dropout(hidden_states)  # torch.Size([1, 17776, 3072])

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]  # torch.Size([1, 226, 3072])
        hidden_states = hidden_states[:, text_seq_length:]  # torch.Size([1, 17550, 3072])

        # 3. Transformer blocks
        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if self.is_train_face:
                if i % self.cross_attn_interval == 0 and valid_face_emb is not None:
                    hidden_states = hidden_states + self.local_face_scale * self.perceiver_cross_attention[ca_idx](
                        valid_face_emb, hidden_states
                    )  # torch.Size([2, 32, 2048])  torch.Size([2, 17550, 3072])
                    ca_idx += 1

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for ConsisID (number of input channels is equal to output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained_cus(cls, pretrained_model_path, subfolder=None, config_path=None, transformer_additional_kwargs={}):
        if subfolder:
            config_path = config_path or pretrained_model_path
            config_file = os.path.join(config_path, subfolder, 'config.json')
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_file = os.path.join(config_path or pretrained_model_path, 'config.json')

        print(f"Loading 3D transformer's pretrained weights from {pretrained_model_path} ...")

        # Check if config file exists
        if not os.path.isfile(config_file):
            raise RuntimeError(f"Configuration file '{config_file}' does not exist")

        # Load the configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if model.state_dict()['patch_embed.proj.weight'].size() != state_dict['patch_embed.proj.weight'].size():
            new_shape   = model.state_dict()['patch_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).expand(new_shape).clone()
                state_dict['patch_embed.proj.weight'][:, :, :-1] = 0
            else:
                if model.state_dict()['patch_embed.proj.weight'].size()[1] > state_dict['patch_embed.proj.weight'].size()[1]:
                    model.state_dict()['patch_embed.proj.weight'][:, :state_dict['patch_embed.proj.weight'].size()[1], :, :] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][:, state_dict['patch_embed.proj.weight'].size()[1]:, :, :] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']
                else:
                    model.state_dict()['patch_embed.proj.weight'][:, :, :, :] = state_dict['patch_embed.proj.weight'][:, :model.state_dict()['patch_embed.proj.weight'].size()[1], :, :]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']

        tmp_state_dict = {}
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        return model

if __name__ == '__main__':
    device = "cuda:0"
    weight_dtype = torch.bfloat16
    pretrained_model_name_or_path = "BestWishYsh/ConsisID-preview"

    transformer_additional_kwargs={
        'torch_dtype': weight_dtype,
        'revision': None,
        'variant': None,
        'is_train_face': True,
        'is_kps': False,
        'LFE_num_tokens': 32,
        'LFE_output_dim': 768,
        'LFE_heads': 12,
        'cross_attn_interval': 2,
    }

    transformer = ConsisIDTransformer3DModel.from_pretrained_cus(
        pretrained_model_name_or_path,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    transformer.to(device, dtype=weight_dtype)
    for param in transformer.parameters():
        param.requires_grad = False
    transformer.eval()

    b = 1
    dim = 32
    pixel_values     = torch.ones(b, 49, 3, 480, 720).to(device, dtype=weight_dtype)
    noisy_latents    = torch.ones(b, 13, dim, 60, 90).to(device, dtype=weight_dtype)
    target           = torch.ones(b, 13, dim, 60, 90).to(device, dtype=weight_dtype)
    latents          = torch.ones(b, 13, dim, 60, 90).to(device, dtype=weight_dtype)
    prompt_embeds    = torch.ones(b, 226, 4096).to(device, dtype=weight_dtype)
    image_rotary_emb = (torch.ones(17550, 64).to(device, dtype=weight_dtype), torch.ones(17550, 64).to(device, dtype=weight_dtype))
    timesteps        = torch.tensor([311]).to(device, dtype=weight_dtype)
    id_vit_hidden    = [torch.ones([1, 577, 1024]).to(device, dtype=weight_dtype)] * 5
    id_cond          = torch.ones(b, 1280).to(device, dtype=weight_dtype)
    assert len(timesteps) == b

    model_output = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    id_vit_hidden=id_vit_hidden if id_vit_hidden is not None else None,
                    id_cond=id_cond if id_cond is not None else None,
                )[0]

    print(model_output)
