from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from nanotron.config.utils_config import InitScalingMethod
from nanotron.nn.attention import ALL_ATTENTION_FUNCTIONS, AttentionImplementation

# The default attention implementation to use
DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"


@dataclass
class RandomInit:
    std: float
    scaling_method: InitScalingMethod = InitScalingMethod.NUM_LAYERS


@dataclass
class SpectralMupInit:
    """This is used to initialize the model with spectral mup. Set it to True to use it."""

    use_mup: bool

    def __post_init__(self):
        assert self.use_mup, "Remove `use_mup` if you don't want to use it"


@dataclass
class ExistingCheckpointInit:
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: Path


@dataclass
class Qwen2Config:
    """Configuration for a QWEN2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_qwen2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    rope_seq_len_interpolation_factor: Optional[float] = None  # if not None, discrete positions will be interpolated by this factor via the trick in https://arxiv.org/abs/2306.15595
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION
    flex_attention_mask: Optional[str] = None
    attention_bias: bool = False
    sliding_window_size: Optional[int] = None
    z_loss_enabled: bool = False  # Z-loss regularization https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf
    z_loss_coefficient: float = 0.0001  # Default from the paper (10^-4)
    no_rope_layer: Optional[
        int
    ] = None  # Skip rope every no_rope_layer layers (see https://arxiv.org/abs/2501.18795 https://arxiv.org/abs/2305.19466 and Llama4)
    _fused_rotary_emb: bool = True
    _fused_rms_norm: bool = True
    _use_qkv_packed: bool = True
    _use_doc_masking: bool = False

    log_attn_probs: bool = True # Whether to log the attention probabilities
    ring_attn_heads_k_stride: Optional[int] = None # Stride of the heads in the key tensor for llama3 ring attention

    def __post_init__(self):
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass LlamaConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate that the attention implementation is valid
        if self._attn_implementation is not None:
            assert (
                self._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

        if self.sliding_window_size is not None:
            assert self._attn_implementation in [
                "flex_attention",
                "flash_attention_2",
                "llama3_ring_attention",
            ], "Sliding window is only supported for Flex Attention and Flash Attention 2"
        if self.flex_attention_mask is not None:
            assert (
                self._attn_implementation == "flex_attention"
            ), "Flex attention mask is only supported for flex attention"
            assert self.flex_attention_mask in [
                "sliding_window",
                "document",
                "sliding_window_document",
            ], "Flex attention mask must be one of ['sliding_window', 'document', 'sliding_window_document']"
        if self.no_rope_layer is not None:
            assert (
                self.num_hidden_layers % self.no_rope_layer == 0
            ), "no_rope_layer must be a multiple of num_hidden_layers"

        if self._attn_implementation == "llama3_ring_attention":
            assert self.ring_attn_heads_k_stride is not None, "ring_attn_heads_k_stride must be specified for llama3 ring attention"
        else:
            assert self.ring_attn_heads_k_stride is None, f"ring_attn_heads_k_stride must be None for non-llama3 ring attention, got attn_implementation={self._attn_implementation}"

    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup


NanotronConfigs = Union[Qwen2Config, Any]
