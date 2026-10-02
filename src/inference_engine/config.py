from dataclasses import dataclass
from transformers import AutoConfig

DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@dataclass(frozen=True)
class ModelConfig:
    """
    Model hyperparameters bridging the Engine Core (Layer 1)
    and the Model Executor (Layer 2).
    """

    model_name: str = DEFAULT_MODEL_NAME
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head (d_head)."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self) -> int:
        """Group size for Grouped-Query Attention (GQA): Q heads per KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    def kv_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        """
        First-principles KV cache memory per token across all layers:
        2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        """
        return (
            2
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * dtype_bytes
        )

    @classmethod
    def from_pretrained(cls, model_name: str = DEFAULT_MODEL_NAME) -> "ModelConfig":
        """Load configuration from a Hugging Face model repository."""
        hf_config = AutoConfig.from_pretrained(model_name)
        eos_id = getattr(hf_config, "eos_token_id", 2)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = rope_params.get(
            "rope_theta", getattr(hf_config, "rope_theta", 100000.0)
        )
        return cls(
            model_name=model_name,
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            rope_theta=float(rope_theta),
            max_position_embeddings=getattr(
                hf_config, "max_position_embeddings", 8192
            ),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
            attention_bias=getattr(hf_config, "attention_bias", False),
            eos_token_id=eos_id,
        )

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Ultra-fast synthetic config for unit tests (< 20ms execution)."""
        return cls(
            model_name="tiny-synthetic",
            vocab_size=512,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
