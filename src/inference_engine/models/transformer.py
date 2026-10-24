from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from inference_engine.config import ModelConfig, DEFAULT_MODEL_NAME
from inference_engine.executor.batch_metadata import BatchMetadata
from inference_engine.executor.kv_cache import ContiguousKVCache
from inference_engine.models.rope import RotaryEmbedding
from inference_engine.models.decoder_layer import TransformerBlock


class TransformerModel(nn.Module):
    """Complete Decoder-only Transformer Language Model."""

    def __init__(self, config: ModelConfig, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, base=config.rope_theta
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(idx, config) for idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        dtype: torch.dtype = torch.float32,
    ) -> "TransformerModel":
        """
        Download raw pretrained weights from Hugging Face and load them directly
        into our custom TransformerModel layers.
        """
        config = ModelConfig.from_pretrained(model_name)
        model = cls(config).to(dtype=dtype)

        hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        hf_state_dict = hf_model.state_dict()

        mapped_state_dict = {
            key.removeprefix("model."): tensor for key, tensor in hf_state_dict.items()
        }
        if config.tie_word_embeddings and "lm_head.weight" not in mapped_state_dict:
            mapped_state_dict["lm_head.weight"] = mapped_state_dict[
                "embed_tokens.weight"
            ]

        model.load_state_dict(mapped_state_dict, strict=True)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        meta: Optional[BatchMetadata] = None,
        kv_cache: Optional[ContiguousKVCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass over a packed batch: the B rows' tokens laid end to end, N in total, no pads.
        `meta` (see `BatchMetadata`) says where each token sits (its slot in `kv_cache` and its
        column there), how attention groups the rows, and which token is each row's last.
        `meta=None` means fresh rows that all start at column 0, passed as a [B, T] rectangle
        (experiments, HF parity); they are packed here.

        Dimensions & Meaning of Matrix `x` (N = packed tokens of all rows, B = rows,
        H = hidden_size (576), V = vocab_size (49152)):
          - input_ids      : [N]    -> the new token IDs of all B rows, end to end.
          - meta.positions : [N]    -> Position of each token within its OWN request (= its column in its slot).
          - x              : [N, H] -> One 576-float vector per token. After passing through the
                                       30 layers, `x[i]` holds the hidden representation predicting
                                       the word that comes AFTER token `i` (within its own request).
                                       For generation, we only care about each row's last token
                                       (`x[meta.last_idx[b]]`), which predicts the word that comes
                                       after each entire sequence!
          - logits         : [B, V] -> Vocabulary similarity scores for the next token of each request.
        """
        if meta is None:
            batch_size, num_new_tokens = input_ids.shape
            meta = BatchMetadata.build(
                slot_ids=list(range(batch_size)),
                starts=[0] * batch_size,
                lengths=[num_new_tokens] * batch_size,
            )
            input_ids = input_ids.reshape(-1)  # rows end to end: the packed layout
        # `meta` is built once, outside the model, and shared by all 30 layers

        # 1. Lookup initial token embeddings: [N] -> [N, hidden_size]
        # each token is represented by `hidden_size` floating point numbers
        # these are the initial representations, Meaning of Row i: "Who am I in isolation?" 
        x = self.embed_tokens(input_ids)

        # 2. Pass through all 30 Transformer layers: [N, hidden_size] -> [N, hidden_size]
        #    (Each layer adds Attention + MLP updates onto `x` via residual connections: x = x + delta)
        # As we pass through the layers
        # Meaning of Row i: "Absorbing context from the words before me", each word "talks" to the previous ones
        #    (`meta.groups` restrict that to earlier words of its OWN request)
        
        for layer in self.layers:
            x = layer(x, meta, self.rope, kv_cache=kv_cache)

        #after completing all the layers
        # Meaning of Row i: "What word should come right after me?"
        # 3. Final RMSNorm: [N, hidden_size] -> [N, hidden_size]
        #    Rescales each row's 576 numbers back to unit root-mean-square magnitude after 30 layers of additions.
        x = self.norm(x)

        #    Keep only each row's last token: [N, hidden_size] -> [B, hidden_size]
        #    It is the only token whose prediction we need; in the packed layout it sits at
        #    `meta.last_idx[b]`. Skipping lm_head for the rest saves ~20% of prefill FLOPs.
        x = x[meta.last_idx]

        # 4. Project to vocabulary scores (logits): [B, hidden_size] -> [B, vocab_size]
        # at this point the last row of x contains 576 floats which in some weird dimension describe the next token
        # lm_head contains a mapping if you remember, it maps tokens to a 576 float vector
        # we do the same thing here, we multiply the predicted vector by each of the 49152 token vectors
        # the dot product tells us how similar the predicted and the multiplied token are
        # giving us a list of raw scores for each token in the vocabulary - called logits
        # we can greedy sample to pick the token with the highest score
        # or use temperature sampling
        logits = self.lm_head(x)
        return logits
