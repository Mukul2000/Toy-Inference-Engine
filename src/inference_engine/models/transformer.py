from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from inference_engine.config import ModelConfig, DEFAULT_MODEL_NAME
from inference_engine.kv_cache import ContiguousKVCache
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
        kv_cache: Optional[ContiguousKVCache] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that automatically derives token `positions`:
          - If `kv_cache` is provided, positions are derived directly from `kv_cache.current_seq_len`.
          - If `kv_cache` is None, positions default to `[0, 1, ..., len(input_ids) - 1]`.

        Dimensions & Meaning of Matrix `x` (let N = len(input_ids), H = hidden_size (576), V = vocab_size (49152)):
          - input_ids : [N]     -> The N input token IDs.
          - positions : [N]     -> The sentence position index (0, 1, 2, ...) of each input token.
          - x         : [N, H]  -> A 2D matrix of N rows (one 576-float vector per input token).
                                   After passing through the 30 layers, Row `i` (`x[i]`) holds the
                                   576-float hidden representation predicting the word that comes AFTER token `i`.
                                   For generation, we only care about the last row (`x[-1]`), which predicts
                                   the word that comes after the entire sequence!
          - logits    : [N, V]  -> Vocabulary similarity scores for each of the N rows.
        """
        # input_ids: [num_tokens]
        if positions is None:
            if kv_cache is not None:
                positions = kv_cache.get_positions(len(input_ids), input_ids.device)
            else:
                positions = torch.arange(
                    len(input_ids), dtype=torch.long, device=input_ids.device
                )

        # 1. Lookup initial token embeddings: [num_tokens] -> [num_tokens, hidden_size]
        # each token is represented by `hidden_size` floating point numbers
        # these are the initial representations, Meaning of Row i: "Who am I in isolation?" 
        x = self.embed_tokens(input_ids)

        # 2. Pass through all 30 Transformer layers: [num_tokens, hidden_size] -> [num_tokens, hidden_size]
        #    (Each layer adds Attention + MLP updates onto `x` via residual connections: x = x + delta)
        # As we pass through the layers
        # Meaning of Row i: "Absorbing context from the words before me", each word "talks" to the previous ones
        
        for layer in self.layers:
            x = layer(x, positions, self.rope, kv_cache=kv_cache)

        #after completing all the layers
        # Meaning of Row i: "What word should come right after me?"
        # 3. Final RMSNorm: [num_tokens, hidden_size] -> [num_tokens, hidden_size]
        #    Rescales each row's 576 numbers back to unit root-mean-square magnitude after 30 layers of additions.
        x = self.norm(x)

        # 4. Project to vocabulary scores (logits): [num_tokens, hidden_size] -> [num_tokens, vocab_size]
        # at this point the last row of x contains 576 floats which in some weird dimension describe the next token
        # lm_head contains a mapping if you remember, it maps tokens to a 576 float vector
        # we do the same thing here, we multiply the predicted vector by each of the 49152 token vectors
        # the dot product tells us how similar the predicted and the multiplied token are
        # giving us a list of raw scores for each token in the vocabulary - called logits
        # we can greedy sample to pick the token with the highest score
        # or use temperature sampling
        logits = self.lm_head(x)
        return logits
