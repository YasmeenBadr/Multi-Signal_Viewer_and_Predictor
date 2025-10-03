import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel
from transformers.models.hubert.modeling_hubert import _compute_mask_indices
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Union, Tuple, List

class HuBERTECGConfig(HubertConfig):
    model_type = "hubert_ecg"

    def __init__(self, ensemble_length: int = 1, vocab_sizes: List[int] = [100], **kwargs):
        super().__init__(**kwargs)
        self.ensemble_length = ensemble_length
        self.vocab_sizes = vocab_sizes if isinstance(vocab_sizes, list) else [vocab_sizes]

class HuBERTECG(HubertModel):
    config_class = HuBERTECGConfig

    def __init__(self, config: HuBERTECGConfig):
        super().__init__(config)
        self.config = config

        # final projection layer to map encodings into codebook space
        self.final_proj = nn.ModuleList([
            nn.Linear(config.hidden_size, config.classifier_proj_size) 
            for _ in range(config.ensemble_length)
        ])

        # embedding for codebooks
        self.label_embedding = nn.ModuleList([
            nn.Embedding(vocab_size, config.classifier_proj_size) 
            for vocab_size in config.vocab_sizes
        ])

        assert len(self.final_proj) == len(self.label_embedding), "Mismatch in final_proj and label_embedding lengths"

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        # SpecAugment masking
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states, mask_time_indices

        batch_size, sequence_length, hidden_size = hidden_states.size()
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = 0
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = 0

        return hidden_states, mask_time_indices

    def logits(self, transformer_output: torch.Tensor) -> List[torch.Tensor]:
        projected_outputs = [final_proj(transformer_output) for final_proj in self.final_proj]
        ensemble_logits = [
            torch.cosine_similarity(
                proj.unsqueeze(2),
                emb.weight.unsqueeze(0).unsqueeze(0),
                dim=-1
            ) / 0.1
            for proj, emb in zip(projected_outputs, self.label_embedding)
        ]
        return ensemble_logits

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        features = self.feature_extractor(input_values).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(features.shape[1], attention_mask)

        hidden_states = self.feature_projection(features)
        hidden_states, mask_time_indices = self._mask_hidden_states(hidden_states, mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:] + (mask_time_indices,)

        final_dict = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        final_dict["mask_time_indices"] = mask_time_indices
        return final_dict
