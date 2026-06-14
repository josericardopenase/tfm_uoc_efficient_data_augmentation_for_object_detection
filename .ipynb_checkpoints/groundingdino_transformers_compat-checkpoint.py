"""Make IDEA-Research groundingdino work with transformers v5 (e.g. 5.x + SAM3).

- Restores ``get_head_mask`` / ``_convert_head_mask_to_5d`` removed in transformers PR #41076.
- Fixes ``BertModelWarper.forward`` passing ``device`` as the third arg to
  ``get_extended_attention_mask`` (v5 expects optional ``dtype``); call without it.

Call ``apply()`` once before importing ``autodistill_grounded_sam`` or ``groundingdino.util.inference``.
"""

from __future__ import annotations

import torch
from typing import Optional


def apply() -> None:
    """Idempotent monkey-patch."""
    _patch_pretrained_head_mask()
    _patch_bert_model_warper_forward()


def _patch_pretrained_head_mask() -> None:
    from transformers import PreTrainedModel

    if getattr(PreTrainedModel, "_grounding_dino_hf_compat", False):
        return

    def _convert_head_mask_to_5d(self, head_mask: torch.Tensor, num_hidden_layers: int) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5
        return head_mask.to(dtype=self.dtype)

    def get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    PreTrainedModel.get_head_mask = get_head_mask
    PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
    PreTrainedModel._grounding_dino_hf_compat = True


def _compat_bertwarp_forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(
            ((batch_size, seq_length + past_key_values_length)), device=device
        )
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # transformers v5: third arg is ``dtype``, not ``device``
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )

    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values if hasattr(encoder_outputs, "past_key_values") else None,
        hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, "hidden_states") else None,
        attentions=encoder_outputs.attentions if hasattr(encoder_outputs, "attentions") else None,
        cross_attentions=encoder_outputs.cross_attentions if hasattr(encoder_outputs, "cross_attentions") else None,
    )


def _patch_bert_model_warper_forward() -> None:
    try:
        from groundingdino.models.GroundingDINO import bertwarper as bm
    except ImportError:
        return

    if getattr(bm.BertModelWarper, "_grounding_dino_hf_compat_forward", False):
        return

    bm.BertModelWarper.forward = _compat_bertwarp_forward
    bm.BertModelWarper._grounding_dino_hf_compat_forward = True
