import os

from swift.template.register import register_template
from swift.template.templates.qwen import Qwen2_5OmniTemplate, QwenTemplateMeta

from .markers import encode_think_markers, find_reason_span

TEMPLATE_NAME = "aural_qwen2_5_omni"


class AuralTemplate(Qwen2_5OmniTemplate):
    def _encode(self, inputs):
        encoded = super()._encode(inputs)
        if self.is_training:
            open_ids, close_ids = encode_think_markers(self.tokenizer)
            find_reason_span(encoded["input_ids"], open_ids, close_ids, require_unique=True)
        return encoded

    def _post_encode(self, model, inputs):
        return inputs


def register_aural_template(compression_factor=None):
    rate = int(compression_factor or os.environ.get("AURAL_COMPRESSION_FACTOR", "4"))
    register_template(
        QwenTemplateMeta(
            TEMPLATE_NAME,
            template_cls=AuralTemplate,
            thinking_prefix=f"(Thinking speed: {rate})###",
        ),
        exist_ok=True,
    )
