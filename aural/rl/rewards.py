import base64
import importlib
import json
import math
import os
import re
import time
from pathlib import Path

import requests


def normalize_answer(answer):
    answer = answer.strip()
    if answer.startswith("Answer:"):
        answer = answer[len("Answer:") :].strip()
    return " ".join(answer.casefold().split())


def validate_scores(scores, count):
    if not isinstance(scores, (list, tuple)) or len(scores) != count:
        raise ValueError("The reward function must return one score per answer")
    result = [float(score) for score in scores]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in result):
        raise ValueError("Quality scores must be finite numbers between 0 and 1")
    return result


class RewardScorer:
    def __init__(self, default="exact"):
        self.default = default
        self.session = requests.Session()

    def _judge(self, sample, answers, audio=False):
        prefix = "AURAL_AUDIO_JUDGE" if audio else "AURAL_TEXT_JUDGE"
        url = os.environ.get(prefix + "_URL", "")
        key = os.environ.get(prefix + "_API_KEY", "")
        model = os.environ.get(prefix + "_MODEL", "")
        if not url or not key or not model:
            raise ValueError(f"Set {prefix}_URL, {prefix}_API_KEY and {prefix}_MODEL")
        prompt = (
            "Score each candidate from 0 to 1 for correctness, relevance and helpfulness. "
            "Accept equivalent wording. Use the source audio when present. "
            "The reference is guidance, not wording to copy. Return JSON only: "
            '{"scores":[0.0,1.0]}. Preserve candidate order.\n'
            + json.dumps(
                {
                    "instruction": sample["instruction"],
                    "reference": sample["reference"],
                    "candidates": answers,
                },
                ensure_ascii=False,
            )
        )
        content = prompt
        if audio:
            path = Path(sample.get("audio", ""))
            if not path.is_file():
                raise ValueError("Audio judging requires a local audio file")
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                        "format": path.suffix.lstrip(".").lower(),
                    },
                },
            ]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 1024,
        }
        for attempt in range(3):
            try:
                response = self.session.post(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                    timeout=60,
                )
                if response.status_code != 200:
                    raise ValueError(f"Judge returned HTTP {response.status_code}")
                text = response.json()["choices"][0]["message"]["content"]
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                return validate_scores(json.loads(text)["scores"], len(answers))
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
                if attempt == 2:
                    raise RuntimeError("Reward service failed to return valid scores") from None
                time.sleep(2**attempt)

    def __call__(self, sample, answers):
        backend = sample.get("reward") or self.default
        if backend == "exact":
            reference = normalize_answer(sample["reference"])
            return [float(normalize_answer(answer) == reference) for answer in answers]
        if backend in {"text_judge", "audio_judge"}:
            return self._judge(sample, answers, audio=backend == "audio_judge")
        if ":" not in backend:
            raise ValueError("Reward must be exact, text_judge, audio_judge or module:function")
        module, function = backend.rsplit(":", 1)
        scorer = getattr(importlib.import_module(module), function)
        return validate_scores(scorer(sample, answers), len(answers))
