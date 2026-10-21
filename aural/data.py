import hashlib
import io
import json
from pathlib import Path

import soundfile as sf


def parquet_records(dataset, config):
    import pyarrow.parquet as pq

    root = Path(dataset)
    if root.is_dir():
        files = sorted((root / "data" / config).glob("*.parquet"))
    else:
        from huggingface_hub import HfApi, hf_hub_download

        names = sorted(
            name
            for name in HfApi().list_repo_files(dataset, repo_type="dataset")
            if name.startswith(f"data/{config}/") and name.endswith(".parquet")
        )
        files = (Path(hf_hub_download(dataset, name, repo_type="dataset")) for name in names)
    for path in files:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=64):
                yield from batch.to_pylist()


def read_jsonl(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {number}")
            audio = row.get("audio")
            if isinstance(audio, str) and audio:
                audio_path = Path(audio).expanduser()
                if not audio_path.is_absolute():
                    audio_path = path.parent / audio_path
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio at line {number}: {audio_path}")
                row["audio"] = str(audio_path.resolve())
            yield row


def safe_text(value):
    return str(value or "").replace("###", "＃＃＃").strip()


def training_row(row, stage, compression_factor=4, reward="exact"):
    instruction = safe_text(row.get("instruction", ""))
    audio = row.get("audio")
    if audio and not isinstance(audio, str):
        raise ValueError("Local training data requires audio to be a file path")
    if not audio and not instruction:
        raise ValueError("Each sample requires audio or instruction")
    question = ("<audio>\n" + instruction).strip() if audio else instruction
    answer = safe_text(row.get("response", ""))
    reasoning = safe_text(row.get("reasoning", ""))
    if stage in {"cot", "sft"} and (not reasoning or not answer):
        raise ValueError("SFT samples require nonempty reasoning and response")
    if stage == "cot":
        target = f"<think>{reasoning}</think><answer>{answer}</answer>"
    elif stage == "sft":
        target = f"(Thinking speed: {compression_factor})###{reasoning}###Answer:{answer}"
    else:
        target = f"(Thinking speed: {compression_factor})###.###Answer:"
    result = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": target},
        ]
    }
    if audio:
        result["audios"] = [audio]
    if stage == "rl":
        sample = {
            "id": str(row.get("id", "")),
            "instruction": instruction,
            "audio": audio or "",
            "reference": str(row.get("reference", row.get("response", ""))),
            "reward": str(row.get("reward", reward)),
            "metadata": row.get("metadata", {}),
        }
        if sample["reward"] == "exact" and not sample["reference"].strip():
            raise ValueError("Exact-match RL rewards require a reference answer")
        if not sample["id"]:
            sample["id"] = hashlib.sha256(
                json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:16]
        result["aural_sample"] = json.dumps(sample, ensure_ascii=False)
    return result


def materialize_audio(audio, path):
    if not audio or not audio.get("bytes"):
        raise ValueError("The dataset audio column must contain embedded bytes")
    path = Path(path)
    waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]))
    if len(waveform) == 0 or sample_rate != 16000:
        raise ValueError("Expected nonempty 16 kHz audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(audio["bytes"])
    temporary.replace(path)
