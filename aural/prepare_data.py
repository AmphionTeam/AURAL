import argparse
import json
from contextlib import closing
from pathlib import Path

from .data import materialize_audio, parquet_records


def main():
    parser = argparse.ArgumentParser(description="Prepare AuralReason for local training")
    parser.add_argument("--dataset", default="YuxiangW/AuralReason")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["LIME", "EmotionCoT", "HumanSpeech"],
        default=["LIME", "EmotionCoT", "HumanSpeech"],
    )
    parser.add_argument("--output", type=Path, default=Path("data/auralreason"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "train.jsonl"
    temporary = target.with_suffix(".jsonl.partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for config in args.configs:
            with closing(parquet_records(args.dataset, config)) as records:
                for row in records:
                    audio_path = args.output / "audio" / config / (row["id"] + ".flac")
                    materialize_audio(row["audio"], audio_path)
                    record = {
                        "id": row["id"],
                        "audio": str(audio_path.relative_to(args.output)),
                        "instruction": row["instruction"],
                        "reasoning": row["reasoning"],
                        "response": row["response"],
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if args.limit and count >= args.limit:
                        break
            if args.limit and count >= args.limit:
                break
    temporary.replace(target)
    print(f"Prepared {count} examples: {target}")


if __name__ == "__main__":
    main()
