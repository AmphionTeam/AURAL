import argparse
import json
from dataclasses import asdict


def main():
    parser = argparse.ArgumentParser(description="Generate an answer with AURAL")
    parser.add_argument("--model", default="YuxiangW/AURAL")
    parser.add_argument("--audio")
    parser.add_argument("--text", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-impl", choices=["flash_attn", "sdpa", "eager"], default="flash_attn"
    )
    parser.add_argument("--max-latent", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.audio and not args.text.strip():
        parser.error("Provide --audio or --text")
    import torch

    from .inference import load_generator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = load_generator(
        args.model,
        device=args.device,
        attn_impl=args.attn_impl,
        max_n_latent_forward=args.max_latent,
        max_new_tokens=args.max_new_tokens,
    )
    content = "<audio>" if args.audio else ""
    if args.text.strip():
        content = (content + "\n" + args.text).strip()
    with torch.inference_mode():
        result = generator.generate(
            [{"role": "user", "content": content}],
            audios=[args.audio] if args.audio else [],
        )
    print(json.dumps(asdict(result), ensure_ascii=False) if args.json else result.answer)


if __name__ == "__main__":
    main()
