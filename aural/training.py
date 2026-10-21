import argparse
import json
import os
from pathlib import Path

from .config import COT_SYSTEM_PROMPT, SYSTEM_PROMPT, AuralConfig, resolve_model
from .data import read_jsonl, training_row


def parse_args(stage):
    parser = argparse.ArgumentParser(description=f"Train AURAL {stage.upper()}")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Omni-7B" if stage == "cot" else None,
        required=stage != "cot",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compression-factor", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--epochs", type=float, default=2 if stage == "rl" else 1)
    parser.add_argument("--batch-size", type=int, default=1 if stage == "rl" else 4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6 if stage == "rl" else 1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-5 if stage == "rl" else 1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--attn-impl", choices=["flash_attn", "sdpa", "eager"], default="flash_attn"
    )
    parser.add_argument("--deepspeed", choices=["zero2", "none"], default="zero2")
    parser.add_argument("--freeze-audio-encoder", action="store_true")
    parser.add_argument("--lss-warmup-steps", type=int, default=500)
    if stage == "rl":
        parser.add_argument("--reward", default="exact")
        parser.add_argument("--group-size", type=int, default=8)
        parser.add_argument("--max-latent", type=int, default=64)
        parser.add_argument("--max-answer-tokens", type=int, default=128)
        parser.add_argument("--rollout-token-budget", type=int, default=32768)
    args = parser.parse_args()
    if min(args.batch_size, args.gradient_accumulation, args.max_length) < 1:
        parser.error("Batch size, gradient accumulation and maximum length must be positive")
    if stage == "rl" and args.batch_size != 1:
        parser.error("RL uses --batch-size 1; change --group-size for more rollouts")
    return args


def main(stage):
    args = parse_args(stage)
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["USE_AUDIO_IN_VIDEO"] = "0"
    os.environ["USE_HF"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("AURAL_DS_CANONICAL_REDUCTION", "1")
    model_path = resolve_model(str(args.resume or args.model))
    if stage == "rl" and not (model_path / "aural_config.json").is_file():
        raise ValueError("RL requires an AURAL-SFT checkpoint with aural_config.json")
    if (model_path / "aural_config.json").is_file():
        config = AuralConfig.load(model_path)
        args.compression_factor = config.compression_factor
        args.chunk_size = config.chunk_size
        os.environ["AURAL_INIT_CHECKPOINT"] = str(model_path)
    else:
        os.environ.pop("AURAL_INIT_CHECKPOINT", None)
    os.environ["AURAL_COMPRESSION_FACTOR"] = str(args.compression_factor)
    os.environ["AURAL_CHUNK_SIZE"] = str(args.chunk_size)
    os.environ["AURAL_LP_LR"] = str(args.head_learning_rate)
    os.environ["AURAL_LSS_WARMUP_STEPS"] = str(args.lss_warmup_steps)
    if stage == "rl":
        for key, value in {
            "GROUP_SIZE": args.group_size,
            "MAX_LATENT_FORWARD": args.max_latent,
            "MAX_ANSWER_TOKENS": args.max_answer_tokens,
            "ROLLOUT_TOKEN_BUDGET": args.rollout_token_budget,
        }.items():
            os.environ["AURAL_RL_" + key] = str(value)
    import torch
    import torch.distributed as dist
    from swift import SftArguments, sft_main
    from swift.trainers import trainer_factory
    from swift.utils import get_logger

    from .deepspeed_patch import install_canonical_zero12_reduction, install_fp32_grad_norm
    from .template import TEMPLATE_NAME, register_aural_template

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    args.output = args.output.expanduser().resolve()
    prepared = args.output / "prepared_train.jsonl"
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        temporary = prepared.with_suffix(".partial")
        count = 0
        with temporary.open("w", encoding="utf-8") as stream:
            for row in read_jsonl(args.data):
                record = training_row(
                    row, stage, args.compression_factor, getattr(args, "reward", "exact")
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        if not count:
            raise ValueError("Training data is empty")
        temporary.replace(prepared)
    if dist.is_initialized():
        dist.barrier()
    install_canonical_zero12_reduction(get_logger())
    install_fp32_grad_norm(get_logger())
    if stage != "cot":
        register_aural_template(args.compression_factor)
        trainer_factory.TrainerFactory.TRAINER_MAPPING["causal_lm"] = (
            "aural.rl.trainer.RLTrainer" if stage == "rl" else "aural.sft.SFTTrainer"
        )
    training_args = SftArguments(
        model=str(model_path),
        model_type="qwen2_5_omni",
        template="qwen2_5_omni" if stage == "cot" else TEMPLATE_NAME,
        system=COT_SYSTEM_PROMPT if stage == "cot" else SYSTEM_PROMPT,
        tuner_type="full",
        freeze_llm=False,
        freeze_vit=args.freeze_audio_encoder,
        freeze_aligner=True,
        freeze_parameters=["thinker.visual"],
        trainable_parameters=["thinker.audio_tower.proj"],
        torch_dtype="bfloat16",
        attn_impl=args.attn_impl,
        dataset=[str(prepared)],
        streaming=False,
        lazy_tokenize=True,
        split_dataset_ratio=0,
        strict=True,
        remove_unused_columns=False,
        dataset_num_proc=1,
        load_from_cache_file=False,
        dataset_shuffle=True,
        max_length=args.max_length,
        truncation_strategy="delete",
        padding_free=False,
        output_dir=str(args.output),
        add_version=False,
        create_checkpoint_symlink=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05 if stage == "rl" else 0.03,
        weight_decay=0 if stage == "rl" else 0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        deepspeed=None if args.deepspeed == "none" else args.deepspeed,
        ddp_find_unused_parameters=False,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        report_to=[],
        dataloader_num_workers=args.workers,
        seed=args.seed,
        data_seed=args.seed,
        resume_from_checkpoint=str(args.resume.resolve()) if args.resume else None,
        use_hf=True,
    )
    sft_main(training_args)
    if dist.is_initialized():
        dist.destroy_process_group()
