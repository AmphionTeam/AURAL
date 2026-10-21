<h1 align="center">AURAL</h1>

<p align="center">
  <strong>面向语音语言模型的<br>联合分块与自适应潜在推理</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.01560"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white&amp;labelColor=343541" alt="arXiv 论文" height="28"></a>
  <a href="https://huggingface.co/YuxiangW/AURAL"><img src="https://img.shields.io/badge/Hugging_Face-Model-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=343541" alt="Hugging Face 模型" height="28"></a>
  <a href="https://huggingface.co/datasets/YuxiangW/AuralReason"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=343541" alt="Hugging Face 数据集" height="28"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#安装">安装</a> · <a href="#推理">推理</a> · <a href="#auralreason">AuralReason</a> · <a href="#训练">训练</a>
</p>

AURAL 在连续潜在空间中完成推理，再生成文本回答。AURAL-SFT 从简洁的思维链中学习未来潜在状态块的分布；AURAL-RL 通过答案质量与推理简洁性相结合的奖励，提升回答质量并学习何时结束推理。

<p align="center">
  <a href="assets/overview.png"><img src="assets/overview.png" alt="AURAL 监督训练、强化学习与实验结果" width="100%"></a>
</p>

公开的 **AURAL-RL** 权重基于 **Qwen2.5-Omni-7B**，默认压缩率 **c = 4**、块大小 **m = 10**、高斯分量数为 4、协方差低秩项的秩为 16。

## 安装

使用 Linux、Python 3.12 和 NVIDIA GPU。训练使用 BF16 与 DeepSpeed ZeRO-2，推理示例使用单张 GPU。

```bash
git clone https://github.com/AmphionTeam/AURAL.git
cd AURAL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

安装文件已指定兼容的依赖版本。编译 FlashAttention 需要 CUDA 工具链；也可以通过 `--attn-impl sdpa` 使用 PyTorch 的注意力实现。

## 推理

首次使用会自动下载模型：

```bash
python -m aural.infer --audio speech.wav
```

可以同时提供音频和任务指令：

```bash
python -m aural.infer \
  --audio speech.wav \
  --text "Describe the emotion in the speaker's voice."
```

纯文本输入使用相同接口：

```bash
python -m aural.infer --text "What is 17 + 25?"
```

使用 `--model /path/to/checkpoint` 加载本地模型，`--device cuda:1` 选择 GPU，`--json` 输出潜在推理步数和结束状态。默认以温度 1 采样潜在状态与 EOL 决策，最多生成 100 个潜在状态，最终回答使用贪心解码。

Python 调用：

```python
from aural.inference import load_generator

model = load_generator("YuxiangW/AURAL")
result = model.generate(
    [{"role": "user", "content": "<audio>\nDescribe the speaker's emotion."}],
    audios=["speech.wav"],
)
print(result.answer)
```

## AuralReason

公开数据包含三个来源中完整的、经过筛选并重新构造 CoT 的样本：

| 配置 | 样本数 | 语言 | 任务 |
|---|---:|---|---|
| `LIME` | 314,855 | 中文、英文 | 感知语音特征的对话 |
| `EmotionCoT` | 17,840 | 英文 | 语音情绪识别 |
| `HumanSpeech` | 74,330 | 中文、英文 | 感知语音特征的对话 |
| **合计** | **407,025** | | |

每个配置提供一个 `train` 划分。EmotionCoT 使用 IEMOCAP Sessions 1–4 和 MELD train/dev。音频以 16 kHz 单声道 FLAC 格式嵌入 Parquet。

| 列名 | 内容 |
|---|---|
| `id` | 稳定的样本标识 |
| `audio` | 音频字节及公开文件名 |
| `instruction` | 任务指令；纯音频对话时为空 |
| `transcript` | 音频中的口述内容 |
| `reasoning` | 简洁的 CoT 监督 |
| `response` | 助手回复或情绪标签 |
| `language` | `zh` 或 `en` |
| `source` | 原始语音数据来源 |

以下方式读取音频，无需额外安装音频解码依赖：

```python
import io
import soundfile as sf
from datasets import Audio, load_dataset

data = load_dataset("YuxiangW/AuralReason", "EmotionCoT", split="train")
data = data.cast_column("audio", Audio(decode=False))
sample = data[0]
waveform, sample_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]))
print(sample["reasoning"])
print(sample["response"])
```

将三个配置下载并转换为本地训练数据：

```bash
python -m aural.prepare_data --output data/auralreason
```

使用 `--configs EmotionCoT` 选择数据来源，或使用 `--limit 32` 准备一个小型子集。命令会生成 `train.jsonl` 及其音频文件。

## 训练

训练流程为 **CoT 初始化 → AURAL-SFT → AURAL-RL**。各阶段读取本地 JSONL，并保存可加载的 checkpoint。下面的 `N` 替换为训练结束时打印的 checkpoint 步数。

### 1. CoT 初始化

```bash
torchrun --standalone --nproc_per_node=8 -m aural.train_cot \
  --model Qwen/Qwen2.5-Omni-7B \
  --data data/auralreason/train.jsonl \
  --output outputs/cot
```

### 2. AURAL-SFT

```bash
torchrun --standalone --nproc_per_node=8 -m aural.train_sft \
  --model outputs/cot/checkpoint-N \
  --data data/auralreason/train.jsonl \
  --output outputs/sft
```

SFT 将每 `c` 个 CoT token 的 embedding 合并为一个潜在目标，并通过低秩高斯混合模型联合预测未来 `m` 个状态。训练结合联合似然与分块语义对齐（CSA）；潜在计划采样（LSS）将采样的状态块送入第二次前向，其损失权重在前 500 次更新中逐步增加。

### 3. AURAL-RL

将自己的 RL 问题准备为 JSONL，例如：

```json
{"instruction":"What is 17 + 25? Reply with just the number.","reference":"42","reward":"exact"}
```

语音问题增加 `audio` 字段。从自己的 SFT checkpoint 开始训练：

```bash
torchrun --standalone --nproc_per_node=8 -m aural.train_rl \
  --model outputs/sft/checkpoint-N \
  --data data/rl.jsonl \
  --output outputs/rl
```

RL 为每个问题采样八条轨迹，优化潜在状态块、EOL 决策和回答 token。奖励同时考虑答案质量和由质量加权的简洁性；同组候选同时包含低质量与高质量回答时，进行组内相对优势更新。

### 使用自己的 SFT 数据

每行一个 JSON 对象：

```json
{"audio":"audio/example.flac","instruction":"Describe the emotion in the voice.","reasoning":"A fast tempo and rising pitch suggest excitement.","response":"The speaker sounds excited."}
```

`audio` 支持相对于 JSONL 的路径或绝对路径。纯音频对话将 `instruction` 留空；纯文本任务省略 `audio`，将问题写入 `instruction`。`reasoning` 和 `response` 必填，推理分隔符及音频标记由训练入口自动添加。

### 自定义奖励

RL 支持 `exact`、`text_judge`、`audio_judge` 和 Python `module:function`。自定义函数接收样本及候选回答，为每个回答返回一个 `[0, 1]` 范围内的质量分数。例如，将以下内容保存为可导入目录中的 `my_reward.py`：

```python
def score(sample, answers):
    expected = sample["reference"].strip().casefold()
    scores = []
    for answer in answers:
        answer = answer.strip().removeprefix("Answer:").strip().casefold()
        scores.append(float(answer == expected))
    return scores
```

在样本中设置 `"reward": "my_reward:score"`，或者通过 `--reward my_reward:score` 指定默认函数。可选的 `metadata` 对象用于向函数传递任务信息。

使用 API 评判时，通过环境变量配置兼容 OpenAI chat-completions 格式的接口：

| 奖励类型 | 环境变量 |
|---|---|
| `text_judge` | `AURAL_TEXT_JUDGE_URL`、`AURAL_TEXT_JUDGE_MODEL`、`AURAL_TEXT_JUDGE_API_KEY` |
| `audio_judge` | `AURAL_AUDIO_JUDGE_URL`、`AURAL_AUDIO_JUDGE_MODEL`、`AURAL_AUDIO_JUDGE_API_KEY` |

`URL` 为完整的 chat-completions 地址。音频接口需要支持携带 base64 WAV 或 FLAC 的 `input_audio`。在 `reference` 中提供参考回复；开放式评分可以留空。

### 训练参数

| 参数 | CoT / SFT 默认值 | RL 默认值 |
|---|---|---|
| `--epochs` | 1 | 2 |
| `--batch-size` | 4 | 1 个问题 |
| `--gradient-accumulation` | 1 | 1 |
| `--learning-rate` | `1e-5` | `1e-6` |
| `--head-learning-rate` | SFT：`1e-4` | `1e-5` |
| `--max-length` | 4096 | 4096 |
| `--compression-factor` | 4 | 从 checkpoint 读取 |
| `--chunk-size` | 10 | 从 checkpoint 读取 |

训练更新语言模型、音频编码器、对齐层及潜在预测头，视觉编码器保持冻结。使用 `--freeze-audio-encoder` 可以冻结音频编码器。根据显存调整 `--batch-size` 和 `--gradient-accumulation`；`--max-steps` 用于短程训练，`--save-steps` 控制保存频率。

从保存点恢复训练：

```bash
torchrun --standalone --nproc_per_node=8 -m aural.train_sft \
  --model outputs/sft/checkpoint-N \
  --resume outputs/sft/checkpoint-N \
  --data data/auralreason/train.jsonl \
  --output outputs/sft
```

RL 同样支持 `--resume`。多机训练使用标准 `torchrun` 参数 `--nnodes`、`--node_rank`、`--master_addr` 和 `--master_port`。

## 引用

```bibtex
@article{wang2026aural,
  title={AURAL: Adaptive Latent Reasoning with Joint Chunk for Speech Language Models},
  author={Wang, Yuxiang and others},
  journal={arXiv preprint arXiv:2610.01560},
  year={2026}
}
```

## 致谢

本项目基于 [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)、[MS-SWIFT](https://github.com/modelscope/ms-swift) 和 PyTorch 构建。感谢各原始语音数据集的作者，AuralReason 每条样本均保留数据来源。
