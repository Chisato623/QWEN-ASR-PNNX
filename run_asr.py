"""
Qwen3-ASR 0.6B 语音识别示例脚本

用法:
    # 使用项目自带的在线示例音频（中文）
    python run_asr.py

    # 识别本地音频文件
    python run_asr.py --audio 你的音频.wav

    # 指定语言（自动检测为 None）
    python run_asr.py --audio 你的音频.wav --language Chinese

硬件要求:
    - RTX 3050 Ti 4GB 显存 → 使用 0.6B 模型（bf16 约 2.4GB）
    - 1.7B 模型需要 6GB+ 显存，不建议在 4GB 显卡上运行
"""

import argparse
import os
import sys
import time

import torch
from qwen_asr import Qwen3ASRModel

# 模型路径 - 如果已下载到本地则用本地路径，否则自动从 HuggingFace 下载
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Qwen3-ASR-0.6B")

# 在线示例音频
URL_ZH = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"


def get_device_str(device: str) -> str:
    """将内部设备标识转为可读字符串"""
    if device == "cuda":
        return f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
    return "CPU"


def load_model(device: str = "auto"):
    """加载 Qwen3-ASR-0.6B 模型

    Args:
        device: "auto" (默认, 优先 CUDA), "cuda" (强制 GPU), "cpu" (强制 CPU)
    """
    # --- 解析设备 ---
    cuda_available = torch.cuda.is_available()

    if device == "cuda":
        if not cuda_available:
            raise RuntimeError("--cuda 已指定但系统未检测到 CUDA GPU")
        dtype = torch.bfloat16
        device_map = "cuda:0"
    elif device == "cpu":
        dtype = torch.float32
        device_map = "cpu"
    else:  # auto
        if cuda_available:
            dtype = torch.bfloat16
            device_map = "cuda:0"
        else:
            dtype = torch.float32
            device_map = "cpu"

    device_label = get_device_str("cuda" if device_map.startswith("cuda") else "cpu")
    print(f"[INFO] 推理设备: {device_label}")
    print(f"[INFO] 推理精度: {'bfloat16' if dtype == torch.bfloat16 else 'float32'}")

    # --- 模型路径 ---
    if os.path.isdir(MODEL_PATH):
        print(f"[INFO] 使用本地模型: {MODEL_PATH}")
        model_path = MODEL_PATH
    else:
        print(f"[INFO] 本地模型未找到，将自动从 HuggingFace 下载 Qwen/Qwen3-ASR-0.6B")
        model_path = "Qwen/Qwen3-ASR-0.6B"

    print("[INFO] 正在加载模型... (首次下载模型可能需要几分钟)")

    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        max_new_tokens=256,
    )

    print("[INFO] 模型加载完成！")
    return asr, device_label


def transcribe(audio, language=None, context="", device="auto"):
    """执行语音识别

    Args:
        device: "auto" (默认), "cuda" (强制 GPU), "cpu" (强制 CPU)
    """
    asr, device_label = load_model(device=device)

    results = asr.transcribe(
        audio=audio,
        language=language,
        context=context,
        return_time_stamps=False,
    )

    return results, device_label


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 语音识别")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="音频文件路径。不指定则使用在线示例音频。",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=[None, "Chinese", "English", "auto"],
        help="指定语言。None 为自动检测（默认）。",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="在线音频 URL。",
    )
    parser.add_argument(
        "--zh",
        action="store_true",
        help="使用中文示例音频 (在线)。",
    )
    parser.add_argument(
        "--en",
        action="store_true",
        help="使用英文示例音频 (在线)。",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="上下文提示词，如 '交易 停滞'（帮助纠正特定领域词汇）。",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU 推理。",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="强制使用 CUDA GPU 推理。",
    )

    args = parser.parse_args()

    # --- 确定推理设备 ---
    if args.cpu and args.cuda:
        print("[ERROR] --cpu 和 --cuda 不能同时使用")
        sys.exit(1)

    if args.cpu:
        device = "cpu"
    elif args.cuda:
        device = "cuda"
    else:
        device = "auto"

    # 确定音频来源
    if args.audio:
        audio = args.audio
        print(f"[INFO] 使用本地音频: {audio}")
    elif args.url:
        audio = args.url
        print(f"[INFO] 使用在线音频: {audio}")
    elif args.en:
        audio = URL_EN
        language = "English"
        print(f"[INFO] 使用英文示例音频")
    elif args.zh:
        audio = URL_ZH
        language = None
        print(f"[INFO] 使用中文示例音频")
    else:
        # 默认：中文示例
        audio = URL_ZH
        language = None
        print(f"[INFO] 未指定音频，使用中文示例音频 (URL)")

    # 语言设置
    lang = args.language if args.language != "auto" else None
    if lang is None and args.language != "auto" and language:
        lang = language

    if lang:
        print(f"[INFO] 指定语言: {lang}")

    if args.context:
        print(f"[INFO] 上下文提示: {args.context}")

    # 开始识别
    start_time = time.time()
    print(f"\n{'='*50}")
    print("开始语音识别...")
    print(f"{'='*50}")

    try:
        results, device_label = transcribe(audio=audio, language=lang, context=args.context, device=device)
    except Exception as e:
        print(f"\n[ERROR] 识别失败: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # 输出结果
    print(f"\n{'='*50}")
    print("识别结果")
    print(f"{'='*50}")
    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"\n--- 样本 {i+1} ---")
        print(f"语言: {r.language}")
        print(f"文本: {r.text}")
    print(f"推理设备: {device_label}")
    print(f"耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()