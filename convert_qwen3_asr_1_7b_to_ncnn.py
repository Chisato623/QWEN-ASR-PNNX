"""
Convert Qwen3-ASR-1.7B audio encoder to NCNN fp16 with pnnx.

The exported model is only the audio encoder:
    mel features [1, 128, frames] -> audio embeddings [1, S, 2048]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
QWEN_ASR_SRC = ROOT_DIR / "Qwen3-ASR"
MODEL_DIR = ROOT_DIR / "Qwen3-ASR-1.7B"
OUTPUT_DIR = ROOT_DIR / "ncnn_export" / "Qwen3-ASR-1.7B"
BASE_NAME = "qwen3_asr_1_7b_audio_encoder"


def import_runtime():
    if QWEN_ASR_SRC.is_dir():
        sys.path.insert(0, str(QWEN_ASR_SRC))

    try:
        import torch
        import torch.nn as nn
        from qwen_asr import Qwen3ASRModel
    except ModuleNotFoundError as exc:
        raise SystemExit(f"[ERROR] Missing Python dependency: {exc.name}") from exc

    return torch, nn, Qwen3ASRModel


def choose_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[ERROR] --device cuda was requested, but CUDA is not available.")
    return requested


def resolve_pnnx(path: str | None) -> str:
    pnnx = path or shutil.which("pnnx")
    if path and os.path.basename(path) == path:
        pnnx = shutil.which(path)
    if not pnnx:
        raise SystemExit("[ERROR] pnnx was not found. Pass --pnnx or add pnnx to PATH.")
    return pnnx


def resolve_trace_dtype(torch, requested: str):
    if requested == "fp16":
        return requested, torch.float16
    return "fp32", torch.float32


class AudioEncoderForPnnx:
    @staticmethod
    def build(torch, nn, audio_tower, frames: int):
        class _Wrapper(nn.Module):
            def __init__(self, tower, fixed_frames: int):
                super().__init__()
                self.tower = tower
                self.register_buffer("feature_lens", torch.tensor([fixed_frames], dtype=torch.long), persistent=False)

            def forward(self, input_features):
                output = self.tower(input_features.squeeze(0), feature_lens=self.feature_lens)
                return output.last_hidden_state.unsqueeze(0)

        return _Wrapper(audio_tower, frames)


def make_output_paths(device: str, trace_dtype_name: str) -> tuple[Path, Path]:
    name = f"{BASE_NAME}_trace_{device}_{trace_dtype_name}_ncnn_fp16"
    return OUTPUT_DIR / f"{name}.pt", OUTPUT_DIR / name


def load_audio_encoder(torch, nn, Qwen3ASRModel, device: str, trace_dtype, frames: int):
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"[ERROR] Local model directory not found: {MODEL_DIR}")

    device_map = "cuda:0" if device == "cuda" else "cpu"
    print(f"[INFO] Loading model: {MODEL_DIR}")
    print(f"[INFO] Device: {device} | trace dtype: {trace_dtype}")

    model = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR),
        dtype=trace_dtype,
        device_map=device_map,
        max_new_tokens=1,
    )

    audio_tower = model.model.thinker.audio_tower
    audio_tower.eval()
    if hasattr(audio_tower, "config"):
        audio_tower.config._attn_implementation = "eager"

    return AudioEncoderForPnnx.build(torch, nn, audio_tower, frames).eval()


def trace_audio_encoder(torch, wrapper, device: str, trace_dtype, frames: int, trace_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_features = torch.randn(1, 128, frames, dtype=trace_dtype)

    if device == "cuda":
        wrapper = wrapper.cuda()
        input_features = input_features.cuda()

    print(f"[INFO] Tracing audio encoder with input [1, 128, {frames}]")
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, input_features, strict=False)
        traced = torch.jit.freeze(traced)
        traced.save(str(trace_path))
        out = traced(input_features)

    print(f"[OK] TorchScript saved: {trace_path}")
    print(f"[INFO] Output shape: {tuple(out.shape)}")
    return trace_path


def run_pnnx(pnnx: str, traced_model: Path, ncnn_prefix: Path, frames: int, device: str) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        pnnx,
        str(traced_model),
        f"inputshape=[1,128,{frames}]",
        f"pnnxparam={ncnn_prefix}.pnnx.param",
        f"pnnxbin={ncnn_prefix}.pnnx.bin",
        f"pnnxpy={ncnn_prefix}_pnnx.py",
        f"pnnxonnx={ncnn_prefix}.pnnx.onnx",
        f"ncnnparam={ncnn_prefix}.ncnn.param",
        f"ncnnbin={ncnn_prefix}.ncnn.bin",
        f"ncnnpy={ncnn_prefix}_ncnn.py",
        f"device={device}",
        "optlevel=2",
        "fp16=1",
    ]

    print("[INFO] Running pnnx:")
    print("       " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(OUTPUT_DIR), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"[ERROR] pnnx failed with exit code {result.returncode}")
        return result.returncode
    return 0


def cleanup_intermediate_files(keep_prefix: Path) -> None:
    keep = {f"{keep_prefix.name}.ncnn.param", f"{keep_prefix.name}.ncnn.bin"}
    for path in OUTPUT_DIR.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def list_outputs() -> None:
    print("[OUTPUT]")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"  {path.name:64s} {path.stat().st_size:>14,} bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR-1.7B audio encoder to NCNN fp16.")
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--trace-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--pnnx-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pnnx", default=None)
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("[ERROR] --frames must be positive.")

    pnnx = resolve_pnnx(args.pnnx)
    torch, nn, Qwen3ASRModel = import_runtime()
    device = choose_device(torch, args.device)
    trace_dtype_name, trace_dtype = resolve_trace_dtype(torch, args.trace_dtype)
    trace_path, ncnn_prefix = make_output_paths(device, trace_dtype_name)

    print("=" * 72)
    print("Qwen3-ASR-1.7B audio encoder -> TorchScript -> pnnx -> NCNN fp16")
    print(f"Model:  {MODEL_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Frames: {args.frames}")
    print(f"Trace:  {device}/{trace_dtype_name}")
    print("=" * 72)

    if args.skip_trace:
        if not trace_path.exists():
            raise SystemExit(f"[ERROR] --skip-trace used but missing: {trace_path}")
        traced_model = trace_path
    else:
        wrapper = load_audio_encoder(torch, nn, Qwen3ASRModel, device, trace_dtype, args.frames)
        traced_model = trace_audio_encoder(torch, wrapper, device, trace_dtype, args.frames, trace_path)

    code = run_pnnx(pnnx, traced_model, ncnn_prefix, args.frames, args.pnnx_device)
    if code == 0 and not args.keep_intermediate:
        cleanup_intermediate_files(ncnn_prefix)
    list_outputs()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
