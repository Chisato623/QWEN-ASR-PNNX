"""
Export static Qwen3-ASR weight blocks to NCNN fp16 with pnnx.

This script intentionally exports only tensor-in/tensor-out modules. Shapes are
derived from the loaded model config, so the same implementation can be reused
by 0.6B and 1.7B wrappers:
    audio_cnn         [N, 1, mel_bins, chunk_frames] -> [N, T, audio_dim]
    audio_transformer [S, audio_dim]                 -> [S, audio_dim]
    audio_proj        [S, audio_dim]                 -> [S, text_dim]
    text_embed        [B, T] int64                   -> [B, T, text_dim]
    text_decoder_*    hidden + rope + mask + kv      -> hidden + kv
    text_norm         [B, T, text_dim]               -> [B, T, text_dim]
    lm_head           [B, T, text_dim]               -> [B, T, vocab]

Dynamic runtime work such as audio chunking, feature lengths, prompt assembly,
audio placeholder scatter, position ids, causal masks, sampling, and KV-cache
management should stay in C++.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
QWEN_ASR_SRC = ROOT_DIR / "Qwen3-ASR"
MODEL_DIR = ROOT_DIR / "Qwen3-ASR-0.6B"
OUTPUT_DIR = ROOT_DIR / "ncnn_export" / "Qwen3-ASR-0.6B"
BASE_NAME = "qwen3_asr_0_6b"


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


def load_model(torch, Qwen3ASRModel, device: str, trace_dtype):
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
    thinker = model.model.thinker
    model.model.eval()
    thinker.eval()
    thinker.audio_tower.eval()
    thinker.model.eval()
    thinker.lm_head.eval()

    if hasattr(thinker.audio_tower, "config"):
        thinker.audio_tower.config._attn_implementation = "eager"
    if hasattr(thinker.model, "config"):
        thinker.model.config._attn_implementation = "eager"

    return model


class AudioCnnForPnnx:
    @staticmethod
    def build(torch, nn, audio_tower):
        class _Wrapper(nn.Module):
            def __init__(self, tower):
                super().__init__()
                self.conv2d1 = tower.conv2d1
                self.conv2d2 = tower.conv2d2
                self.conv2d3 = tower.conv2d3
                self.conv_out = tower.conv_out

            def forward(self, input_features):
                hidden_states = torch.nn.functional.gelu(self.conv2d1(input_features))
                hidden_states = torch.nn.functional.gelu(self.conv2d2(hidden_states))
                hidden_states = torch.nn.functional.gelu(self.conv2d3(hidden_states))
                b, c, f, t = hidden_states.shape
                hidden_states = hidden_states.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
                return self.conv_out(hidden_states)

        return _Wrapper(audio_tower).eval()


class AudioTransformerForPnnx:
    @staticmethod
    def build(torch, nn, audio_tower, seq_len: int):
        class _Wrapper(nn.Module):
            def __init__(self, tower, fixed_seq_len: int):
                super().__init__()
                self.positional_embedding = tower.positional_embedding
                self.layers = tower.layers
                self.ln_post = tower.ln_post
                self.register_buffer(
                    "cu_seqlens",
                    torch.tensor([0, fixed_seq_len], dtype=torch.int32),
                    persistent=False,
                )

            def forward(self, hidden_states):
                position = self.positional_embedding.positional_embedding[: hidden_states.shape[0], :]
                hidden_states = hidden_states + position.to(hidden_states.dtype)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, self.cu_seqlens)[0]
                return self.ln_post(hidden_states)

        return _Wrapper(audio_tower, seq_len).eval()


class AudioProjForPnnx:
    @staticmethod
    def build(torch, nn, audio_tower):
        class _Wrapper(nn.Module):
            def __init__(self, tower):
                super().__init__()
                self.proj1 = tower.proj1
                self.act = tower.act
                self.proj2 = tower.proj2

            def forward(self, hidden_states):
                hidden_states = self.proj1(hidden_states)
                hidden_states = self.act(hidden_states)
                return self.proj2(hidden_states)

        return _Wrapper(audio_tower).eval()


class TextDecoderLayerForPnnx:
    @staticmethod
    def build(torch, nn, layer):
        class _Wrapper(nn.Module):
            def __init__(self, decoder_layer):
                super().__init__()
                self.input_layernorm = decoder_layer.input_layernorm
                self.q_proj = decoder_layer.self_attn.q_proj
                self.k_proj = decoder_layer.self_attn.k_proj
                self.v_proj = decoder_layer.self_attn.v_proj
                self.o_proj = decoder_layer.self_attn.o_proj
                self.q_norm = decoder_layer.self_attn.q_norm
                self.k_norm = decoder_layer.self_attn.k_norm
                self.post_attention_layernorm = decoder_layer.post_attention_layernorm
                self.mlp = decoder_layer.mlp
                self.num_heads = decoder_layer.self_attn.config.num_attention_heads
                self.num_key_value_heads = decoder_layer.self_attn.config.num_key_value_heads
                self.num_key_value_groups = self.num_heads // self.num_key_value_heads
                self.head_dim = decoder_layer.self_attn.head_dim
                self.scaling = decoder_layer.self_attn.scaling

            def rotate_half(self, x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            def repeat_kv(self, hidden_states):
                if self.num_key_value_groups == 1:
                    return hidden_states
                batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
                hidden_states = hidden_states[:, :, None, :, :].expand(
                    batch,
                    num_key_value_heads,
                    self.num_key_value_groups,
                    seq_len,
                    head_dim,
                )
                return hidden_states.reshape(batch, self.num_heads, seq_len, head_dim)

            def forward(self, hidden_states, cos, sin, attention_mask, cache_k, cache_v):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)

                batch, query_len, _ = hidden_states.shape
                query_states = self.q_proj(hidden_states).view(batch, query_len, self.num_heads, self.head_dim)
                key_states = self.k_proj(hidden_states).view(batch, query_len, self.num_key_value_heads, self.head_dim)
                value_states = self.v_proj(hidden_states).view(
                    batch,
                    query_len,
                    self.num_key_value_heads,
                    self.head_dim,
                )

                query_states = self.q_norm(query_states).transpose(1, 2)
                key_states = self.k_norm(key_states).transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
                query_states = (query_states * cos) + (self.rotate_half(query_states) * sin)
                key_states = (key_states * cos) + (self.rotate_half(key_states) * sin)

                out_cache_k = torch.cat((cache_k, key_states), dim=2)
                out_cache_v = torch.cat((cache_v, value_states), dim=2)

                key_for_attn = self.repeat_kv(out_cache_k)
                value_for_attn = self.repeat_kv(out_cache_v)

                attn_weights = torch.matmul(query_states, key_for_attn.transpose(2, 3)) * self.scaling
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_output = torch.matmul(attn_weights, value_for_attn)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, query_len, -1)
                hidden_states = residual + self.o_proj(attn_output)

                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = residual + self.mlp(hidden_states)
                return hidden_states, out_cache_k, out_cache_v

        return _Wrapper(layer).eval()


@dataclass(frozen=True)
class ExportJob:
    name: str
    module_group: str
    module: object
    inputs: tuple
    inputshape: str
    input_names: tuple[str, ...]
    input_shapes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[str, ...]


def move_to_device(module, inputs: tuple, device: str):
    if device != "cuda":
        return module, inputs
    module = module.cuda()
    return module, tuple(x.cuda() if hasattr(x, "cuda") else x for x in inputs)


def trace_module(torch, module, inputs: tuple, trace_path: Path) -> None:
    with torch.inference_mode():
        traced = torch.jit.trace(module, inputs, strict=False)
        traced = torch.jit.freeze(traced)
        traced.save(str(trace_path))
        output = traced(*inputs)
    if hasattr(output, "shape"):
        print(f"[INFO] Output shape: {tuple(output.shape)}")
    elif isinstance(output, tuple):
        shapes = [tuple(x.shape) for x in output if hasattr(x, "shape")]
        print(f"[INFO] Output shapes: {shapes}")


def run_pnnx(pnnx: str, traced_model: Path, prefix: Path, inputshape: str, device: str) -> int:
    cmd = [
        pnnx,
        str(traced_model),
        f"inputshape={inputshape}",
        f"pnnxparam={prefix}.pnnx.param",
        f"pnnxbin={prefix}.pnnx.bin",
        f"pnnxpy={prefix}_pnnx.py",
        f"pnnxonnx={prefix}.pnnx.onnx",
        f"ncnnparam={prefix}.ncnn.param",
        f"ncnnbin={prefix}.ncnn.bin",
        f"ncnnpy={prefix}_ncnn.py",
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


def remove_existing_param_bin() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = OUTPUT_DIR.resolve()
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file() or path.suffix not in {".param", ".bin"}:
            continue
        if not path.name.startswith(f"{BASE_NAME}_"):
            continue
        if path.resolve().parent != root:
            raise SystemExit(f"[ERROR] Refusing to delete outside output dir: {path}")
        path.unlink()
        print(f"[CLEAN] removed {path.relative_to(OUTPUT_DIR)}")


def cleanup_intermediate_files(keep_intermediate: bool) -> None:
    if keep_intermediate:
        return
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(f"{BASE_NAME}_"):
            continue
        if not (path.name.endswith(".ncnn.param") or path.name.endswith(".ncnn.bin")):
            for attempt in range(5):
                try:
                    path.unlink()
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(1.0)


def list_outputs() -> None:
    print("[OUTPUT]")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"  {path.name:48s} {path.stat().st_size:>14,} bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Export static {MODEL_DIR.name} blocks to NCNN fp16.")
    parser.add_argument("--chunk-frames", type=int, default=100)
    parser.add_argument("--audio-seq-len", type=int, default=390)
    parser.add_argument("--text-seq-len", type=int, default=1)
    parser.add_argument("--past-len", type=int, default=16)
    parser.add_argument(
        "--decoder-layers",
        default="all",
        help="Decoder layer indices to export, e.g. all, 0, 0-3, 0,3,7.",
    )
    parser.add_argument(
        "--modules",
        default="all",
        help=(
            "Module groups to export, e.g. all, audio, text, audio_cnn,audio_transformer,"
            "audio_proj,text_embed,text_norm,lm_head,text_decoder."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--trace-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--pnnx-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pnnx", default=None)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing .param/.bin files under the model output directory before exporting.",
    )
    return parser.parse_args()


def parse_layer_indices(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))

    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))

    unique = sorted(set(indices))
    for index in unique:
        if index < 0 or index >= num_layers:
            raise SystemExit(f"[ERROR] Decoder layer index out of range: {index}")
    return unique


def parse_module_groups(spec: str) -> set[str]:
    aliases = {
        "all": {
            "audio_cnn",
            "audio_transformer",
            "audio_proj",
            "text_embed",
            "text_norm",
            "lm_head",
            "text_decoder",
        },
        "audio": {"audio_cnn", "audio_transformer", "audio_proj"},
        "text": {"text_embed", "text_norm", "lm_head", "text_decoder"},
    }
    valid = aliases["all"]

    selected: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part in aliases:
            selected.update(aliases[part])
        elif part in valid:
            selected.add(part)
        else:
            raise SystemExit(f"[ERROR] Unknown module group: {part}")

    return selected or aliases["all"]


def copy_runtime_files() -> None:
    files = [
        "chat_template.json",
        "configuration.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for name in files:
        src = MODEL_DIR / name
        if not src.is_file():
            continue
        dst = OUTPUT_DIR / name
        shutil.copy2(src, dst)
        print(f"[ASSET] copied {name}")


def write_fbank_assets(torch) -> dict:
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(MODEL_DIR))
    mel_filters = feature_extractor.mel_filters.astype("float32", copy=False)
    hann_window = torch.hann_window(feature_extractor.n_fft).cpu().numpy().astype("float32", copy=False)

    mel_filters_path = OUTPUT_DIR / "mel_filters.f32.bin"
    hann_window_path = OUTPUT_DIR / "hann_window.f32.bin"
    mel_filters.tofile(mel_filters_path)
    hann_window.tofile(hann_window_path)

    fbank_config = {
        "feature_extractor_type": feature_extractor.__class__.__name__,
        "sample_rate": int(getattr(feature_extractor, "sampling_rate", 16000)),
        "chunk_length": int(getattr(feature_extractor, "chunk_length", 30)),
        "n_samples": int(getattr(feature_extractor, "n_samples", 480000)),
        "nb_max_frames": int(getattr(feature_extractor, "nb_max_frames", 3000)),
        "n_fft": int(feature_extractor.n_fft),
        "hop_length": int(feature_extractor.hop_length),
        "feature_size": int(feature_extractor.feature_size),
        "padding_value": float(getattr(feature_extractor, "padding_value", 0.0)),
        "dither": float(getattr(feature_extractor, "dither", 0.0)),
        "mel_filters_file": mel_filters_path.name,
        "mel_filters_shape": list(mel_filters.shape),
        "hann_window_file": hann_window_path.name,
        "hann_window_shape": list(hann_window.shape),
        "note": "Use these constants for C++ streaming fbank; this exporter does not emit a fixed-length fbank ncnn graph.",
    }

    with (OUTPUT_DIR / "fbank_config.json").open("w", encoding="utf-8") as f:
        json.dump(fbank_config, f, indent=2)
        f.write("\n")

    print("[ASSET] wrote fbank_config.json, mel_filters.f32.bin, hann_window.f32.bin")
    return fbank_config


def write_runtime_metadata(torch, model, args: argparse.Namespace, jobs: list[ExportJob]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    thinker = model.model.thinker
    audio_tower = thinker.audio_tower
    text_model = thinker.model
    text_config = text_model.config
    audio_config = audio_tower.config
    fbank_config = write_fbank_assets(torch)
    copy_runtime_files()

    runtime_config = {
        "model_name": MODEL_DIR.name,
        "base_name": BASE_NAME,
        "output_dir": str(OUTPUT_DIR),
        "ncnn_precision": "fp16",
        "trace_defaults": {
            "chunk_frames": args.chunk_frames,
            "audio_seq_len": args.audio_seq_len,
            "text_seq_len": args.text_seq_len,
            "past_len": args.past_len,
        },
        "audio": {
            "num_mel_bins": int(audio_config.num_mel_bins),
            "d_model": int(audio_config.d_model),
            "num_hidden_layers": int(getattr(audio_config, "encoder_layers", len(audio_tower.layers))),
            "n_window": int(getattr(audio_config, "n_window", 0)),
            "n_window_infer": int(getattr(audio_config, "n_window_infer", 0)),
            "conv_chunksize": int(getattr(audio_config, "conv_chunksize", 0)),
            "downsample_hidden_size": int(getattr(audio_config, "downsample_hidden_size", 0)),
        },
        "text": {
            "hidden_size": int(text_config.hidden_size),
            "intermediate_size": int(text_config.intermediate_size),
            "num_hidden_layers": int(text_config.num_hidden_layers),
            "num_attention_heads": int(text_config.num_attention_heads),
            "num_key_value_heads": int(text_config.num_key_value_heads),
            "head_dim": int(getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)),
            "vocab_size": int(text_config.vocab_size),
            "rope_theta": float(getattr(text_config, "rope_theta", 0.0)),
            "rms_norm_eps": float(getattr(text_config, "rms_norm_eps", 0.0)),
        },
        "fbank": fbank_config,
        "runtime_responsibilities": [
            "audio decode/resample to 16 kHz mono",
            "streaming fbank with mel_filters.f32.bin and hann_window.f32.bin",
            "audio chunking, padding, feature length tracking",
            "prompt and audio placeholder assembly",
            "RoPE cos/sin generation",
            "causal attention mask generation",
            "KV cache allocation and updates",
            "decode loop, stopping rules, sampling or greedy selection",
            "tokenizer encode/decode",
        ],
    }

    manifest = {
        "model_name": MODEL_DIR.name,
        "base_name": BASE_NAME,
        "blob_name_convention": "pnnx generated NCNN graphs use in0,in1,... and out0,out1,... unless inspected otherwise.",
        "modules": {
            job.name: {
                "group": job.module_group,
                "param": f"{job.name}.ncnn.param",
                "bin": f"{job.name}.ncnn.bin",
                "inputs": dict(zip(job.input_names, job.input_shapes)),
                "outputs": dict(zip(job.output_names, job.output_shapes)),
            }
            for job in jobs
        },
    }

    with (OUTPUT_DIR / "runtime_config.json").open("w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2)
        f.write("\n")
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("[ASSET] wrote runtime_config.json and manifest.json")


def build_jobs(torch, nn, model, trace_dtype, args: argparse.Namespace) -> list[ExportJob]:
    thinker = model.model.thinker
    audio_tower = thinker.audio_tower
    text_model = thinker.model
    text_config = text_model.config
    audio_dim = audio_tower.config.d_model
    text_dim = text_config.hidden_size
    num_heads = text_config.num_attention_heads
    num_key_value_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", text_dim // num_heads)
    decoder_layer_indices = parse_layer_indices(args.decoder_layers, len(text_model.layers))

    jobs = [
        ExportJob(
            name=f"{BASE_NAME}_audio_cnn",
            module_group="audio_cnn",
            module=AudioCnnForPnnx.build(torch, nn, audio_tower),
            inputs=(torch.randn(1, 1, audio_tower.config.num_mel_bins, args.chunk_frames, dtype=trace_dtype),),
            inputshape=f"[1,1,{audio_tower.config.num_mel_bins},{args.chunk_frames}]",
            input_names=("in0",),
            input_shapes=(f"[1,1,{audio_tower.config.num_mel_bins},{args.chunk_frames}]",),
            output_names=("out0",),
            output_shapes=(f"[1,T,{audio_dim}]",),
        ),
        ExportJob(
            name=f"{BASE_NAME}_audio_transformer",
            module_group="audio_transformer",
            module=AudioTransformerForPnnx.build(torch, nn, audio_tower, args.audio_seq_len),
            inputs=(torch.randn(args.audio_seq_len, audio_dim, dtype=trace_dtype),),
            inputshape=f"[{args.audio_seq_len},{audio_dim}]",
            input_names=("in0",),
            input_shapes=(f"[{args.audio_seq_len},{audio_dim}]",),
            output_names=("out0",),
            output_shapes=(f"[{args.audio_seq_len},{audio_dim}]",),
        ),
        ExportJob(
            name=f"{BASE_NAME}_audio_proj",
            module_group="audio_proj",
            module=AudioProjForPnnx.build(torch, nn, audio_tower),
            inputs=(torch.randn(args.audio_seq_len, audio_dim, dtype=trace_dtype),),
            inputshape=f"[{args.audio_seq_len},{audio_dim}]",
            input_names=("in0",),
            input_shapes=(f"[{args.audio_seq_len},{audio_dim}]",),
            output_names=("out0",),
            output_shapes=(f"[{args.audio_seq_len},{text_dim}]",),
        ),
        ExportJob(
            name=f"{BASE_NAME}_text_embed",
            module_group="text_embed",
            module=text_model.embed_tokens,
            inputs=(torch.randint(0, 100, (1, args.text_seq_len), dtype=torch.long),),
            inputshape=f"[1,{args.text_seq_len}]i64",
            input_names=("in0",),
            input_shapes=(f"[1,{args.text_seq_len}]i64",),
            output_names=("out0",),
            output_shapes=(f"[1,{args.text_seq_len},{text_dim}]",),
        ),
        ExportJob(
            name=f"{BASE_NAME}_text_norm",
            module_group="text_norm",
            module=text_model.norm,
            inputs=(torch.randn(1, args.text_seq_len, text_dim, dtype=trace_dtype),),
            inputshape=f"[1,{args.text_seq_len},{text_dim}]",
            input_names=("in0",),
            input_shapes=(f"[1,{args.text_seq_len},{text_dim}]",),
            output_names=("out0",),
            output_shapes=(f"[1,{args.text_seq_len},{text_dim}]",),
        ),
        ExportJob(
            name=f"{BASE_NAME}_lm_head",
            module_group="lm_head",
            module=thinker.lm_head,
            inputs=(torch.randn(1, args.text_seq_len, text_dim, dtype=trace_dtype),),
            inputshape=f"[1,{args.text_seq_len},{text_dim}]",
            input_names=("in0",),
            input_shapes=(f"[1,{args.text_seq_len},{text_dim}]",),
            output_names=("out0",),
            output_shapes=(f"[1,{args.text_seq_len},{text_config.vocab_size}]",),
        ),
    ]

    total_len = args.past_len + args.text_seq_len
    for layer_idx in decoder_layer_indices:
        jobs.append(
            ExportJob(
                name=f"{BASE_NAME}_text_decoder_layer_{layer_idx:02d}",
                module_group="text_decoder",
                module=TextDecoderLayerForPnnx.build(torch, nn, text_model.layers[layer_idx]),
                inputs=(
                    torch.randn(1, args.text_seq_len, text_dim, dtype=trace_dtype),
                    torch.randn(1, args.text_seq_len, head_dim, dtype=trace_dtype),
                    torch.randn(1, args.text_seq_len, head_dim, dtype=trace_dtype),
                    torch.zeros(1, 1, args.text_seq_len, total_len, dtype=trace_dtype),
                    torch.randn(1, num_key_value_heads, args.past_len, head_dim, dtype=trace_dtype),
                    torch.randn(1, num_key_value_heads, args.past_len, head_dim, dtype=trace_dtype),
                ),
                inputshape=(
                    f"[1,{args.text_seq_len},{text_dim}],"
                    f"[1,{args.text_seq_len},{head_dim}],"
                    f"[1,{args.text_seq_len},{head_dim}],"
                    f"[1,1,{args.text_seq_len},{total_len}],"
                    f"[1,{num_key_value_heads},{args.past_len},{head_dim}],"
                    f"[1,{num_key_value_heads},{args.past_len},{head_dim}]"
                ),
                input_names=("hidden_states", "cos", "sin", "attention_mask", "cache_k", "cache_v"),
                input_shapes=(
                    f"[1,{args.text_seq_len},{text_dim}]",
                    f"[1,{args.text_seq_len},{head_dim}]",
                    f"[1,{args.text_seq_len},{head_dim}]",
                    f"[1,1,{args.text_seq_len},{total_len}]",
                    f"[1,{num_key_value_heads},{args.past_len},{head_dim}]",
                    f"[1,{num_key_value_heads},{args.past_len},{head_dim}]",
                ),
                output_names=("hidden_states", "out_cache_k", "out_cache_v"),
                output_shapes=(
                    f"[1,{args.text_seq_len},{text_dim}]",
                    f"[1,{num_key_value_heads},{total_len},{head_dim}]",
                    f"[1,{num_key_value_heads},{total_len},{head_dim}]",
                ),
            )
        )

    return jobs


def main() -> int:
    args = parse_args()
    if args.chunk_frames <= 0:
        raise SystemExit("[ERROR] --chunk-frames must be positive.")
    if args.audio_seq_len <= 0:
        raise SystemExit("[ERROR] --audio-seq-len must be positive.")
    if args.text_seq_len <= 0:
        raise SystemExit("[ERROR] --text-seq-len must be positive.")
    if args.past_len < 0:
        raise SystemExit("[ERROR] --past-len must be non-negative.")

    pnnx = resolve_pnnx(args.pnnx)
    torch, nn, Qwen3ASRModel = import_runtime()
    device = choose_device(torch, args.device)
    trace_dtype_name, trace_dtype = resolve_trace_dtype(torch, args.trace_dtype)

    print("=" * 72)
    print(f"{MODEL_DIR.name} static blocks -> TorchScript -> pnnx -> NCNN fp16")
    print(f"Model:  {MODEL_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Trace:  {device}/{trace_dtype_name}")
    print("=" * 72)

    if not args.no_clean:
        remove_existing_param_bin()

    model = load_model(torch, Qwen3ASRModel, device, trace_dtype)
    jobs = build_jobs(torch, nn, model, trace_dtype, args)
    write_runtime_metadata(torch, model, args, jobs)
    selected_groups = parse_module_groups(args.modules)
    export_jobs = [job for job in jobs if job.module_group in selected_groups]

    for job in export_jobs:
        print("-" * 72)
        print(f"[INFO] Exporting {job.name}")
        module, inputs = move_to_device(job.module, job.inputs, device)
        trace_path = OUTPUT_DIR / f"{job.name}.pt"
        prefix = OUTPUT_DIR / job.name
        trace_module(torch, module, inputs, trace_path)
        code = run_pnnx(pnnx, trace_path, prefix, job.inputshape, args.pnnx_device)
        if code != 0:
            return code

    cleanup_intermediate_files(args.keep_intermediate)
    list_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
