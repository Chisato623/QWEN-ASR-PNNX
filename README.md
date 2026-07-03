# QWEN-ASR-PNNX

Utilities for exporting Qwen3-ASR and Whisper models with PNNX/NCNN.

This repository keeps conversion and runtime code in Git while excluding local
model weights, virtual environments, caches, and generated model artifacts.

## Layout

- `convert_qwen3_asr_0_6b_to_ncnn.py` exports the Qwen3-ASR-0.6B audio encoder.
- `convert_qwen3_asr_1_7b_to_ncnn.py` exports the Qwen3-ASR-1.7B audio encoder.
- `export_ncnn.py` exports Whisper components with PNNX.
- `run_asr.py` runs a local Qwen3-ASR transcription demo.
- `Qwen3-ASR/` contains the Qwen3-ASR Python package source used by the scripts.

Local model directories such as `Qwen3-ASR-0.6B/`, `Qwen3-ASR-1.7B/`,
`whisper/`, and generated `ncnn_export/` files are intentionally ignored when
they contain model weights or exported artifacts.
