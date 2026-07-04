"""
Export static Qwen3-ASR-1.7B weight blocks to NCNN fp16 with pnnx.

This reuses the 0.6B exporter implementation and only changes the model
directory, output directory, and file-name prefix.
"""

from __future__ import annotations

import export_ncnn_qwenasr as exporter


exporter.MODEL_DIR = exporter.ROOT_DIR / "Qwen3-ASR-1.7B"
exporter.OUTPUT_DIR = exporter.ROOT_DIR / "ncnn_export" / "Qwen3-ASR-1.7B"
exporter.BASE_NAME = "qwen3_asr_1_7b"


if __name__ == "__main__":
    raise SystemExit(exporter.main())
