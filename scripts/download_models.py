#!/usr/bin/env python3
"""Download target models for SLO-Guard experiments.

Downloads via HuggingFace Hub. Requires: pip install huggingface_hub
"""
from __future__ import annotations

import sys

MODELS = [
    "Qwen/Qwen2-1.5B",
    "microsoft/phi-2",
    "mistralai/Mistral-7B-v0.1",
]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    for model_id in MODELS:
        print(f"Downloading {model_id}...")
        try:
            snapshot_download(
                model_id,
                ignore_patterns=["*.bin", "*.safetensors.index.json"],
            )
            print(f"  Done: {model_id}")
        except Exception as e:
            print(f"  Warning: Failed to download {model_id}: {e}")
            print("  (You may need to accept model terms on HuggingFace)")

    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()
