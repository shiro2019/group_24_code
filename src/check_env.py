"""Sanity check for a compute node: GPU, library imports and Hugging Face access.

    sbatch --job-name=check --time=00:20:00 slurm/job.sbatch src.check_env
"""

import urllib.error
import urllib.request

import torch

print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    print(f"fp16 matmul ok: {(x @ x).float().abs().mean().item():.3f}")

for name in ("diffusers", "transformers", "peft", "accelerate", "cleanfid", "lpips", "torchmetrics"):
    try:
        mod = __import__(name)
        print(f"{name} {getattr(mod, '__version__', '?')} ok")
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {e}")

try:
    from src.evaluate import _patch_transformers_for_imagereward
    _patch_transformers_for_imagereward()
    import ImageReward  # noqa: F401
    print("ImageReward import ok")
except Exception as e:
    print(f"ImageReward FAILED: {type(e).__name__}: {e}")

for url in ("https://huggingface.co", "http://images.cocodataset.org", "https://github.com"):
    try:
        urllib.request.urlopen(url, timeout=10)
        print(f"internet ok: {url}")
    except urllib.error.HTTPError as e:
        print(f"internet ok: {url} (HTTP {e.code})")
    except Exception as e:
        print(f"internet FAILED: {url} ({type(e).__name__})")
