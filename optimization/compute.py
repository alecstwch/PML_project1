"""Device reporting and duration formatting."""

from typing import Dict


def fmt_hms(seconds: float) -> str:
    """Format a duration as m:ss or h:mm:ss."""
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def report_compute() -> Dict[str, object]:
    """Print CPU/GPU devices and allow TensorFlow to grow GPU memory as needed."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    info = {
        "tf": tf.__version__,
        "cuda_built": bool(tf.test.is_built_with_cuda()),
        "gpu_names": [gpu.name for gpu in gpus],
    }
    print("TensorFlow", info["tf"], "| CUDA build:", info["cuda_built"])
    if gpus:
        print("Using GPU:", info["gpu_names"])
    else:
        print("No GPU visible. For CUDA, run this notebook with the WSL kernel (pml_venv).")
    return info
