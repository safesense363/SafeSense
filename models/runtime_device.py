"""Select inference devices and hardware-appropriate model variants."""
import torch


CPU_POSE_MODEL = "yolo26n-pose.pt"
CUDA_POSE_MODEL = "yolo26s-pose.pt"


def resolve_runtime_device(requested: str = "auto") -> torch.device:
    """Resolve auto/CPU/CUDA selection and fail clearly for unavailable CUDA."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled "
            "PyTorch build and verify that the NVIDIA driver is available, "
            "or run with --device cpu."
        )
    if requested == "cpu" or not cuda_available:
        return torch.device("cpu")
    return torch.device("cuda:0")


def pose_model_for_device(device: torch.device | str) -> str:
    """Use the compact pose checkpoint on CPU and a larger one on CUDA."""
    return CUDA_POSE_MODEL if torch.device(device).type == "cuda" else CPU_POSE_MODEL
