import pynvml
import os

def pick_best_gpu(policy="free-mem"):
    """
    Return the index of the “least busy” NVIDIA GPU and set CUDA_VISIBLE_DEVICES
    so frameworks (PyTorch, TensorFlow, JAX…) will automatically use it.

    policy
    ------
    "free-mem"   – prefer the card with the most free memory
    "low-util"   – prefer the card with the lowest compute utilisation
    "hybrid"     – most free mem, break ties with lowest utilisation
    """
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()

    best_idx, best_score = None, None
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)          # bytes
        util = pynvml.nvmlDeviceGetUtilizationRates(h)   # %
        if policy == "free-mem":
            score = mem.free
        elif policy == "low-util":
            score = -util.gpu                            # negative ⇒ lower is better
        else:  # hybrid
            score = (mem.free, -util.gpu)                # tuple is fine for max()

        if best_score is None or score > best_score:
            best_idx, best_score = i, score

    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)   # frameworks see *only* this GPU
    print(f"👉  Selected GPU {best_idx}")
    return best_idx