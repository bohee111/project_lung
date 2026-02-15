"""재현성을 위한 시드 고정 유틸리티."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """파이썬/NumPy/PyTorch 시드를 동시에 고정한다.

    인자:
        seed (int): 고정할 시드 값.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
