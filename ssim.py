from skimage.metrics import structural_similarity as ssim
from typing import List
import numpy as np
from strategy import Strategy

class SSIM(Strategy):
    def get_similarity(self, images: List[np.ndarray]) -> float:
        score, diff = ssim(images[0], images[1], full=True)
        return score
