from typing import List
import numpy as np
from strategy import Strategy


class GaborStrategy(Strategy):
    def get_similarity(self, images: List[np.ndarray]) -> float:
        return 0.1
    