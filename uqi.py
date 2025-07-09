from typing import List

import numpy as np
from sewar.full_ref import uqi
from strategy import Strategy


class UQI(Strategy):
    def get_similarity(self, images: List[np.ndarray]) -> float:
        return uqi(images[0], images[1])
