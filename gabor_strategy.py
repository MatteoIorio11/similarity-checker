from typing import List, Tuple

import cv2 as cv
import numpy as np
from strategy import Strategy


class GaborStrategy(Strategy):
    filters: List[np.ndarray] = []

    def __init__(self):
        self.filters: List[np.ndarray] = self.generate_filters()

    def get_similarity(self, images: List[np.ndarray]) -> float:
        for image in images:
            a = self.similarity_for_image(image)
        return 0.1

    def similarity_for_image(self, image: np.ndarray) -> float:
        grid_size: Tuple[int, int] = (8, 8)
        patch_size: int = 32
        height: int = image.shape[0]
        width: int = image.shape[1]

        return 0.0

    def generate_filters(self) -> List[np.ndarray]:
        ksize: int = 21
        orientations: int = 4
        sigma: float = 4.0
        glambda: float = 10.0
        gamma: float = 0.5
        psi: int = 0
        return [cv.getGaborKernel((ksize, ksize), sigma, theta,
                                  glambda, gamma, psi, ktype=cv.CV_32F)
                for theta in range(0, np.pi, np.pi / orientations)]

