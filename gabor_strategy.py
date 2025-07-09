import random
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from strategy import Strategy

import cv2 as cv
import numpy as np


def generate_filters() -> List[np.ndarray]:
    ksize: int = 21
    orientations: int = 4
    sigma: float = 4.0
    g_lambda: float = 10.0
    gamma: float = 0.5
    psi: int = 0
    return [cv.getGaborKernel((ksize, ksize), sigma, theta,
                              g_lambda, gamma, psi, ktype=cv.CV_32F)
            for theta in np.arange(0, np.pi, np.pi / orientations)]


class GaborStrategy(Strategy):
    filters: List[np.ndarray] = []

    def __init__(self):
        self.filters: List[np.ndarray] = generate_filters()

    def get_similarity(self, images: List[np.ndarray]) -> float:
        scores = []
        for image in images:
            scores.append(normalize([self.similarity_for_image(image)])[0])
        return cosine_similarity([scores[0]], [scores[1]])[0][0]

    def similarity_for_image(self, image: np.ndarray) -> np.ndarray:
        grid_size: Tuple[int, int] = (8, 8)
        patch_size: int = 32
        height: int = image.shape[0]
        width: int = image.shape[1]
        step_y: int = height // grid_size[0]
        step_x: int = width // grid_size[1]
        features: List[float] = []
        filtered: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        for y in range(0, height - patch_size + 1, step_y):
            for x in range(0, width - patch_size + 1, step_x):
                patch: np.ndarray = image[y:y + patch_size, x:x + patch_size]
                for current_filter in self.filters:
                    filtered: np.ndarray = cv.filter2D(patch, cv.CV_32F, current_filter)
                features.extend([np.sum(filtered ** 2), filtered.std()])
        return np.array(features)
