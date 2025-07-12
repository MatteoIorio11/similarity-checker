from typing import List
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from strategy import Strategy


def get_metrics(glcm: np.ndarray):
    return {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'entropy': -np.sum(glcm * np.log2(glcm + 1e-10)),  # Custom entropy
        'correlation': graycoprops(glcm, 'correlation')[0, 0]
    }


class GLCM(Strategy):
    def get_similarity(self, images: List[np.ndarray]) -> float:
        features1 = get_metrics(graycomatrix(images[0], [1], [0], 256, True, True))
        features2 = get_metrics(graycomatrix(images[1], [1], [0], 256, True, True))
        vec1 = np.array(list(features1.values()))
        vec2 = np.array(list(features2.values()))
        return float(np.linalg.norm(vec1 - vec2))

