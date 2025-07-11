from typing import List
import cv2 as cv
import numpy as np

from strategy import Strategy
from sklearn.metrics.pairwise import cosine_similarity


def get_descriptors(image: np.ndarray, patch_radius: int = 1, search_radius: int = 1,
                    sigma: float = 0.5) -> np.ndarray:
    h, w = image.shape
    pad = patch_radius + search_radius

    # Gaussian kernel
    ksize: int = 2 * patch_radius + 1  # patch size
    gk1d: np.ndarray = cv.getGaussianKernel(ksize, sigma)
    gk2d: np.ndarray = gk1d @ gk1d.T

    # Allocate descriptor volume
    descriptor = np.zeros((h, w, (2 * search_radius + 1) ** 2 - 1), dtype=np.float32)

    # Pad image
    image = image.astype(np.float32)
    padded = np.pad(image, pad, mode='reflect')

    # Center patch (always same region)
    center_patch = padded[pad:-pad, pad:-pad]
    center_patch = cv.filter2D(center_patch, cv.CV_32F, gk2d)

    idx = 0
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue

            y1 = pad + dy
            y2 = y1 + h
            x1 = pad + dx
            x2 = x1 + w

            shifted = padded[y1:y2, x1:x2]
            shifted = cv.filter2D(shifted, cv.CV_32F, gk2d)

            dist = (center_patch - shifted) ** 2
            descriptor[..., idx] = dist
            idx += 1

    # Normalize the descriptor
    mean: float = np.mean(descriptor, axis=-1, keepdims=True)
    descriptor = np.exp(-descriptor / (mean + 1e-6))
    descriptor /= np.sum(descriptor, axis=-1, keepdims=True) + 1e-6

    return descriptor


class MIND(Strategy):
    def get_similarity(self, images: List[np.ndarray]) -> float:
        descriptor1: np.ndarray = get_descriptors(images[0])
        descriptor2: np.ndarray = get_descriptors(images[1])
        assert descriptor1.shape == descriptor2.shape, "Descriptor shapes must match"
        H, W, D = descriptor1.shape

        # Define patch/grid size (e.g., 32x32 blocks)
        grid_size = 32
        similarities = []

        for y in range(0, H, grid_size):
            for x in range(0, W, grid_size):
                y_end = min(y + grid_size, H)
                x_end = min(x + grid_size, W)

                patch1 = descriptor1[y:y_end, x:x_end].reshape(-1, D)
                patch2 = descriptor2[y:y_end, x:x_end].reshape(-1, D)

                if patch1.shape[0] < 2:
                    continue  # Skip very small patches

                # Compute cosine similarity for the patch
                sim = cosine_similarity(patch1, patch2).diagonal().mean()
                similarities.append(sim)

        # Return average local similarity
        return float(np.mean(similarities))
