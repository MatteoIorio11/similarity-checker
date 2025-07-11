from typing import List, Set
import os
import numpy as np
from gabor_strategy import GaborStrategy
from uqi import UQI
from strategy import Strategy
import tkinter as tk
from tkinter import filedialog
import cv2 as cv
from ssim import SSIM

gabor_strategy: Strategy = GaborStrategy()
uqi: Strategy = UQI()
ssim: Strategy = SSIM()


def check_similarity(images: List[np.ndarray], strategy: Strategy) -> float:
    for image in images:
        assert image is not None
    return strategy.get_similarity(images)


def main():
    images: List[np.ndarray] = reshape_images(read_images(""))
    cosine_score: float = check_similarity(images, gabor_strategy)
    uqi_score: float = check_similarity(images, uqi)
    ssim_score: float = check_similarity(images, ssim)

    print(f"Similarity Gabor: {cosine_score}")
    print(f"Similarity UQI: {uqi_score}")
    print(f"Similarity SSIM: {ssim_score}")


def reshape_images(images: List[np.ndarray]) -> List[np.ndarray]:
    min_width: float = float('inf')
    min_height: float = float('inf')
    new_images: List[np.ndarray] = []
    for image in images:
        height, width = image.shape[:2]
        min_height: float = min(min_height, height)
        min_width: float = min(min_width, width)
    for image in images:
        new_images.append(cv.resize(image, [int(min_width), int(min_height)]))
    return new_images


def read_images(prefix: str) -> List[np.ndarray]:
    root = tk.Tk()
    root.withdraw()
    file_path: str = filedialog.askdirectory(initialdir="Desktop")
    files: List[str] = os.listdir(file_path)
    image_format: Set[str] = {"jpg", "png", "jpeg", "tif"}
    return [cv.imread(os.path.join(file_path, file), cv.IMREAD_GRAYSCALE)
            for file in files if file.split(".")[1] in image_format and prefix in file]


if __name__ == "__main__":
    main()
