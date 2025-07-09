from typing import List, Set
import os
import numpy as np
from gabor_strategy import GaborStrategy
from strategy import Strategy
import cv2 as cv

gabor_strategy = GaborStrategy()


def check_similarity(images: List[np.ndarray], strategy: Strategy) -> float:
    for image in images:
        assert image is not None
    return strategy.get_similarity(images)


def main():
    images: List[np.ndarray] = read_images()
    score: float = check_similarity(images, gabor_strategy)
    print(f"Similarity {score}")


def read_images() -> List[np.ndarray]:
    files: List[str] = os.listdir("./images")
    image_format: Set[str] = {"jpg", "png", "jpeg"}
    return [cv.imread(os.path.join("./images", file), cv.IMREAD_GRAYSCALE) for file in files if file.split(".")[1] in image_format]


if __name__ == "__main__":
    main()
