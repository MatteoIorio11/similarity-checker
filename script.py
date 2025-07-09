import opencv as cv
from typing import List, Callable
import numpy as np

def check_similarity(images: List[np.ndarray], strategy: Callable[[List[np.ndarray]], float]) -> float:
    for image in images:
        assert image != None
    return strategy(images)

def main():
    print("Hello from tool!")


if __name__ == "__main__":
    main()
