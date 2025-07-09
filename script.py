import opencv as cv
from typing import List
import numpy as n
from gabor_strategy import GaborStrategy
from strategy import Strategy

gabor_strategy = GaborStrategy()


def check_similarity(images: List[np.ndarray], strategy: Strategy) -> float:
    for image in images:
        assert image != None
    print(strategy.get_similarity(images))
    return 0.0

def main():
    print("Hello from tool!")


if __name__ == "__main__":
    main()
