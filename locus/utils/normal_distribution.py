import numpy as np


def normal_distribution(x, mean, std):
    coefficient = 1 / (std * np.sqrt(2 * np.pi))
    exponent = -((x - mean) ** 2) / (2 * std**2)
    return coefficient * np.exp(exponent)


if __name__ == "__main__":
    # Test normal_distribution
    x = 0
    mean = 0
    std = 1
    print(normal_distribution(x, mean, std))  # 0.3989422804014327
    x = 1
    print(normal_distribution(x, mean, std))  # 0.24197072451914337
    x = 2
    print(normal_distribution(x, mean, std))  # 0.05399096651318806
