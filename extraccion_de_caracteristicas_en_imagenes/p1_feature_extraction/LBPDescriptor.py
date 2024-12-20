import numpy as np


class LBPDescriptor:
    def __init__(self, radius: int = 1, n_neighbors: int = 8):
        if radius < 1:
            raise ValueError("The radius must be greater than 0.")
        if n_neighbors % 8 != 0 or n_neighbors < 8:
            raise ValueError(
                "The number of neighbors (n_neighbors) must be a multiple of 8 and at least 8."
            )
        self.radius = radius
        self.n_neighbors = n_neighbors

        # Precompute relative offsets for neighbors
        self.neighbor_offsets = [
            (
                radius * np.cos(2 * np.pi * i / n_neighbors),
                radius * np.sin(2 * np.pi * i / n_neighbors),
            )
            for i in range(n_neighbors)
        ]

    def compute(self, img: np.ndarray) -> np.ndarray:
        """
        Computes the Local Binary Pattern (LBP) for the given image.

        Args:
            img (np.ndarray): Grayscale image as a 2D numpy array.
        Returns:
            np.ndarray: Flattened normalized histogram of LBP values.
        """
        if len(img.shape) != 2:
            raise ValueError("Input image must be a 2D grayscale image.")

        rows, cols = img.shape
        lbps = np.zeros(
            (rows - 2 * self.radius, cols - 2 * self.radius), dtype=np.uint8
        )

        for i in range(self.radius, rows - self.radius):
            for j in range(self.radius, cols - self.radius):
                center_pixel = img[i, j]
                binary_pattern = 0

                for idx, (dx, dy) in enumerate(self.neighbor_offsets):
                    neighbor_value = self._bilinear_interpolation(img, i + dy, j + dx)
                    binary_pattern |= (neighbor_value > center_pixel) << idx

                lbps[i - self.radius, j - self.radius] = binary_pattern

        # Compute histogram of LBP values
        hist, _ = np.histogram(lbps.flatten(), bins=np.arange(255), range=(0, 255))
        return hist / hist.sum()

    @staticmethod
    def _bilinear_interpolation(img: np.ndarray, y: float, x: float) -> float:
        """
        Performs bilinear interpolation on the given image at the specified (x, y) coordinates.
        https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python

        Args:
            img (np.ndarray): The input image.
            x, y (float): Coordinates for interpolation.
        Returns:
            float: Interpolated pixel value.
        """
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1

        # Ensure coordinates are within image bounds
        if not (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]):
            return 0
        x2 = min(x2, img.shape[1] - 1)
        y2 = min(y2, img.shape[0] - 1)

        Q11 = img[y1, x1]
        Q21 = img[y1, x2]
        Q12 = img[y2, x1]
        Q22 = img[y2, x2]

        wx = x - x1
        wy = y - y1

        return (
            Q11 * (1 - wx) * (1 - wy)
            + Q21 * wx * (1 - wy)
            + Q12 * (1 - wx) * wy
            + Q22 * wx * wy
        )
