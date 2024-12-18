from typing import Tuple
import numpy as np


class LBPDescriptor:
    def __init__(self, radius: int = 1, n_neighbors: int = 8):
        if radius < 1:
            raise ValueError("The radius must be greater than 0.")
        self.radius = radius

        # Ensure n_neighbors is multiple of eight
        if n_neighbors % 8 != 0:
            raise ValueError(
                "The number of neighbors (n_neighbors) must be multiple of 8."
            )
        if n_neighbors < 8:
            raise ValueError(
                "The number of neighbors (n_neighbors) must be equal 8 or greater."
            )
        self.n_neighbors = n_neighbors

    def compute(self, img: np.ndarray):
        lbps = []
        
        # Grid size generated based on a radius
        grid_size = 2 * self.radius + 1

        m = img.shape[0]

        # Initial center pixel based on grid size (mxm)
        center = (grid_size - 1) // 2

        for i in range(0, img.shape[0] - center, grid_size):
            for j in range(0, img.shape[1] - center, grid_size):
                center_pixel = (i + center, j + center)
                neighbors = self._get_coordinates_neighbors(center_pixel)

                if any([x > m or y > m or x < 0 or y < 0 for x, y in neighbors]):
                    raise ValueError(
                        "Combination of radius and number of neighbors exceeded image size."
                    )
                lbp = [
                    1
                    if self._bilinear_interpolation(img, neighbor_pixel)
                    > img[center_pixel]
                    else 0
                    for neighbor_pixel in neighbors
                ]
            lbps.append(lbp)
        return lbps

    def _get_coordinates_neighbors(self, pixel: Tuple[int]):
        """
        This function computes the neighbors surrounding a pixel in a circle knowing
        the radius of it and the number of neighbors in clockwise order.

        Args:
            pixel (Tuple[int]): Coordinate of a pixel (x,y).
        Returns:
            List: List of neighbors surrounding a pixel.
        """
        return [
            (
                pixel[0] + self.radius * np.cos(-2 * np.pi * i / self.n_neighbors),
                pixel[1] + self.radius * np.sin(-2 * np.pi * i / self.n_neighbors),
            )
            for i in range(self.n_neighbors)
        ]

    @classmethod
    def _bilinear_interpolation(
        cls, img: np.ndarray, pixel: Tuple[float | int, float | int]
    ):
        """
        Performs bilinear interpolation on the given image at the specified (x, y) coordinates.

        Args:
            img (numpy.ndarray): The input image on which interpolation is to be performed.
            pixel (Tuple[int]): Coordinate of a pixel (x,y).

        Returns:
            float: The interpolated pixel value at the specified (x, y) coordinates.
        """
        # If pixel coordinates are integers, return the direct pixel value
        if isinstance(pixel[0], int) and isinstance(pixel[1], int):
            return img[int(pixel[0]), int(pixel[1])]

        # Get the four surrounding pixel coordinates
        x, y = pixel
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img.shape[1] - 1))
        x2 = max(0, min(x2, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 1))
        y2 = max(0, min(y2, img.shape[0] - 1))

        # Get surrounding pixel values
        Q11 = img[y1, x1]
        Q12 = img[y2, x1]
        Q21 = img[y1, x2]
        Q22 = img[y2, x2]

        # Compute interpolation weights
        wx = x - x1
        wy = y - y1

        # Perform bilinear interpolation
        interpolated_value = (
            Q11 * (1 - wx) * (1 - wy)
            + Q21 * wx * (1 - wy)
            + Q12 * (1 - wx) * wy
            + Q22 * wx * wy
        )

        return interpolated_value
