from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt


def get_interpolation_function(
    original_image: npt.NDArray,
    map_transform: npt.NDArray,
    method: str = "nearest_neighbor",
) -> Callable:
    max_x = original_image.shape[1] - 1
    max_y = original_image.shape[0] - 1

    def get_original_coordinates_and_distances(
        new_y: int, new_x: int
    ) -> Tuple[float, float, float, float]:
        new_coordinates = np.array([new_x, new_y, 1])
        x_, y_, _ = np.matmul(map_transform, new_coordinates)
        dx, dy = x_ - int(x_), y_ - int(y_)

        return x_, y_, dx, dy

    def nearest_neighbor(new_y: int, new_x: int) -> int:
        x_, y_, _, _ = get_original_coordinates_and_distances(new_y, new_x)

        y_nearest, x_nearest = (round(y_), round(x_))

        # Check if the neighbor exists within original image
        if 0 <= y_nearest <= max_y and 0 <= x_nearest <= max_x:
            return original_image[(y_nearest, x_nearest)]

        return 0

    def bilinear(new_y: int, new_x: int) -> int:
        x_, y_, dx, dy = get_original_coordinates_and_distances(new_y, new_x)

        coords = {
            (n, m): (int(y_ + n), int(x_ + m)) for n in range(0, 2) for m in range(0, 2)
        }
        valid_coords = {
            k: (h, g)
            for k, (h, g) in coords.items()
            if 0 <= h <= max_y and 0 <= g <= max_x
        }

        def T(d, i):
            return d * i + (1 - d) * (1 - i)

        value = 0
        for (n, m), coord in valid_coords.items():
            value += T(dy, n) * T(dx, m) * original_image[coord]

        return value

    def bicubic(new_y: int, new_x: int) -> int:
        x_, y_, dx, dy = get_original_coordinates_and_distances(new_y, new_x)

        def L(n):
            coords = [(int(y_ + n - 2), int(x_ + e)) for e in range(-1, 3)]

            p = []
            p.append((-dx * (dx - 1) * (dx - 2)) / 6)
            p.append(((dx + 1) * (dx - 1) * (dx - 2)) / 2)
            p.append((-dx * (dx + 1) * (dx - 2)) / 2)
            p.append((dx * (dx + 1) * (dx - 1)) / 6)

            partial = 0
            for i, (h, g) in enumerate(coords):
                if 0 <= h <= max_y and 0 <= g <= max_x:
                    partial += p[i] * original_image[h, g]
            return partial

        value = (-dy * (dy - 1) * (dy - 2) * L(1)) / 6
        value += ((dy + 1) * (dy - 1) * (dy - 2) * L(2)) / 2
        value += (-dy * (dy + 1) * (dy - 2) * L(3)) / 2
        value += (dy * (dy + 1) * (dy - 1) * L(4)) / 6

        return value

    def lagrange(new_y: int, new_x: int) -> int:
        x_, y_, dx, dy = get_original_coordinates_and_distances(new_y, new_x)

        coords = {
            (n, m): (int(y_ + n), int(x_ + m))
            for n in range(-1, 3)
            for m in range(-1, 3)
        }
        valid_coords = {
            k: (h, g)
            for k, (h, g) in coords.items()
            if 0 <= h <= max_y and 0 <= g <= max_x
        }

        def R(s):
            return (
                (P(s + 2) ** 3)
                - 4 * (P(s + 1) ** 3)
                + 6 * (P(s) ** 3)
                - 4 * (P(s - 1) ** 3)
            ) / 6

        def P(t):
            return 0 if t < 0 else t

        value = 0
        for (n, m), coord in valid_coords.items():
            value += R(dy - n) * R(m - dx) * original_image[coord]

        return value

    methods = {
        "nearest_neighbor": nearest_neighbor,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "lagrange": lagrange,
    }

    def interpolation_function(i: npt.NDArray, j: npt.NDArray) -> npt.NDArray:
        vectorized_value_for_coordinate = np.vectorize(methods[method])
        return vectorized_value_for_coordinate(i, j)

    return interpolation_function
