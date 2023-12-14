from copy import copy

import numpy as np
import itertools
from enum import Enum, auto
from functools import partial, lru_cache
from pathlib import Path
from typing import Any, Generator

from functional import seq

from transformer import Transformer


class TransformerImpl(Transformer):

    def transform_2(self, data: str) -> Any:
        arr: np.ndarray = np.vstack([seq(data.splitlines(keepends=False)).map(list).to_list()])
        arr = ((arr == 'O').astype(int) * 1) + ((arr == '#').astype(int) * 2)

        result = self._cycle(arr)
        return result

    def _cycle(self, arr: np.ndarray) -> int:
        num_iterations = 1_000_000_000
        results = np.empty(num_iterations, dtype=int)
        for i in range(num_iterations):
            # north
            self._tilt_north(arr)
            # west
            self._rotate_clockwise(arr)
            self._tilt_north(arr)
            # south
            self._rotate_clockwise(arr)
            self._tilt_north(arr)
            # east
            self._rotate_clockwise(arr)
            self._tilt_north(arr)
            # north
            self._rotate_clockwise(arr)

            result = sum(
                (i + 1) * (row == 1).sum()
                for i, row in enumerate(arr[::-1, :])
            )
            results[i] = result
            if (repeated_sequence_width := self._has_repeating_sequence(results[:i])) > 0:
                pass
                print(f"repeating sequence found: {results[i-repeated_sequence_width:i]}")
                result = self._project_repeating_sequence(results[i-repeated_sequence_width:i], num_iterations - i)
                return result

    def _project_repeating_sequence(self, arr: np.ndarray, num_iterations: int) -> int:
        return arr[num_iterations % len(arr) - 1]

    def _has_repeating_sequence(self, arr: np.ndarray) -> int:
        for width in range(2, len(arr) // 2 + 1):
            if np.array_equal(arr[-width:], arr[-2*width:-width]):
                return width
        return 0

    def _rotate_clockwise(self, arr: np.ndarray) -> None:
        arr[:] = arr[::-1, :].T

    def transform_1(self, data: str) -> Any:
        arr: np.ndarray = np.vstack([seq(data.splitlines(keepends=False)).map(list).to_list()])
        arr = ((arr == 'O').astype(int) * 1) + ((arr == '#').astype(int) * 2)

        self._tilt_north(arr)
        result = sum(
            (i + 1) * (row == 1).sum()
            for i, row in enumerate(arr[::-1, :])
        )

        return result

    def _tilt_north(self, arr: np.ndarray) -> None:
        for col in range(arr.shape[1]):
            index = 0
            round_rocks = (arr[index:, col] == 1)
            while round_rocks.any():
                round_rock_index = np.where(round_rocks)[0][0]
                old_rock_index = index + round_rock_index
                all_rocks = arr[index:old_rock_index, col] > 0
                rock_above_in_range = all_rocks.any()
                if rock_above_in_range:
                    non_rock_index = index + np.where(all_rocks)[0][-1] + 1
                else:
                    non_rock_index = index

                new_rock_index = non_rock_index
                arr[old_rock_index, col] = 0
                arr[new_rock_index, col] = 1

                index = new_rock_index + 1
                round_rocks = (arr[index:, col] == 1)



if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    print(sut.transform_2(data))
