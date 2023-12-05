from dataclasses import dataclass
from functools import partial

import numpy as np
import re
from pathlib import Path
from typing import Any

from transformer import Transformer
from functional import seq


@dataclass
class Number:
    value: int
    indices: list[int]
    line_number: int


@dataclass
class Symbol:
    index: int
    line_number: int


class TransformerImpl(Transformer):

    def transform_1(self, data: str) -> Any:
        numbers = []
        lines = []
        splitlines = data.splitlines(keepends=False)
        empty_grid = np.zeros((len(splitlines), len(splitlines[0])), dtype=bool)
        symbols = empty_grid.copy()
        for line_number, line in enumerate(splitlines):
            current_number = []
            for i, character in enumerate(line):
                if character.isdigit():
                    current_number.append(character)
                else:
                    if current_number:
                        numbers.append(
                            Number(
                                value=int("".join(current_number)),
                                indices=[i - len(current_number), i - 1],
                                line_number=line_number
                            )
                        )
                        current_number = []
                    if character != ".":
                        symbols[line_number, i] = True
            if current_number:
                numbers.append(
                    Number(
                        value=int("".join(current_number)),
                        indices=[len(line) - len(current_number), len(line) - 1],
                        line_number=line_number
                    )
                )
            lines.append(line)

        result = (
            seq(numbers)
            .filter(partial(self._number_is_adjacent_to_symbol, symbols, empty_grid))
            .map(lambda number: number.value)
            .sum()
        )
        return result

    def _number_is_adjacent_to_symbol(self, symbols, empty_grid, number):
        adjacency_matrix = empty_grid

        indices = number.indices
        line_number = number.line_number

        self._set_adjacency(adjacency_matrix, line_number, indices)

        result = np.any(np.logical_and(adjacency_matrix, symbols))
        empty_grid[:, :] = False
        return result

    def _set_adjacency(self, adjacency_matrix, line_number, indices):
        line = line_number - 1
        height = adjacency_matrix.shape[0]
        width = adjacency_matrix.shape[1]
        if line >= 0:
            adjacency_matrix[line, max(0, indices[0] - 1):min(indices[-1] + 2, width - 1)] = True
        line = line_number + 1
        if line < height:
            adjacency_matrix[line, max(0, indices[0] - 1):min(indices[-1] + 2, width - 1)] = True
        index = indices[0] - 1
        if index >= 0:
            adjacency_matrix[line_number, index] = True
        index = indices[-1] + 1
        if index < width:
            adjacency_matrix[line_number, index] = True

    def transform_2(self, data: str) -> Any:
        splitlines = data.splitlines(keepends=False)
        height = len(splitlines)
        width = len(splitlines[0])
        empty_grid = np.zeros((height, width), dtype=bool)
        gears: list[Symbol] = []
        numbers: list[Number] = []
        for line_number, line in enumerate(splitlines):
            current_number = []
            for i, character in enumerate(line):
                if character.isdigit():
                    current_number.append(character)
                else:
                    if current_number:
                        numbers.append(
                            Number(
                                value=int("".join(current_number)),
                                indices=[i - len(current_number), i - 1],
                                line_number=line_number
                            )
                        )
                        current_number = []
                    if character == "*":
                        gears.append(Symbol(index=i, line_number=line_number))
            if current_number:
                numbers.append(
                    Number(
                        value=int("".join(current_number)),
                        indices=[len(line) - len(current_number), len(line) - 1],
                        line_number=line_number
                    )
                )

        result = (
            seq(gears)
            .map(partial(self._product_of_gear_numbers_if_is_missing_gear, numbers, empty_grid))
            .sum()
        )
        return result

    def _product_of_gear_numbers_if_is_missing_gear(self, numbers, empty_grid, gear):
        gear_adjacency_matrix = empty_grid.copy()

        index = gear.index
        line_number = gear.line_number
        gear_adjacency_matrix[line_number, index] = True

        numbers_are_adjacent = (
            seq(numbers)
            .filter(partial(self._number_is_adjacent_to_gear, gear_adjacency_matrix, empty_grid))
            .to_list()
        )
        is_missing_gear = len(numbers_are_adjacent) == 2
        if is_missing_gear:
            result = (
                seq(numbers_are_adjacent)
                .map(lambda number: number.value)
                .product()
            )
        else:
            result = 0
        return result

    def _number_is_adjacent_to_gear(self, gear_adjacency_matrix, empty_grid, number):
        number_adjacency_matrix = empty_grid

        indices = number.indices
        line_number = number.line_number

        self._set_adjacency(number_adjacency_matrix, line_number, indices)

        result = np.any(np.logical_and(number_adjacency_matrix, gear_adjacency_matrix))
        empty_grid[:, :] = False
        return result


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    print(sut.transform_2(data))
