import itertools
import re

import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from functional import seq

from transformer import Transformer


@dataclass
class Galaxy:
    x: int
    y: int
    number: int


class TransformerImpl(Transformer):

    def __init__(self):
        self._expansion_factor = 2

    def _expand_columns(self, galaxies: list[Galaxy]) -> None:
        max_x = (
            seq(galaxies)
            .map(lambda galaxy: galaxy.x)
            .max()
        )
        empty_columns = []
        for x in range(max_x * 2):
            num_galaxies = (
                seq(galaxies)
                .count(lambda galaxy: galaxy.x == x)
            )
            if num_galaxies == 0:
                empty_columns.append(x)
        for galaxy in galaxies:
            galaxy.x += (
                seq(empty_columns)
                .count(lambda x: x < galaxy.x)
            ) * (self._expansion_factor - 1)

    def transform_2(self, data: str) -> Any:
        ...

    def transform_1(self, data: str) -> Any:
        y = 0
        number = 1
        galaxies: list[Galaxy] = []
        for line in data.splitlines(keepends=False):
            line = line.strip()
            if len(line) > 0:
                x = 0
                try:
                    while True:
                        x = line.index("#", x)
                        galaxies.append(Galaxy(x, y, number))
                        x += 1
                        number += 1
                except ValueError:
                    pass
                if x == 0:
                    y += (self._expansion_factor - 1)
                y += 1
        self._expand_columns(galaxies)

        result = (
            seq(itertools.combinations(galaxies, 2))
            .map(self._compute_distance)
            .sum()
        )
        return result

    def with_expansion_factor(self, expansion_factor: int) -> "TransformerImpl":
        self._expansion_factor = expansion_factor
        return self

    def _compute_distance(self, galaxies: tuple[Galaxy, Galaxy]) -> int:
        a, b = galaxies
        result = abs(a.x - b.x) + abs(a.y - b.y)
        return result


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    answer_2 = sut.with_expansion_factor(1_000_000).transform_1(data)
    print(answer_2)
