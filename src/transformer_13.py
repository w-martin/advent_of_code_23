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
        patterns: list[list[str]] = []
        current_pattern = []
        for line in data.splitlines(keepends=False):
            line = line.strip()
            if len(line) > 0:
                current_pattern.append(line)
            else:
                patterns.append(current_pattern)
                current_pattern = []
        if len(current_pattern) > 0:
            patterns.append(current_pattern)

        result = (
            seq(patterns)
            .map(self._solve_2)
            .sum()
        )
        return result

    def transform_1(self, data: str) -> Any:
        patterns: list[list[str]] = []
        current_pattern = []
        for line in data.splitlines(keepends=False):
            line = line.strip()
            if len(line) > 0:
                current_pattern.append(line)
            else:
                patterns.append(current_pattern)
                current_pattern = []
        if len(current_pattern) > 0:
            patterns.append(current_pattern)

        result = (
            seq(patterns)
            .map(self._solve_1)
            .sum()
        )
        return result

    def _solve_1(self, pattern: list[str]) -> int:
        arr = np.vstack([seq(pattern).map(list).to_list()])
        arr = (arr == '#').astype(int)
        result = 0
        # column wise
        for i in range(1, len(pattern[0])):
            rindex = min(arr.shape[1], 2 * i)
            right_diff = rindex - i
            left_diff = i
            diff = min(left_diff, right_diff)
            rindex = i + diff
            index = i - diff
            a = arr[:, index:i][:, ::-1]
            b = arr[:, i:rindex]
            is_mirror = (a == b).all().all()
            if is_mirror:
                result += i
        # row wise
        for i in range(1, arr.shape[0]):
            rindex = min(arr.shape[0], 2 * i)
            right_diff = rindex - i
            left_diff = i
            diff = min(left_diff, right_diff)
            rindex = i + diff
            index = i - diff
            a = arr[index:i, :][::-1, :]
            b = arr[i:rindex, :]
            is_mirror = (a == b).all().all()
            if is_mirror:
                result += (i * 100)
        return result

    def _solve_2(self, pattern: list[str]) -> int:
        arr = np.vstack([seq(pattern).map(list).to_list()])
        arr = (arr == '#').astype(int)
        result = 0
        # column wise
        for i in range(1, len(pattern[0])):
            rindex = min(arr.shape[1], 2 * i)
            right_diff = rindex - i
            left_diff = i
            diff = min(left_diff, right_diff)
            rindex = i + diff
            index = i - diff
            a = arr[:, index:i][:, ::-1]
            b = arr[:, i:rindex]
            is_mirror = (a!=b).sum() == 1
            if is_mirror:
                result += i
        # row wise
        for i in range(1, arr.shape[0]):
            rindex = min(arr.shape[0], 2 * i)
            right_diff = rindex - i
            left_diff = i
            diff = min(left_diff, right_diff)
            rindex = i + diff
            index = i - diff
            a = arr[index:i, :][::-1, :]
            b = arr[i:rindex, :]
            is_mirror = (a!=b).sum() == 1
            if is_mirror:
                result += (i * 100)
        return result


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    print(sut.transform_2(data))
