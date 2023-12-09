from pathlib import Path
from typing import Any

import numpy as np
from functional import seq

from transformer import Transformer


class TransformerImpl(Transformer):

    def _predict_next(self, x: np.ndarray) -> int:
        result = (
            seq(range(len(x)))
            .map(lambda n: np.diff(x, n)[-1])
            .sum()
        )
        return result

    def _predict_next_backwards(self, x: np.ndarray) -> int:
        x = x[::-1]
        result = (
            seq(range(len(x)))
            .map(lambda n: np.diff(x, n)[-1])
            .sum()
        )
        return result

    def transform_1(self, data: str) -> Any:
        sequences = self._read_sequences(data)
        result = (
            seq(sequences)
            .map(np.array)
            .map(self._predict_next)
            .sum()
        )

        return result

    def _read_sequences(self, data):
        sequences = []
        for line in data.splitlines(keepends=False):
            line = line.strip()
            if len(line) > 0:
                sequences.append(
                    seq(line.split())
                    .map(int)
                    .to_list()
                )
        return sequences

    def transform_2(self, data: str) -> Any:
        sequences = self._read_sequences(data)
        result = (
            seq(sequences)
            .map(np.array)
            .map(self._predict_next_backwards)
            .sum()
        )
        return result


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    answer_2 = sut.transform_2(data)
    print(answer_2)
