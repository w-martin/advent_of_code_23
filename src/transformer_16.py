from collections import deque
from pathlib import Path
from typing import Any, Generator

import numpy as np
from functional import seq

from transformer import Transformer

LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)


class TransformerImpl(Transformer):

    def __init__(self):
        self._data: np.ndarray | None = None

    def transform_2(self, data: str) -> Any:
        lines = []
        for line in data.splitlines(False):
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
        self._data = np.array([list(line) for line in lines])

        beams = (
            # top down
                [((-1, column), (1, 0)) for column in range(self._data.shape[1])]
                # bottom up
                + [((self._data.shape[0] - 1, column), (-1, 0)) for column in range(self._data.shape[1])]
                # left right
                + [((row, -1), (0, 1)) for row in range(self._data.shape[0])]
                # right left
                + [((row, self._data.shape[1] - 1), (0, -1)) for row in range(self._data.shape[0])]
        )

        result = (
            seq(beams)
            .map(self._energise)
            .map(np.sum)
            .max()
        )

        return result

    def transform_1(self, data: str) -> Any:
        lines = []
        for line in data.splitlines(False):
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
        self._data = np.array([list(line) for line in lines])

        beam = (0, -1), (0, 1)
        energised = self._energise(beam)
        result = np.sum(energised)

        return result

    def _energise(self, beam: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
        energised = np.zeros_like(self._data, dtype=bool)
        seen = set()
        beam_queue = deque((beam,))
        while len(beam_queue) > 0:
            beam = beam_queue.popleft()
            if beam in seen:
                continue
            else:
                seen.add(beam)

            new_beams = self._move(beam, energised)
            beam_queue.extend(new_beams)
            pass

        return energised

    def _move(self, beam, energised) -> Generator[tuple[tuple[int, int], tuple[int, int]], None, None]:
        # move
        position, trajectory = beam
        position = (position[0] + trajectory[0], position[1] + trajectory[1])
        is_in_bounds = (0 <= position[0] < self._data.shape[0] and 0 <= position[1] < self._data.shape[1])
        if not is_in_bounds:
            return None

        beams = []
        energised[*position] = True
        # adjust trajectory
        match self._data[*position]:
            case ".":
                pass
            case "|":
                if trajectory[1] != 0:
                    trajectory = DOWN
                    yield position, UP
            case "-":
                if trajectory[0] != 0:
                    trajectory = RIGHT
                    yield position, LEFT
            case "\\":
                match trajectory:
                    case 0, 1:
                        trajectory = DOWN
                    case 0, -1:
                        trajectory = UP
                    case 1, 0:
                        trajectory = RIGHT
                    case -1, 0:
                        trajectory = LEFT
            case "/":
                match trajectory:
                    case 0, 1:
                        trajectory = UP
                    case 0, -1:
                        trajectory = DOWN
                    case 1, 0:
                        trajectory = LEFT
                    case -1, 0:
                        trajectory = RIGHT
        yield position, trajectory


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    answer_2 = sut.transform_2(data)
    assert 7901 < answer_2
    print(answer_2)
