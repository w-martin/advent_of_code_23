import heapq
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from transformer import Transformer

LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)


class TransformerImpl(Transformer):

    def __init__(self):
        self._arr: np.ndarray | None = None

    def transform_1(self, data: str) -> Any:
        lines = []
        for line in data.splitlines(False):
            line = line.strip()
            if len(line) > 0:
                lines.append(line)

        self._arr = np.array([list(line) for line in lines]).astype(int)
        result = self._solve(1, 3)

        return result

    def _solve(self, min_steps: int, max_steps: int) -> int:
        queue = []
        destination = (self._arr.shape[0] - 1, self._arr.shape[1] - 1)
        loss: int = 0
        location: tuple[int, int] = (0, 0)
        last_trajectory: tuple[int, int] | None = None
        option = loss, location, last_trajectory
        heapq.heappush(queue, option)

        dist = defaultdict(lambda *_: 9223372036854775807)  # idea from Dijkstra's algo

        while len(queue) > 0:
            loss, location, last_trajectory = heapq.heappop(queue)
            if location == destination:
                return loss
            trajectories = self._get_trajectories(last_trajectory)

            for new_trajectory in trajectories:
                new_loss = loss
                for n_steps in range(1, max_steps + 1):
                    new_location = (
                        location[0] + new_trajectory[0] * n_steps,
                        location[1] + new_trajectory[1] * n_steps
                    )
                    if (0 <= new_location[0] < self._arr.shape[0]
                            and 0 <= new_location[1] < self._arr.shape[1]):
                        new_loss += self._arr[new_location]
                        if new_loss < dist[(new_location, new_trajectory)] and n_steps >= min_steps:
                            dist[(new_location, new_trajectory)] = new_loss
                            option = new_loss, new_location, new_trajectory
                            heapq.heappush(queue, option)

    @lru_cache()
    def _get_trajectories(self, trajectory):
        match trajectory:
            case (-1, 0):
                trajectories = (LEFT, RIGHT)
            case (0, 1):
                trajectories = (UP, DOWN)
            case (1, 0):
                trajectories = (RIGHT, LEFT)
            case (0, -1):
                trajectories = (DOWN, UP)
            case _:
                trajectories = (LEFT, UP, RIGHT, DOWN)
        return trajectories

    def transform_2(self, data: str) -> Any:
        lines = []
        for line in data.splitlines(False):
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
        self._arr = np.array([list(line) for line in lines]).astype(int)

        return self._solve(4, 10)


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    answer_2 = sut.transform_2(data)
    print(answer_2)
