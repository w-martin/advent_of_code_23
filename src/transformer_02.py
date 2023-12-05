import re
from pathlib import Path
from typing import Any

from transformer import Transformer
from functional import seq

class TransformerImpl(Transformer):
    def _compute_line_1(self, line: str):
        limits = {
            "red": 12,
            "green": 13,
            "blue": 14
        }

        game_number = int(re.match(r"Game (?P<group>\d+):", line).group("group"))

        game_was_possible = True
        for match in re.finditer(r"(?P<group>\d+) (?P<color>\w+)", line):
            color = match.group("color")
            number = int(match.group("group"))
            if number > limits[color]:
                game_was_possible = False
                break

        if game_was_possible:
            return game_number
        else:
            return 0
        
    def _compute_line_2(self, line: str):
        limits = {
            "red": 0,
            "green": 0,
            "blue": 0
        }

        for match in re.finditer(r"(?P<group>\d+) (?P<color>\w+)", line):
            color = match.group("color")
            number = int(match.group("group"))
            if number > limits[color]:
                limits[color] = number

        return seq(limits.values()).product()

    def transform_1(self, data: str) -> Any:
        return (
            seq(data.splitlines(keepends=False))
            .map(self._compute_line_1)
            .sum()
        )


    def transform_2(self, data: str) -> Any:
        return (
            seq(data.splitlines(keepends=False))
            .map(self._compute_line_2)
            .sum()
        )


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_2(data))