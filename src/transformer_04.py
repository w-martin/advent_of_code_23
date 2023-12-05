import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from functional import seq

from transformer import Transformer


class TransformerImpl(Transformer):

    def _compute_line_1(self, line: str) -> int:
        num_matches = self._get_num_matches(line)
        return int(math.pow(2, num_matches - 1))

    def _get_num_matches(self, line: str) -> int:
        separator_index = line.index("|")
        winning_numbers = (
            seq(re.finditer(r"(\d+)", line[line.index(":"):separator_index]))
            .map(lambda m: int(m.group(1))
                 )
            .to_set()
        )
        your_numbers = (
            seq(re.finditer(r"(\d+)", line[separator_index:]))
            .map(lambda m: int(m.group(1))
                 )
            .to_set()
        )
        num_matches = len(winning_numbers.intersection(your_numbers))
        return num_matches

    def transform_1(self, data: str) -> Any:
        return (
            seq(data.splitlines(keepends=False))
            .map(self._compute_line_1)
            .sum()
        )

    def transform_2(self, data: str) -> Any:
        num_match_list = (
            seq(data.splitlines(keepends=False))
            .map(self._get_num_matches)
            .to_list()
        )
        card_match_map = {
            i + 1: 1
            for i in range(len(num_match_list))
        }
        for i, num_matches in enumerate(num_match_list):
            card_number = i + 1
            for j in range(card_number + 1, card_number + num_matches + 1):
                copy_number = j
                # print(f"card_number: {card_number} wins {card_match_map[card_number]} instances of copy_number: {copy_number}: {card_match_map[copy_number]} -> {card_match_map[copy_number] + card_match_map[card_number]}")
                card_match_map[copy_number] += card_match_map[card_number]



        result = (
            seq(iter(card_match_map.values()))
            .sum()
        )
        return result


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    print(sut.transform_2(data))
