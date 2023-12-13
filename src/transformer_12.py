from copy import copy

import numpy as np
import itertools
from enum import Enum, auto
from functools import partial, lru_cache
from pathlib import Path
from typing import Any, Generator

from functional import seq

from transformer import Transformer


class SpringCondition(Enum):
    operational = auto()
    damaged = auto()
    unknown = auto()


class TransformerImpl(Transformer):
    spring_condition_map = {
        ".": SpringCondition.operational,
        "#": SpringCondition.damaged,
        "?": SpringCondition.unknown,
    }

    def transform_2(self, data: str) -> Any:
        conditions: list[list[SpringCondition]] = []
        patterns: list[list[int]] = []
        lines = []
        for condition in data.splitlines(keepends=False):
            condition = condition.strip()
            if len(condition) > 0:
                parts = condition.split()
                conditions.append(
                    ((seq(list(parts[0]))
                      .map(self.spring_condition_map.__getitem__)
                      .to_list()
                      + [SpringCondition.unknown])
                     * 5)[:-1]
                )
                patterns.append(
                    seq(parts[1].split(","))
                    .map(int)
                    .to_list()
                    * 5
                )
                lines.append("?".join([parts[0]]*5))
        total = 0
        for i, (condition, pattern, line) in enumerate(zip(conditions, patterns, lines)):
            line_result = self._solve("." + line + ".", tuple(pattern))
            # line_result = 1 + line_result - len(pattern)
            total += line_result
        return total

    def transform_1(self, data: str) -> Any:
        conditions: list[list[SpringCondition]] = []
        patterns: list[list[int]] = []
        lines: list[str] = []
        for condition in data.splitlines(keepends=False):
            condition = condition.strip()
            if len(condition) > 0:
                parts = condition.split()
                conditions.append(
                    seq(list(parts[0]))
                    .map(self.spring_condition_map.__getitem__)
                    .to_list()
                )
                patterns.append(
                    seq(parts[1].split(","))
                    .map(int)
                    .to_list()
                )
                lines.append(parts[0])
        total = 0
        for i, (condition, pattern, line) in enumerate(zip(conditions, patterns, lines)):
            line_result = self._solve("." + line + ".", tuple(pattern))
            # line_result = 1 + line_result - len(pattern)
            total += line_result
        return total

    @lru_cache()
    def get_search_strings(self, n: int) -> set[str]:
        unmasked = "." + "#" * n + "."
        result = {unmasked}
        for masked in self._all_possibilities(unmasked, 0):
            result.add(masked)
        return result

    def _all_possibilities(self, s: str, index: int) -> Generator[str, None, None]:
        if index == len(s):
            yield s
        if index < len(s):
            for possibility in (s[index], "?"):
                yield from self._all_possibilities(s[:index] + possibility + s[index + 1:], index + 1)

    @lru_cache()
    def _solve(self, text: str, pattern: tuple[int]) -> int | None:
        if len(pattern) == 0:
            if "#" in text:
                return None
            else:
                return 0
        max_contiguous = max(pattern)
        arg_max_contiguous = pattern.index(max_contiguous)
        search_strings = self.get_search_strings(max_contiguous)
        matches = (
            seq(search_strings)
            .map(partial(self._safe_index, text))
            .flatten()
            .filter(lambda x: x is not None)
            .to_set()
        )
        if not matches:
            return None
        result = 0
        for i, match_index in enumerate(matches):
            match_rindex = match_index + max_contiguous + 2
            text_match = text[match_index:match_rindex]

            left_index = 0
            left_rindex = match_index + 1
            left_text = text[left_index:left_rindex]

            left_pattern_index = 0
            left_pattern_rindex = arg_max_contiguous
            left_pattern = pattern[left_pattern_index:left_pattern_rindex]

            right_index = match_rindex - 1
            right_rindex = len(text)
            right_text = text[right_index:right_rindex]

            right_pattern_index = arg_max_contiguous + 1
            right_pattern_rindex = len(pattern)
            right_pattern = pattern[right_pattern_index:right_pattern_rindex]

            pass
            if (
                    (left := self._solve(text[left_index:left_rindex], pattern[left_pattern_index:left_pattern_rindex])) is not None
                and (right := self._solve(text[right_index:right_rindex], pattern[right_pattern_index:right_pattern_rindex])) is not None
            ):

                combinations_for_match = max(left, right, 1)
                result += combinations_for_match

                # print(f"Searching patterns {pattern} in '{text}'")
                # print(f"Found match {i + 1}/{len(matches)} for {max_contiguous} at {match_index}: '{text_match}'")
                # print(f"Now searching...")
                # print(f"Left: {left_text} {left_pattern}: {left}")
                # print(f"Right: {right_text} {right_pattern}: {right}")
                # print(f"Result for section: {result}")
                # print("")

                pass
        return result

    def _safe_index(self, text: str, search_string: str) -> Generator[int | None, None, None]:
        index = 0
        while index < len(text):
            try:
                result = text.index(search_string, index)
                yield result
                index = result + 1
            except ValueError:
                return None

    def print_matches(self, matches: list[list[SpringCondition]]):
        reverse_map = {
            v: k for k, v in self.spring_condition_map.items()
        }
        for match in matches:
            print("".join(
                seq(match)
                .map(reverse_map.__getitem__)
            ))

    def iterate_possibilities(self, conditions: list[SpringCondition]) -> Generator[list[SpringCondition], None, None]:
        if len(conditions) == 0:
            yield ()
        else:
            c = conditions[0]
            if c == SpringCondition.unknown:
                for potential in (SpringCondition.operational, SpringCondition.damaged):
                    for potential_suffix in self.iterate_possibilities(conditions[1:]):
                        yield itertools.chain((potential,), potential_suffix)
            else:
                for potential_suffix in self.iterate_possibilities(conditions[1:]):
                    yield itertools.chain((c,), potential_suffix)

    def matches_contiguous(self, pattern: list[int], condition: list[SpringCondition]) -> bool:
        pattern_position = 0
        need_to_match: int | None = None
        for c in condition:
            match c:
                case SpringCondition.operational:
                    match need_to_match:
                        case None:
                            pass
                        case 0:
                            need_to_match = None
                            pattern_position += 1
                        case _:
                            return False
                case SpringCondition.damaged:
                    if need_to_match is None:
                        if pattern_position >= len(pattern):
                            return False
                        else:
                            need_to_match = pattern[pattern_position]
                    elif need_to_match == 0:
                        return False
                    need_to_match -= 1
        has_completed_all_patterns = (
                (pattern_position == len(pattern))
                or (
                        (pattern_position == len(pattern) - 1)
                        and (need_to_match == 0)
                )
        )
        return has_completed_all_patterns


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    # print(sut.transform_2(data))
