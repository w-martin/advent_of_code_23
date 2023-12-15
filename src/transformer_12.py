import itertools
from enum import Enum, auto
from functools import partial, lru_cache
from pathlib import Path
from typing import Any, Generator

from functional import seq

from transformer import Transformer


class TransformerImpl(Transformer):

    def transform_2(self, data: str) -> Any:
        patterns: list[list[int]] = []
        lines = []
        for condition in data.splitlines(keepends=False):
            condition = condition.strip()
            if len(condition) > 0:
                parts = condition.split()
                patterns.append(
                    seq(parts[1].split(","))
                    .map(int)
                    .to_list()
                    * 5
                )
                lines.append("?".join([parts[0]] * 5))
        total = 0
        for i, (pattern, line) in enumerate(zip(patterns, lines)):
            line_result = self._solve("." + line + ".", tuple(pattern))
            # line_result = 1 + line_result - len(pattern)
            total += line_result
            print(f"Line {i + 1}/{len(lines)}: {line} -> {line_result}")
        return total

    def transform_1(self, data: str) -> Any:
        patterns: list[list[int]] = []
        lines: list[str] = []
        for condition in data.splitlines(keepends=False):
            condition = condition.strip()
            if len(condition) > 0:
                parts = condition.split()
                patterns.append(
                    seq(parts[1].split(","))
                    .map(int)
                    .to_list()
                )
                lines.append(parts[0])
        total = 0
        for i, (pattern, line) in enumerate(zip(patterns, lines)):
            line_result = self._solve("." + line + ".", tuple(pattern))
            # line_result = 1 + line_result - len(pattern)
            total += line_result
            print(f"Line {i + 1}/{len(lines)}: {line} -> {line_result}")
        return total

    @lru_cache(maxsize=None)
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

    @lru_cache(maxsize=None)
    def _solve(self, text: str, pattern: tuple[int]) -> int | None:
        if len(pattern) == 0:
            if "#" in text:
                return None
            else:
                return 1
        max_contiguous = max(pattern)
        arg_max_contiguous = pattern.index(max_contiguous)
        num_max_contiguous = pattern.count(max_contiguous)
        matches = sorted(self._get_matches(max_contiguous, text))
        num_matches = len(matches)
        if not matches:
            return None
        if num_matches < num_max_contiguous:
            return None
        spare_matches = (num_matches - num_max_contiguous)
        result = 0
        for i in range(1 + spare_matches):
            match_index = matches[i]
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

            if (
                    (left := self._solve(left_text, left_pattern)) is not None
                    and (right := self._solve(right_text, right_pattern)) is not None
            ):
                result += left * right

        return result

    @lru_cache(maxsize=None)
    def _get_matches(self, max_contiguous, text):
        search_strings = self.get_search_strings(max_contiguous)
        matches = (
            seq(search_strings)
            .map(partial(self._safe_index, text))
            .flatten()
            .to_set()
        )
        return list(matches)

    @lru_cache(maxsize=None)
    def _safe_index(self, text: str, search_string: str) -> list[int]:
        index = 0
        results = []
        while index < len(text):
            try:
                result = text.index(search_string, index)
                results.append(result)
                index = result + 1
            except ValueError:
                break
        return results


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    print(sut.transform_2(data))
