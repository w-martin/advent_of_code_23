import re
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from functional import seq

from transformer import Transformer


@dataclass
class Range:
    source: int
    dest: int
    span: int


@dataclass
class Map:
    source: str
    dest: str
    ranges: list[Range] = field(default_factory=list)


class TransformerImpl(Transformer):

    def transform_1(self, data: str) -> Any:
        seeds = []
        maps = []
        for line in data.splitlines(keepends=False):
            if match := re.match(r"seeds:([ \d]*)", line):
                seeds += seq(match.group(1).split()).map(int).to_list()
            elif match := re.match(r"([a-z]+)-to-([a-z]+) map:", line):
                maps.append(Map(source=match.group(1), dest=match.group(2)))
            elif match := re.match(r"([ \d]+)", line):
                numbers = seq(match.group(1).split()).map(int).to_list()
                source = numbers[1]
                dest = numbers[0]
                maps[-1].ranges.append(Range(source=source, dest=dest, span=numbers[2]))
        starting_locations = seq(seeds).map(partial(self._get_location, maps, "seed"))
        return min(starting_locations)

    def _get_location(self, maps, source, value):
        while source != "location":
            m = self._get_map(maps, source)
            source = m.dest
            r = self._get_range(m.ranges, value)
            if r is None:
                value = value
            else:
                value = value - r.source + r.dest
        return value

    def _get_map_range(self, maps, source, value):
        m = self._get_map(maps, source)
        r = self._get_range(m.ranges, value)
        return m, r

    def _fill_missing_ranges(self, maps):
        max_value = 9223372036854775807
        for m in maps:
            value = 0
            ranges = seq(m.ranges).sorted(key=lambda r: r.source).to_list()
            new_ranges = []
            for r in ranges:
                if r.source > value:
                    new_ranges.append(Range(source=value, dest=value, span=r.source - value))
                new_ranges.append(r)
                value = r.source + r.span
            new_ranges.append(Range(source=value, dest=value, span=max_value - value))
            m.ranges = seq(m.ranges + new_ranges).sorted(key=lambda r: r.source).to_list()

    def _get_locations(self, maps, source, value, span):
        if span < 0:
            return set()
        if source == "location":
            return {value}

        m = self._get_map(maps, source)
        options = set()
        for r in m.ranges:
            if r.source <= value:
                if value < r.source + r.span:
                    # in current range
                    new_value = value - r.source + r.dest
                    new_span = span
                    options |= (self._get_locations(maps, m.dest, new_value, new_span))
            elif value + span >= r.source:
                # in next range
                new_span = span - (r.source - value)
                new_value = r.dest
                options |= (self._get_locations(maps, m.dest, new_value, new_span))
        return options

    def transform_2(self, data: str) -> Any:
        seeds = []
        maps = []
        for line in data.splitlines(keepends=False):
            if match := re.match(r"seeds:([ \d]*)", line):
                seeds += seq(match.group(1).split()).map(int).to_list()
            elif match := re.match(r"([a-z]+)-to-([a-z]+) map:", line):
                maps.append(Map(source=match.group(1), dest=match.group(2)))
            elif match := re.match(r"([ \d]+)", line):
                numbers = seq(match.group(1).split()).map(int).to_list()
                source = numbers[1]
                dest = numbers[0]
                maps[-1].ranges.append(Range(source=source, dest=dest, span=numbers[2]))
        self._fill_missing_ranges(maps)
        location = -1
        while True:
            location += 1
            seed = self._backtrack(maps, "location", location)
            if seed is not None and self._in_initial_seeds(seeds, seed):
                return location
            if location % 10_000_000 == 0:
                print(f"Part 2 not found at {location}")

    def _backtrack(self, maps, source, value):
        if source == "seed":
            return value
        m = self._get_map_reversed(maps, source)
        r = self._get_range_reversed(m.ranges, value)
        if r is None:
            return None
        else:
            return self._backtrack(maps, m.source, value - r.dest + r.source)

    def _in_initial_seeds(self, seeds, value):
        for i in range(int(len(seeds) / 2)):
            index = i * 2
            start = seeds[index]
            span = seeds[1 + index]
            if start <= value < start + span:
                return True
        return False

    def _get_map(self, maps, source):
        for m in maps:
            if m.source == source:
                return m

    def _get_map_reversed(self, maps, source):
        for m in maps:
            if m.dest == source:
                return m

    def _get_range(self, ranges, value):
        for r in ranges:
            start = r.source
            stop = r.source + r.span
            if start <= value < stop:
                return r
        return None

    def _get_range_reversed(self, ranges, value):
        for r in ranges:
            start = r.dest
            stop = r.dest + r.span
            if start <= value < stop:
                return r
        return None


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    answer_2 = sut.transform_2(data)
    assert answer_2 < 216635734
    print(answer_2)
