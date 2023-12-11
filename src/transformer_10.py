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
class Pipe:
    north: bool = False
    south: bool = False
    east: bool = False
    west: bool = False

    start: bool = False
    min_distance: int | None = None
    visited: bool = False
    enclosed: bool = False
    row: int = -1
    column: int = -1


class Direction(Enum):
    north = auto()
    south = auto()
    east = auto()
    west = auto()


@dataclass
class Option:
    position: tuple[int, int]
    pipe: Pipe
    length: int


class PipeFactory:

    def create(self, pipe_character: str) -> Pipe:
        match pipe_character:
            case "|":
                return Pipe(north=True, south=True)
            case "-":
                return Pipe(east=True, west=True)
            case "L":
                return Pipe(north=True, east=True)
            case "J":
                return Pipe(north=True, west=True)
            case "7":
                return Pipe(south=True, west=True)
            case "F":
                return Pipe(south=True, east=True)
            case ".":
                return Pipe()
            case "S":
                return Pipe(start=True)


class TransformerImpl(Transformer):

    def _fix_start(self, grid: list[list[Pipe]]) -> None:
        start: tuple[int, int] | None = None
        starting_pipe : Pipe|None = None
        for row in range(len(grid)):
            if start:
                break
            for col in range(len(grid[row])):
                pipe = grid[row][col]
                if pipe.start:
                    start = (row, col)
                    starting_pipe = pipe
                    break
        for row, col in (
            (start[0]  -1, start[1]),
            (start[0] + 1, start[1]),
            (start[0], start[1] - 1),
            (start[0], start[1] + 1)
        ):
            if 0 <= row <= len(grid) and 0 <= col <= len(grid[0]):
                pipe = grid[row][col]
                if row < start[0] and pipe.south:
                    starting_pipe.north = True
                elif row > start[0] and pipe.north:
                    starting_pipe.south = True
                elif col < start[1] and pipe.east:
                    starting_pipe.west = True
                elif col > start[1] and pipe.west:
                    starting_pipe.east = True
        return start

    def transform_1(self, data: str) -> Any:
        grid: list[list[Pipe]] = []
        pipe_factory = PipeFactory()
        for line in data.splitlines(keepends=False):
            line = line.strip()
            if len(line) > 0:
                grid.append(
                    seq(list(line))
                    .map(pipe_factory.create)
                    .to_list()
                )
        start: tuple[int, int] = self._fix_start(grid)

        result = self._bfs(grid, start)

        return result

    def _get_options(self, grid: list[list[Pipe]], option: Option) -> list[Option]:
        options = []
        if option.pipe.north:
            row = option.position[0] - 1
            if row >= 0:
                next_position = (row, option.position[1])
                next_pipe = grid[next_position[0]][next_position[1]]
                if not next_pipe.visited:
                    options.append(
                        Option(
                            position=next_position,
                            pipe=next_pipe,
                            length=option.length + 1
                        )
                    )
        if option.pipe.east:
            column = option.position[1] + 1
            if column < len(grid[0]):
                next_position = (option.position[0], column)
                next_pipe = grid[next_position[0]][next_position[1]]
                if not next_pipe.visited:
                    options.append(
                        Option(
                            position=next_position,
                            pipe=next_pipe,
                            length=option.length + 1
                        )
                    )
        if option.pipe.south:
            row = option.position[0] + 1
            if row < len(grid):
                next_position = (row, option.position[1])
                next_pipe = grid[next_position[0]][next_position[1]]
                if not next_pipe.visited:
                    options.append(
                        Option(
                            position=next_position,
                            pipe=next_pipe,
                            length=option.length + 1
                        )
                    )
        if option.pipe.west:
            column = option.position[1] - 1
            if column >= 0:
                next_position = (option.position[0], column)
                next_pipe = grid[next_position[0]][next_position[1]]
                if not next_pipe.visited:
                    options.append(
                        Option(
                            position=next_position,
                            pipe=next_pipe,
                            length=option.length + 1
                        )
                    )
        return options


    def _bfs(self, grid: list[list[Pipe]], current_position: tuple[int, int]):
        q = deque()

        current_pipe = grid[current_position[0]][current_position[1]]
        q.append(
            Option(
                position=current_position,
                pipe=current_pipe,
                length=0
            ))
        max_distance = 0
        while len(q) > 0:
            option = q.popleft()
            if not option.pipe.visited:
                option.pipe.visited = True
                new_min_distance = min(option.length, option.pipe.min_distance or option.length)
                option.pipe.min_distance = new_min_distance
                max_distance = max(max_distance, option.pipe.min_distance)
                new_options = self._get_options(grid, option)
                q.extend(new_options)
                pass
        # self._print_grid(grid)
        return max_distance

    def _print_grid(self, grid: list[list[Pipe]]):
        for row in grid:
            print("".join(str(column.min_distance if column.min_distance is not None else ".") for column in row))
        print("")

    def _print_enclosed_grid(self, grid: list[list[Pipe]]):
        for row in grid:
            print("".join(str(column.min_distance if column.min_distance is not None else ("I" if column.enclosed else "O")) for column in row))
        print("")

    def _print_pipe(self, pipe: Pipe) -> str:
        if not pipe.visited:
            return "."
        if pipe.north and pipe.south:
            return "|"
        elif pipe.east and pipe.west:
            return "-"
        elif pipe.north and pipe.east:
            return "L"
        elif pipe.north and pipe.west:
            return "J"
        elif pipe.south and pipe.west:
            return "7"
        elif pipe.south and pipe.east:
            return "F"
        else:
            return "."

    def _reprint_line(self, grid_line: list[Pipe]) -> str:
        return "".join(self._print_pipe(p) for p in grid_line)

    def _reprint(self, grid: list[list[Pipe]]) -> str:
        return "\n".join(self._reprint_line(line) for line in grid)

    def transform_2(self, data: str) -> Any:
        grid: list[list[Pipe]] = []
        pipe_factory = PipeFactory()
        for line in data.splitlines(keepends=False):
            line = line.strip()
            if len(line) > 0:
                grid.append(
                    seq(list(line))
                    .map(pipe_factory.create)
                    .to_list()
                )
        start  = self._fix_start(grid)
        self._get_vertices(grid, start)

        data = self._reprint(grid)
        enclosed_area = 0
        for i, line in enumerate(data.splitlines(keepends=False)):
            line = line.strip()
            if len(line) > 0:
                enclosed = False
                nested_part = ""
                for j, char in enumerate(line):
                    # if not enclosed and char in ("|", "J", "7"):
                    #     enclosed = True
                    # elif enclosed and char in ("|", "L", "J", "7", "F"):
                    #     enclosed = not enclosed
                    # elif char == "S":
                    #     if (
                    #             ((nj := j - 1) >= 0 and line[nj] in ("-", "L", "F"))
                    #         and ((nj := j + 1) < len(line) and line[nj] in ("-", "J", "7"))
                    #     ) :
                    #         pass
                    #     else:
                    #         enclosed = not enclosed
                    pass
                    match char:
                        case "|":
                            enclosed = not enclosed
                        case char if char in ("L", "F"):
                            enclosed = not enclosed
                            nested_part = char
                        case "7":
                            match nested_part:
                                case "F":
                                    enclosed = not enclosed
                                case "L":
                                    enclosed = enclosed
                        case "J":
                            match nested_part:
                                case "F":
                                    enclosed = enclosed
                                case "L":
                                    enclosed = not enclosed
                        case ".":
                            if enclosed:
                                enclosed_area += 1


        # start: tuple[int, int] = self._fix_start(grid)
        # max_distance = self._bfs(grid, start)
        # for pipe in seq(grid).flatten():
        #     pipe.visited = False
        # vertices = self._get_vertices(grid, start)
        # enclosed_area = self._shoelace(vertices)
        # num_pipes = (seq(grid).flatten().filter(lambda pipe: pipe.visited).len())
        # result = enclosed_area - (max_distance - 1)

        return enclosed_area

    def _shoelace(self, vertices: list[tuple[int, int]]) -> int:
        area = 0
        for pair in zip(vertices, vertices[1:] + vertices[:1]):
            first = pair[0]
            second = list(pair[1])
            # if second[0] < first[0]:
            #     second[0] += 1
            # else:
            #     second[0] -= 1
            # if second[1] < first[1]:
            #     second[1] += 1
            # else:
            #     second[1] -= 1
            section = int(np.linalg.det([pair[0], second]))
            # section = (second[0] - first[0]) * (second[1] - first[1])
            area += section
        # result = abs(area)
        result = abs(area) // 2
        return result

    def _get_vertices(self, grid: list[list[Pipe]], start: tuple[int, int]):
        current_vertex = start
        vertices: list[tuple[int, int]] = []
        direction = None

        # change direction
        vertices.append(current_vertex)
        current_pipe: Pipe = grid[current_vertex[0]][current_vertex[1]]
        options = self._get_options(grid, Option(current_vertex, current_pipe, 0))
        next_option = options[0]
        if next_option.position[0] < current_vertex[0]:
            direction = Direction.north
        elif next_option.position[0] > current_vertex[0]:
            direction = Direction.south
        elif next_option.position[1] < current_vertex[1]:
            direction = Direction.west
        elif next_option.position[1] > current_vertex[1]:
            direction = Direction.east
        current_vertex = next_option.position

        while True:
            current_pipe: Pipe = grid[current_vertex[0]][current_vertex[1]]
            current_pipe.visited = True
            # keep going
            if direction in (None, Direction.north) and current_pipe.north:
                current_vertex = (current_vertex[0] - 1, current_vertex[1])
            elif direction in (None, Direction.east) and current_pipe.east:
                current_vertex = (current_vertex[0], current_vertex[1] + 1)
            elif direction in (None, Direction.south) and current_pipe.south:
                current_vertex = (current_vertex[0] + 1, current_vertex[1])
            elif direction in (None, Direction.west) and current_pipe.west:
                current_vertex = (current_vertex[0], current_vertex[1] - 1)
            else:
                # change direction
                options = self._get_options(grid, Option(current_vertex, current_pipe, 0))
                if len(options) == 0:
                    break
                next_option = options[0]
                if next_option.position[0] < current_vertex[0]:
                    direction = Direction.north
                elif next_option.position[0] > current_vertex[0]:
                    direction = Direction.south
                elif next_option.position[1] < current_vertex[1]:
                    direction = Direction.west
                elif next_option.position[1] > current_vertex[1]:
                    direction = Direction.east
                # if current_vertex[0] != vertices[-1][0] and current_vertex[1] != vertices[-1][1]:
                vertices.append(current_vertex)
                current_vertex = next_option.position

        return vertices


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    answer_2 = sut.transform_2(data)
    print(answer_2)
