import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from functional import seq

from transformer import Transformer


@dataclass
class Box:
    number: int
    lenses: dict[str, int] = field(default_factory=OrderedDict)


class TransformerImpl(Transformer):

    def transform_2(self, data: str) -> Any:
        boxes = [Box(i) for i in range(256)]
        for sequence in data.strip().split(","):
            groups = re.match(r"(\w+)([=-])(\d*)", sequence).groups()
            label  = groups[0]
            box_index = self._hash(label)
            box = boxes[box_index]
            match groups[1]:
                case  "=":
                    box.lenses[label] = int(groups[2])
                case "-":
                    try:
                        del box.lenses[label]
                    except KeyError:
                        pass
        result = (
            seq(boxes)
            .map(self._get_focusing_power)
            .sum()
        )
        return result

    def _get_focusing_power(self, box: Box) -> int:
        result = 0
        for i, focal_length in enumerate(box.lenses.values()):
            result += (i + 1) * focal_length
        result *= box.number + 1
        return result

    def transform_1(self, data: str) -> Any:
        result = (
            seq(data.strip().split(","))
            .map(self._hash)
            .sum()
        )
        return result

    def _hash(self, data: str) -> int:
        value = 0
        for c in data:
            value += ord(c)
            value *= 17
            value %= 256
        return value


if __name__ == "__main__":
    file_path = Path(__file__)
    data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
    data = data_path.read_text()
    sut = TransformerImpl()
    print(sut.transform_1(data))
    print(sut.transform_2(data))
