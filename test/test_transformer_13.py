import sys
from pathlib import Path
from unittest import TestCase, skip

import numpy

from transformer_13 import TransformerImpl

numpy.set_printoptions(threshold=sys.maxsize)


class TestTransformer11(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sut = TransformerImpl()

        file_path = Path(__file__)
        data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
        cls.real_data = data_path.read_text()

    def test_transform_1(self):
        self.assertEqual(405, self.sut.transform_1("""#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#"""))

    def test_transform_2(self):
        self.assertEqual(400, self.sut.transform_2("""#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#"""))
