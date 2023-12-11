import sys
from pathlib import Path
from unittest import TestCase, skip

import numpy

from transformer_10 import TransformerImpl

numpy.set_printoptions(threshold=sys.maxsize)


class TestTransformer10(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sut = TransformerImpl()

        file_path = Path(__file__)
        data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
        cls.real_data = data_path.read_text()

    def test_transform_1(self):
        self.assertEqual(4, self.sut.transform_1(""".....
.S-7.
.|.|.
.L-J.
....."""))
        self.assertEqual(8, self.sut.transform_1("""..F7.
.FJ|.
SJ.L7
|F--J
LJ..."""))

    def test_transform_2(self):
        self.assertEqual(1, self.sut.transform_2(""".....
.S-7.
.|.|.
.L-J.
....."""))
        self.assertEqual(4, self.sut.transform_2("""...........
.S-------7.
.|F-----7|.
.||.....||.
.||.....||.
.|L-7.F-J|.
.|..|.|..|.
.L--J.L--J.
..........."""))
        self.assertEqual(4, self.sut.transform_2("""...........
.S------7.
.|F----7|.
.||....||.
.||....||.
.|L-7F-J|.
.|..||..|.
.L--JL--J.
.........."""))
        self.assertEqual(8, self.sut.transform_2(""".F----7F7F7F7F-7....
.|F--7||||||||FJ....
.||.FJ||||||||L7....
FJL7L7LJLJ||LJ.L-7..
L--J.L7...LJS7F-7L7.
....F-J..F7FJ|L7L7L7
....L7.F7||L7|.L7L7|
.....|FJLJ|FJ|F7|.LJ
....FJL-7.||.||||...
....L---J.LJ.LJLJ..."""))
        self.assertEqual(10, self.sut.transform_2("""FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L"""))
