import sys
from pathlib import Path
from unittest import TestCase, skip

import numpy
from transformer_17 import TransformerImpl

numpy.set_printoptions(threshold=sys.maxsize)


class TestTransformer17(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sut = TransformerImpl()

        file_path = Path(__file__)
        data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
        cls.real_data = data_path.read_text()
        cls.data = """2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533"""

    def test_transform_1(self):
        self.assertEqual(102, self.sut.transform_1(self.data))

    def test_transform_1_real_data(self):
        self.assertLess(904, self.sut.transform_1(self.real_data))

    def test_transform_2_1(self):
        self.assertEqual(94, self.sut.transform_2(self.data))

    def test_transform_2_2(self):
        self.assertEqual(71, self.sut.transform_2("""111111111111
999999999991
999999999991
999999999991
999999999991"""))
