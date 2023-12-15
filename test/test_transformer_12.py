import sys
from pathlib import Path
from unittest import TestCase, skip

import numpy

from transformer_12 import TransformerImpl

numpy.set_printoptions(threshold=sys.maxsize)


class TestTransformer11(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sut = TransformerImpl()

        file_path = Path(__file__)
        data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
        cls.real_data = data_path.read_text()

    def test_transform_1(self):
        self.assertEqual(21, self.sut.transform_1("""???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""))

    def test_solve_1(self):
        self.assertEqual(10,
                         self.sut._solve(".?###????????.",
                                         (3, 2, 1)))

    def test_solve_2(self):
        self.assertEqual(4, self.sut._solve(".????.", (1,)))

    def test_solve_3(self):
        self.assertEqual(1, self.sut._solve(".???.", (1, 1)))

    def test_transform_2(self):
        self.assertEqual(525152, self.sut.transform_2("""???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""))

    def test_transform_2_individual_2(self):
        self.assertEqual(16384, self.sut.transform_2(".??..??...?##. 1,1,3"))

    def test_transform_2_individual(self):
        self.assertEqual(1, self.sut.transform_2("???.### 1,1,3"))
        self.assertEqual(16384, self.sut.transform_2(".??..??...?##. 1,1,3"))
        self.assertEqual(1, self.sut.transform_2("?#?#?#?#?#?#?#? 1,3,1,6"))
        self.assertEqual(16, self.sut.transform_2("????.#...#... 4,1,1"))
        self.assertEqual(2500, self.sut.transform_2("????.######..#####. 1,6,5"))
        self.assertEqual(506250, self.sut.transform_2("?###???????? 3,2,1"))

    def test_get_search_strings(self):
        self.assertEqual(
            set([
                ".##.",
                "?##.",
                ".?#.",
                ".#?.",
                ".##?",
                "??#.",
                "?#?.",
                "?##?",
                ".??.",
                ".?#?",
                ".#??",
                "???.",
                "??#?",
                "?#??",
                ".???",
                "????",
            ]),
            self.sut.get_search_strings(2)
        )
