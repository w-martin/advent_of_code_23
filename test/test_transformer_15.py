import sys
from pathlib import Path
from unittest import TestCase, skip

import numpy

from transformer_15 import TransformerImpl

numpy.set_printoptions(threshold=sys.maxsize)


class TestTransformer15(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sut = TransformerImpl()

        file_path = Path(__file__)
        data_path = file_path.parents[1].joinpath("data", f"data_{file_path.name[-5:-3]}.txt")
        cls.real_data = data_path.read_text()

    def test_transform_1(self):
        self.assertEqual(1320, self.sut.transform_1("""rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7"""))

    def test_transform_2(self):
        self.assertEqual(145, self.sut.transform_2("""rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7"""))
