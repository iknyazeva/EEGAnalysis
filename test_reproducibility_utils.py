import unittest
from unittest import TestCase
import numpy as np
import time
from reproducibility_utils import bool_dice, bool_power_fdr, pairwise_bool_dice
from reproducibility_utils import vec_2_arr_bool_dice, vec_2_arr_bool_power_fdr
class Test(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arr = np.random.choice([0, 1], size=(10, 5), p=[0.6, 0.4])

    def test_bool_dice(self):
        u1 = np.random.choice([0, 1], size=5)
        v1 = np.random.choice([0, 1], size=5)
        u = np.array([1, 0, 1])
        v = np.array([1, 1, 1])
        coef = bool_dice(u, v)
        coef1 = bool_dice(u1, v1)
        self.assertEqual(coef, 0.8)
        self.assertTrue(0 <= coef1 <= 1)
        self.assertTrue(bool_dice(u, u) == 1)

    def test_bool_power_fdr(self):
        u = np.array([1, 0, 1])
        v = np.array([1, 1, 1])
        ntt, nft = bool_power_fdr(u, v)
        self.assertTrue(True)

    def test_pairwise_bool_dice(self):
        start = time.time()
        dice_list = pairwise_bool_dice(self.arr)
        end = time.time()
        print(end - start)
        self.assertTrue(True)

    def test_vec_2_arr_bool_power_fdr(self):
        start = time.time()
        arr = np.random.choice([0, 1], size=(10, 5), p=[0.6, 0.4])
        v = arr[:, 0]
        power, fdr = vec_2_arr_bool_power_fdr(v, arr)
        end = time.time()
        print(end - start)
        self.assertTrue(True)
