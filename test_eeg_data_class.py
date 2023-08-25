from unittest import TestCase
from eeg_data_class import Electrodes, PairsElectrodes
from itertools import combinations
class TestPairsElectrodes(TestCase):
    def test_electrode_pairs(self):

        els = map(lambda x: x.name, Electrodes)
        res = list(combinations(els, 2))
        pairs = PairsElectrodes(Electrodes)
        self.assertEqual(sum([el_pair in pairs.nearest for el_pair in pairs.electrode_pairs]),47)
        test_pair1 = ('Fp1', 'Fp2')
        test_pair2 = ('Fp2', 'Fp1')
        test_pair3 = ('Fp1', 'Fp12')
        self.assertTrue(pairs._eq_pair(test_pair1, test_pair1))
        self.assertTrue(pairs._eq_pair(test_pair1, test_pair2))
        self.assertFalse(pairs._eq_pair(test_pair1, test_pair3))
        self.assertTrue(pairs._pair_in_list(test_pair1, res))
        self.assertEqual(res, pairs)
        self.assertNotEquals(res[:3], pairs)
        self.fail()
