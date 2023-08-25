from unittest import TestCase
from tables import EEGSynchronizationTable, PairedNonParametric, DataTable
from eeg_data_class import PairsElectrodes, Bands, Electrodes
import numpy as np


class TestEEGSynchronizationTable(TestCase):
    def test_get_2dim_data(self):
        data = np.random.randn(20, 171, 7)
        subj_list = [f'sub_{i}' for i in range(20)]
        pairs = PairsElectrodes(Electrodes)
        stable = EEGSynchronizationTable(data, subj_list, pairs.electrode_pairs, Bands)
        el_pair = ('Fp1', 'Fp2')
        df = stable.get_2dim_data(band=None, el_pairs=el_pair)
        self.assertEqual(df.shape, (20, 7))
        df = stable.get_2dim_data(band='delta', el_pairs=None)
        self.assertTrue(True)


class TestPairedNonParametric(TestCase):
    def test_get_permutational_sample(self):
        arr1 = np.random.randn(20, 3, 4,)
        arr2 = 10+np.random.randn(20, 3, 4,)
        data = np.random.randn(20, 171, 7)
        subj_list = [f'sub_{i}' for i in range(20)]
        table1 = DataTable(arr1, subj_list)
        table2 = DataTable(arr2, subj_list)
        pt = PairedNonParametric(table1, table2)
        emp = pt.get_emp_diff()
        psample = pt.get_perm_diff()

        self.fail()
