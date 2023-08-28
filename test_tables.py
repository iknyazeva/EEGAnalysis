from unittest import TestCase
from tables import EEGSynchronizationTable, PairedNonParametric, DataTable
from eeg_data_class import PairsElectrodes, Bands, Electrodes
import numpy as np


class TestDataTable(TestCase):

    def setUp(self):
        arr1 = np.random.randn(20, 3, 4)
        arr2 = 10 + np.random.randn(20, 3, 4)
        subj_list = [f'sub_{i}' for i in range(20)]
        self.table1 = DataTable(arr1, subj_list)
        self.table2 = DataTable(arr1, subj_list)
        self.table3 = DataTable(arr2, subj_list)

    def test_magic(self):
        self.assertEqual(self.table1, self.table2)
        diff = self.table3 - self.table2
        self.assertIsInstance(diff, DataTable)
        self.assertTrue(True)

    def test_compute_stat(self):
        self.assertRaises(NotImplementedError, self.table1.compute_stat, 'efff_size')
        eff_sizes = self.table1.compute_stat(type_stat='eff_size')
        self.assertEqual(eff_sizes.shape, self.table1.data[0].shape)


class TestEEGSynchronizationTable(TestCase):

    def setUp(self) -> None:
        data = np.random.randn(20, 171, 7)
        subj_list = [f'sub_{i}' for i in range(20)]
        pairs = PairsElectrodes(Electrodes)
        self.stable = EEGSynchronizationTable(data, subj_list, pairs.electrode_pairs, Bands)

    def test_get_2dim_data(self):

        el_pair = ('Fp1', 'Fp2')
        df = self.stable.get_2dim_data(band=None, el_pairs=el_pair)
        self.assertEqual(df.shape, (20, 7))
        df = self.stable.get_2dim_data(band='delta', el_pairs=None)
        self.assertTrue(True)


class TestPairedNonParametric(TestCase):

    def setUp(self) -> None:

        arr1 = np.random.randn(20, 3, 4, )
        arr2 = np.random.randn(20, 3, 4, )
        data1 = 0.3*np.random.randn(20, 171, 7)
        data2 = 0.5+0.3*np.random.randn(20, 171, 7)
        pairs = PairsElectrodes(Electrodes)
        subj_list = [f'sub_{i}' for i in range(20)]
        self.table1 = DataTable(data1, subj_list)
        self.table2 = DataTable(data2, subj_list)
        self.stable = EEGSynchronizationTable(data1, subj_list, pairs.electrode_pairs, Bands)

    def test_get_empirical_stat(self):

        pt = PairedNonParametric(self.table1, self.table2)
        diff = pt.get_emp_diff()
        stat_res = pt.get_empirical_stat(diff)

        self.assertEqual(stat_res.shape, diff.data[0].shape)

    def test_get_zero_eff_perm_non_parametric(self):
        pt = PairedNonParametric(self.table1, self.table2, type_stat='eff_size')
        rejected, p_vals = pt.get_zero_eff_perm_non_parametric(num=1000, alpha=0.05)
        share_rejected = np.mean(p_vals < 0.05)
        self.assertTrue((p_vals <= 1).all())

    def test_get_zero_eff_parametric(self):
        pt = PairedNonParametric(self.table1, self.table2, type_stat='eff_size')
        share_rejected = []
        for correction in [None, 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by',
                              'fdr_tsbh']:
            rejected, p_vals = pt.get_zero_eff_parametric(correction=correction)
            share_rejected.append(np.mean(p_vals < 0.05))
        self.assertTrue((p_vals <= 1).all())
