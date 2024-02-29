from unittest import TestCase
from tables import EEGSynchronizationTable, PairedNonParametric, DataTable, Reproducibility
from eeg_data_class import PairsElectrodes, Bands, Electrodes
import numpy as np
import pickle


class TestDataTable(TestCase):

    def setUp(self):
        arr1 = np.random.randn(50, 3, 4)
        arr2 = 10 + np.random.randn(50, 3, 4)
        subj_list = [f'sub_{i}' for i in range(50)]
        self.table1 = DataTable(arr1, subj_list)
        self.table2 = DataTable(arr1, subj_list)
        self.table3 = DataTable(arr2, subj_list)

    def test_magic(self):
        self.assertEqual(self.table1, self.table2)
        diff = self.table3 - self.table2
        self.assertIsInstance(diff, DataTable)
        self.assertTrue(True)

    def test_get_subj_subsamples(self):
        subs = self.table1.get_subj_subsamples(20, type_subs='bs', num=5)
        self.assertEqual(len(subs), 5)
        subs = self.table1.get_subj_subsamples(20, type_subs='perm', num=5)
        self.assertTrue(len(subs), 5)

    def test_get_subtable_by_subjs(self):
        subs = self.table1.get_subj_subsamples(20, type_subs='bs', num=2)
        subtable = self.table1.get_subtable_by_subjs(subs[0])
        self.assertTrue(True)

    def test_compute_stat(self):
        self.assertRaises(NotImplementedError, self.table1.compute_stat, 'efff_size')
        eff_sizes = self.table1.compute_stat(type_stat='eff_size')
        self.assertEqual(eff_sizes.shape, self.table1.data[0].shape)

    def test_compute_eff_size_distribution(self):
        eff_array = self.table1.compute_eff_size_distribution(bs_num=100)
        self.assertTrue(True)


class TestEEGSynchronizationTable(TestCase):

    def setUp(self) -> None:
        data = np.random.randn(20, 171, 7)
        subj_list = [f'sub_{i}' for i in range(20)]
        pairs = PairsElectrodes(Electrodes)
        self.stable = EEGSynchronizationTable(data, subj_list, pairs.electrode_pairs, Bands)

    def test_read_from_eeg_dataframe(self):
        path_to_df = 'eeg_dataframe_nansfilled.csv'
        stable_fo = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_df, cond_prefix='fo')
        stable_fz = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_df, cond_prefix='fz')
        self.assertTrue(True)

    def test_get_2dim_data(self):
        el_pair = ('Fp1', 'Fp2')
        df = self.stable.get_2dim_data(band=None, el_pairs=el_pair)
        self.assertEqual(df.shape, (20, 7))
        df = self.stable.get_2dim_data(band='delta', el_pairs=None)
        self.assertTrue(True)


class TestPairedNonParametric(TestCase):

    def setUp(self) -> None:

        data1 = 0.3 * np.random.randn(20, 171, 7)
        data2 = 0.5 + 0.3 * np.random.randn(20, 171, 7)
        pairs = PairsElectrodes(Electrodes)
        subj_list = [f'sub_{i}' for i in range(20)]
        self.table1 = DataTable(data1, subj_list)
        self.table2 = DataTable(data2, subj_list)
        self.stable = EEGSynchronizationTable(data1, subj_list, pairs.electrode_pairs, Bands)

        path_to_df = 'eeg_dataframe_nansfilled.csv'
        self.stable_fo = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_df, cond_prefix='fo')
        self.stable_fz = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_df, cond_prefix='fz')

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

    def test_open_close(self):

        subs = self.stable_fo.get_subj_subsamples(150, type_subs='bs', num=2)
        stable_fo = self.stable_fo.get_subtable_by_subjs(subs[0])
        stable_fz = self.stable_fz.get_subtable_by_subjs(subs[0])
        pt = PairedNonParametric(stable_fz, stable_fo, type_stat='eff_size')
        rejected, p_vals_np = pt.get_zero_eff_perm_non_parametric(num=1000, alpha=0.05)
        share_rejected_np = np.mean(p_vals_np < 0.05)
        rejected_nocorr, p_vals_nocorr = pt.get_zero_eff_parametric(correction=None)
        share_rejected_nocorr = np.mean(p_vals_nocorr < 0.05)
        share_rejected_mtp = []
        for correction in ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by',
                           'fdr_tsbh']:
            rejected, p_vals = pt.get_zero_eff_parametric(correction=correction)
            share_rejected_mtp.append(np.mean(p_vals < 0.05))
        self.assertTrue(True)


class TestReproducibility(TestCase):

    def setUp(self) -> None:

        path_to_df = 'eeg_dataframe_nansfilled.csv'
        self.stable_fo = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_df, cond_prefix='fo')
        self.stable_fz = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_df, cond_prefix='fz')

    def test__compute_p_vals(self):
        #subj_lists = self.stable_fz.get_subj_subsamples(177, type_subs='bs', num=1)
        rpr = Reproducibility(self.stable_fz, self.stable_fo)
        rejected, p_vals = rpr._compute_p_vals(self.stable_fz,
                        self.stable_fo,
                        correction='np',
                        per_num=100, alpha=0.05, agg='wmean')
        self.assertTrue(True)

    def test__compute_p_vals_eff_size(self):

        rpr = Reproducibility(self.stable_fz, self.stable_fo)
        sample_size = 30
        subj_lists = self.stable_fz.get_subj_subsamples(sample_size, type_subs='bs', num=1)
        stable1 = rpr.table1.get_subtable_by_subjs(subj_lists[0])
        stable2 = rpr.table2.get_subtable_by_subjs( subj_lists[0])
        p_vals, eff_size = rpr._compute_p_vals_eff_size(stable1,
                                                        stable2,
                                                        correction='uncorr',
                                                        bs_num=10)
        self.assertTrue(True)




    def test_compute_p_vals_bs_samples(self):
        rpr = Reproducibility(self.stable_fz, self.stable_fo)
        for sample_size in [20, 30, 40, 50, 60, 70, 80]:
            print(f'In process {sample_size}')
            res = rpr.compute_p_vals_bs_samples(sample_size=sample_size, bs_num=500,
                               correction='np', agg='wmean', per_num=5000,
                               save_path='./repr_results')
        self.assertTrue(True)
    def test_dice_by_methods(self):
        subj_lists = self.stable_fz.get_subj_subsamples(177, type_subs='bs', num=1)
        rpr = Reproducibility(self.stable_fz, self.stable_fo)
        dice = rpr.dice_by_methods(subj_lists[0])
        self.assertTrue(True)

    def test_power_fdr_rel_full(self):
        rpr = Reproducibility(self.stable_fz, self.stable_fo)
        # ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by',
        # 'fdr_tsbh', 'np', None]
        # None, 'bonferroni', 'fdr_bh',
        av_p, av_fdr, av_count = {}, {}, {}
        for correction in ['uncorr', 'bonferroni', 'sidak', 'holm', 'fdr_bh', 'fdr_by', 'fdr_by', 'fdr_tsbh', 'np']:
            power, fdr, count = rpr.power_fdr_rel_full(sample_size=30, correction=correction, perm_num=5000,
                                                       full_correction='fdr_by', bs_num=5, agg='max')
            av_p[correction] = power
            av_fdr[correction] = fdr
            av_count[correction] = count[1:]
        self.assertTrue(True)

    def test_compare_btsp_to_full(self):
        rpr = Reproducibility(self.stable_fz, self.stable_fo)

        for sample_size in [20, 30, 40, 50, 60, 70, 80]:
            print(f'In process {sample_size}')
            res = rpr.compare_btsp_to_full(sample_size=sample_size, full_correction='bonferroni',
                                           bs_num=1000, perm_num=5000, agg='wmean', save_path='./repr_results')
        self.assertTrue(True)

    def test_bootstrap_reproducibility(self):

        rpr = Reproducibility(self.stable_fz, self.stable_fo)
        save_path = './repr_results'
        per_num = 5000
        for sample_size in [20, 25, 30, 40, 50, 60]:
            print(sample_size)
            dice_res = dict()
            for correction in ['uncorr', 'bonferroni', 'sidak', 'holm', 'fdr_bh', 'fdr_by', 'np']:

                dice_list = rpr.bootstrap_reproducibility(sample_size=sample_size, num=10, per_num=per_num, correction=correction,
                                                  agg='max', alpha=0.05)

                dice_res[correction]=dice_list
            dice_list = rpr.bootstrap_reproducibility(sample_size=sample_size, num=10, per_num=per_num, correction='np',
                                                  agg='wmean', alpha=0.05)
            dice_res['np_wmean'] = dice_list
            with open(f'{save_path}/dice_{sample_size}_in_group.pkl',
                  'wb') as f:
                pickle.dump(dice_res, f, protocol=pickle.HIGHEST_PROTOCOL)


        self.fail()


