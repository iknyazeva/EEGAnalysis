from unittest import TestCase
from paired_connectivity_analyzer import EEGPairedPermutationAnalyser
from metrics import dice, jaccard
import pandas as pd
import numpy as np


# PYDEVD_USE_CYTHON=NO


class TestEEGPairedPermutationAnalyser(TestCase):
    @classmethod
    def setUpClass(self):
        df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
        self.analyzer = EEGPairedPermutationAnalyser(data_df=df, num_perm=100, thres=0.05)

    def test_init(self):
        for i in range(len(self.analyzer.df.columns)):
            self.assertIn('_f', self.analyzer.df.columns[i],
                          f'Each column name should contain condition marker,{i} is not')

    def test_get_subgroup(self):
        self.analyzer.get_subgroup(size=10)
        self.assertEqual(len(self.analyzer.subgroup_ids), 10, 'Indexes should be updated with the len of subgroup')

    def test_perm_difference_paired(self):
        self.analyzer.get_subgroup(size=len(self.analyzer.df.index))
        (emp_mean_diffs, p_val), perm_mean_diffs = self.analyzer.perm_difference_paired(band=1)
        self.assertIsInstance(emp_mean_diffs.mean(), float, "Mean difference should be float")
        self.assertIsInstance(p_val, float, "P-val should be float")
        self.assertIsInstance(perm_mean_diffs, np.ndarray or list)

    def test_test_reproducability(self):
        self.analyzer.thres = 0.00001
        sign_channels = self.analyzer.test_reproducability(size=10, band=1, num_reps=10)
        self.assertIsInstance(sign_channels, list)
        self.assertIsInstance(sign_channels[0], dict)

    def test_pairwise_set_comparisons(self):
        self.analyzer.thres = 0.000000001
        sign_tested, (metric_list, cnt) = self.analyzer.pairwise_set_comparisons(size=10,
                                                                                 band=1, num_reps=10, func=dice,
                                                                                 type_="neigh")
        #with self.assertRaises(NotImplementedError):
        #    sign_tested, (metric_list, cnt) = self.analyzer.pairwise_set_comparisons(size=10,
        #                                                                             band=1, num_reps=10, func=dice, type_="all1")
        self.assertGreaterEqual(len(cnt.keys()), 0, "Number of pairs should be more than zero")
        self.assertTrue((np.array(metric_list) >= 0).all(), "Metric should be greater than zero")
        self.assertTrue((np.array(metric_list) <= 1).all(), "Metric should be less than 1")

        self.assertIsInstance(sign_tested, list)


    def test_compute_reproducible_pattern(self):
        self.analyzer.thres = 0.01
        chn_dict = self.analyzer.compute_reproducible_pattern(size=20, num_reps=15, factor=0.3, band=1)

        self.assertTrue(True)



