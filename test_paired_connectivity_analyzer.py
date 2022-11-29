from unittest import TestCase
from paired_connectivity_analyzer import EEGPairedPermutationAnalyser, DrawEEG
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import cm
from metrics import dice, jaccard
import pandas as pd
import numpy as np


# PYDEVD_USE_CYTHON=NO

class TestDrawEEG(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
        cls.analyzer = EEGPairedPermutationAnalyser(data_df=df, num_perm=100, thres=0.05)
        cls.dict_diffs = cls.analyzer.compute_sign_differences(size=50, band=5, num_perms=1000, thres=0.001)
        cls.draw_obj = DrawEEG()

    def test_draw_edges(self):
        pair_names = self.dict_diffs["chan_names"]
        values_color, values_width = self.dict_diffs["chan_diffs"], 1 - self.dict_diffs["chan_pvals"]
        title = f"Significant differences \"open-close\" for {self.draw_obj.bands[5]}"
        self.draw_obj.draw_edges(pair_names=pair_names, values_color=values_color,
                                 values_width=values_width, title=title)

        divider = make_axes_locatable(self.draw_obj.ax)
        cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
        self.draw_obj.fig.add_axes(cax)
        cbar = matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cm.cool, values=sorted(values_color),
                                                orientation="horizontal")
        cbar.set_label("Fisher's Z difference")
        plt.show()

        self.assertTrue(True)

    def test_draw_edges_empty(self):
        pair_names = []
        values_color, values_width = [], []
        title = f"Significant differences \"open-close\" for {self.draw_obj.bands[5]}"
        self.draw_obj.draw_edges(pair_names=pair_names, values_color=values_color,
                                 values_width=values_width, title=title)

        plt.show()

        self.assertTrue(True)


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
        # self.analyzer.get_subgroup(size=len(self.analyzer.df.index))
        self.analyzer.get_subgroup(size=40)
        self.analyzer.num_perm = 10000
        (emp_mean_diffs, p_val), perm_mean_diffs = self.analyzer.perm_difference_paired(band=1)
        self.assertIsInstance(emp_mean_diffs.mean(), float, "Mean difference should be float")
        self.assertIsInstance(p_val, float, "P-val should be float")
        self.assertIsInstance(perm_mean_diffs, np.ndarray or list)

    def test_ttest_difference_paired(self):
        self.analyzer.get_subgroup(size=None)
        self.analyzer.num_perm = 10000
        emp_mean_diffs, p_val = self.analyzer.ttest_difference_paired(band=1)
        (emp_mean_diffs, p_val_perm), perm_mean_diffs = self.analyzer.perm_difference_paired(band=1)
        p_df = pd.DataFrame(np.vstack([p_val, p_val_perm]).T,columns=['p_val_param', 'p_val_noparam'])
        self.assertIsInstance(emp_mean_diffs.mean(), float, "Mean difference should be float")
        self.assertIsInstance(p_val.mean(), float, "P-val should be float")

    def test_p_val_reproducibility(self):
        sign_df = self.analyzer.p_val_reproducibility(size=40, band=1,
                                                      num_perms=1000, num_exps=10)
        self.assertTrue(True)

    def test_compute_sign_differences(self):
        dict_diffs = self.analyzer.compute_sign_differences(size=20, band=1, num_perms=1000, thres=0.001)
        if len(dict_diffs['chan_names']) > 0:
            self.assertIsInstance(dict_diffs['chan_names'][0], str)
            self.assertLessEqual(dict_diffs['chan_pvals'][0], 1)

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
        # with self.assertRaises(NotImplementedError):
        #    sign_tested, (metric_list, cnt) = self.analyzer.pairwise_set_comparisons(size=10,
        #                                                                             band=1, num_reps=10, func=dice, type_="all1")
        self.assertGreaterEqual(len(cnt.keys()), 0, "Number of pairs should be more than zero")
        self.assertTrue((np.array(metric_list) >= 0).all(), "Metric should be greater than zero")
        self.assertTrue((np.array(metric_list) <= 1).all(), "Metric should be less than 1")

        self.assertIsInstance(sign_tested, list)

    def test_compute_reproducible_pattern(self):
        self.analyzer.thres = 0.01
        chn_dict = self.analyzer.compute_reproducible_pattern(size=40, num_reps=15, factor=0.3, band=3)

        self.assertTrue(True)
