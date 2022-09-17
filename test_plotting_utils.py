from unittest import TestCase
from paired_connectivity_analyzer import EEGPairedPermutationAnalyser, DrawEEG
from plotting_utils import plot_compute_sign_differences, draw_edges_by_dict, plot_reproducibility_pattern
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
from matplotlib import cm


class Test(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
        cls.analyzer = EEGPairedPermutationAnalyser(data_df=df, num_perm=100, thres=0.001)
        cls.draw_obj = DrawEEG()

    def test_draw_edges_by_custom_dict(self):
        dict_diffs = {'chan_names': np.array(['P3/O2', 'Pz/O2'], dtype='<U7'),
                        'chan_diffs': np.array([0.37406792, 0.30447801]),
                        'chan_pvals': np.array([0., 0.])}
        draw_obj = draw_edges_by_dict(dict_diffs, 1,
                                      cmap=cm.PRGn)
        plt.show()
        self.assertTrue(True)


    def test_draw_edges_by_dict(self):
        band = 2
        num_reps = 50
        factor = 0.4
        dict_reproducible = self.analyzer.compute_reproducible_pattern(size=70,
                                                                       num_reps=num_reps, factor=factor, band=band)
        title = f"Reproducible patterns in \"open-close\" contrast \n Rythm: {self.draw_obj.bands[band]}, num_reps: {num_reps}, repr.thrsh.: {factor}"
        draw_obj = draw_edges_by_dict(dict_reproducible, 1, keys=["channels", "mean_diff", "frequency"],
                                      cmap=cm.PRGn, title=title)
        plt.show()
        self.assertTrue(True)

    def test_draw_edges_by_empty_dict(self):
        band = 2
        num_reps = 50
        factor = 0.4
        dict_diffs = {'chan_names': [], 'chan_diffs': [], 'chan_pvals': []}
        draw_obj = draw_edges_by_dict(dict_diffs, 1)
        plt.show()
        self.assertTrue(True)


    def test_plot_reproducibility_pattern(self):
        obj = plot_reproducibility_pattern(size=70, band=2, num_perms=100, num_reps=50, factor=0.4,
                                 thres=0.001, cmap=cm.cool, figsize=(18, 4))
        obj = plot_reproducibility_pattern(size=70, band=[1,2,3,4,5,6,7], num_perms=100, num_reps=50, factor=0.4,
                                           thres=0.001, cmap=cm.cool, figsize=(18, 4))
        self.assertTrue(True)

    def test_plot_compute_sign_differences_int_band(self):
        obj = plot_compute_sign_differences(size=70, band=2, num_perms=100, thres=0.001,
                                            title=None, cmap=cm.gist_heat, figsize=(12, 4))
        self.assertTrue(True)

    def test_plot_compute_sign_differences_list_band(self):
        obj = plot_compute_sign_differences(size=70, band=[2, 3, 4], num_perms=100, thres=0.001,
                                            title=None, cmap=cm.gist_heat, figsize=(12, 4))
        self.assertTrue(True)
