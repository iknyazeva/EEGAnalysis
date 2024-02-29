from unittest import TestCase
from paired_connectivity_analyzer import EEGPairedPermutationAnalyser
from plotting_utils import DrawEEG
from table_analyzer import SynchronizationTable
from plotting_utils import plot_compute_sign_differences, draw_edges_by_dict, plot_reproducibility_pattern, plot_reproducibility_by_frequency
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
from matplotlib import cm


class Test(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
        cls.table = SynchronizationTable(df)
        cls.draw_obj = DrawEEG()

    def test_draw_empty_edges(self):
        pair_names = []
        values_color = None
        values_width = None
        self.draw_obj.draw_edges(pair_names, values_color=values_color,
                                 values_width=values_width, normalize_values=True,
                                 cmap=cm.seismic)
        plt.show()
        self.assertTrue(True)

    def test_draw_edges(self):
        pair_names = ['P3/O2', 'Pz/O2']
        values_color = np.array([0.37406792, 1.70447801])
        values_width = np.exp([0.17406792, 0.7])* np.exp(1)
        self.draw_obj.draw_edges(pair_names, values_color=values_color,
                                 values_width=values_width, normalize_values=True,
                                 cmap=cm.seismic)
        plt.show()
        self.assertTrue(True)

    def test_extract_array_to_draw_edges_by_df(self):
        t_df = self.table.compute_individual_ttests(return_df=True)
        pair_names, values_color, values_width = self.draw_obj._extract_array_to_draw_edges_by_df(df=t_df,
                                           chan_col='chan_pair',
                                           color_col='t_stat',
                                           width_col=None)
        self.assertIsInstance(pair_names, np.ndarray)
        self.assertIsInstance(values_color, np.ndarray)
        self.assertTrue(values_width == None)


    def test_draw_by_df_single_band(self):
        stat_df = self.table.compute_stat_df(bts_num=100)
        stat_df['width'] = stat_df[['mean_eff_size']].applymap(lambda x: np.exp(abs(x)) * np.exp(1))
        self.draw_obj.draw_by_df(stat_df, band='alpha1',
                                 filter_by='sign_abh',
                                 color_col='mean_eff_size',
                                 width_col='width',
                                 sign='separate',
                                 color_label='t_stat')
        plt.show()

        self.assertTrue(True)

    def test_draw_by_df_band_list_same(self):
        stat_df = self.table.compute_stat_df(bts_num=100)
        stat_df['width'] = stat_df[['mean_eff_size']].applymap(lambda x: np.exp(abs(x)) * np.exp(1))
        fig = self.draw_obj.draw_by_df(stat_df, band=['beta1', 'alpha1', 'alpha2'],
                                 filter_by='sign_sidak',
                                 color_col='mean_eff_size',
                                 width_col='width',
                                 sign='same')
        plt.show()

        self.assertTrue(True)

    def test_draw_by_df_band_list_separate(self):
        stat_df = self.table.compute_stat_df(bts_num=100)
        stat_df['width'] = stat_df[['mean_eff_size']].applymap(lambda x: np.exp(abs(x)) * np.exp(1))
        fig = self.draw_obj.draw_by_df(stat_df, band=['beta1', 'alpha1', 'alpha2'],
                                 filter_by='sign_sidak',
                                 color_col='mean_eff_size',
                                 width_col='width',
                                 sign='separate')
        plt.show()

        self.assertTrue(True)




    def test_draw_edges_by_custom_dict(self):
        dict_diffs = {'chan_names': np.array(['P3/O2', 'Pz/O2'], dtype='<U7'),
                        'chan_diffs': np.array([0.37406792, 0.7]),
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
                                 thres=0.001, cmap=cm.cool, figsize=(6, 4))
        obj = plot_reproducibility_pattern(size=70, band=[1,2,3,4,5,6,7], num_perms=100, num_reps=50, factor=0.4,
                                           thres=0.001, cmap=cm.cool, figsize=(18, 4))
        self.assertTrue(True)

    def test_plot_reproducibility_by_frequency(self):

        pattern, (fig1,fig2) = plot_reproducibility_by_frequency(size=30, band=1, num_reps=10, num_perms=1000, thres=0.05,
                                      factor=0.01, replace=False,
                                      cmap=cm.cool, figsize=(18, 4), is_param=True)
        self.assertTrue(True)


    def test_plot_compute_sign_differences_int_band(self):
        obj = plot_compute_sign_differences(size=70, band=2, num_perms=100, thres=0.001,
                                            title=None, cmap=cm.gist_heat, figsize=(12, 4))
        self.assertTrue(True)

    def test_plot_compute_sign_differences_list_band(self):
        obj = plot_compute_sign_differences(size=70, band=[2, 3, 4], num_perms=100, thres=0.001,
                                            title=None, cmap=cm.gist_heat, figsize=(12, 4))
        self.assertTrue(True)

    def test_plot_compute_sign_differences_list_band(self):
        group_size = 30
        obj = plot_compute_sign_differences(size=group_size, band=[1, 2, 3, 4, 5, 6, 7], num_perms=10000, thres=0.0003,
                                            title=None, cmap=cm.PRGn, figsize=(18, 4))
        self.assertTrue(True)

    def test_plot_reproducibility_pattern(self):
        group_size = 60
        obj, pattern = plot_reproducibility_pattern(size=group_size, band=[1, 2, 3, 4, 5, 6, 7], num_perms=100,
                                                    num_reps=10, factor=0.51,
                                                    thres=0.001, cmap=cm.gist_heat, figsize=(18, 4), replace=True)
        self.assertTrue(True)

