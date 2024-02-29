from unittest import TestCase
from plotting_tables import DrawEEG1020, DrawDfEEG
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
from matplotlib import cm


class TestDrawEEG1020(TestCase):

    def test_draw_edges(self):
        pair_names = [('P3', 'O2'), ('Pz', 'O2')]
        values_color = np.array([0.37406792, 1.70447801])
        values_width = np.exp([0.17406792, 0.7]) * np.exp(1)
        draw_obj = DrawEEG1020()
        draw_obj.draw_edges(chan_pairs=pair_names, values_color=values_color,
                            values_width=values_width)
        plt.show()


class TestDrawDfEEG(TestCase):

    def setUp(self):
        filename = './repr_results/stats_full_with_sbs_np_wmean_10000_num_smpls_500_eff_bs_num_1000.csv'
        df = pd.read_csv(filename)
        self.df = df.rename(columns={df.columns[0]: 'chan_pair'})

    def test_get_values_to_draw(self):
        obj = DrawDfEEG(self.df)
        filter_by = self.df['abs_eff_size'].values > 0.3
        pair_names, values_color, values_width = obj.get_values_to_draw(chan_col='chan_pair',
                                                                        band_col='band',
                                                                        band='alpha1',
                                                                        sign='both',
                                                                        filter_by=filter_by,
                                                                        color_col='mean_eff_size',
                                                                        width_col='abs_eff_size')
        obj.draw_obj.draw_edges(chan_pairs=pair_names,
                                values_color=values_color,
                                values_width=values_width,
                                normalize_values=True,
                                cmap=cm.seismic)

        plt.show()

    def test_draw_edges(self):
        obj = DrawDfEEG(self.df)
        filter_by = self.df['abs_eff_size'].values > 0.3
        obj.draw_edges(band_col='band',
                       band='alpha1',
                       chan_col='chan_pair',
                       filter_by=filter_by,
                       color_col='mean_eff_size',
                       width_col='abs_eff_size',
                       sign='same')
        plt.show()

    def test_draw_edges_band_list(self):
        obj = DrawDfEEG(self.df)
        filter_by = self.df['abs_eff_size'].values >= 0.8
        band_list = ['delta', 'theta', 'alpha1', 'alpha2', 'beta1', 'beta2', 'gamma']
        obj.draw_edges(band_col='band',
                       band=band_list,
                       chan_col='chan_pair',
                       filter_by=filter_by,
                       color_col='mean_eff_size',
                       width_col='abs_eff_size',
                       sign='same')
        #plt.show()
        obj.draw_obj.fig.suptitle(f"Significant differences for Open/Close conditions (eff_size>0.8)", fontsize=16)
        plt.savefig('SuperBigEff.png')
        self.assertTrue(True)

    def test_draw_edges_band_list_repe(self):
        obj = DrawDfEEG(self.df)
        filter_by = self.df['abs_eff_size'].values > 0.2
        band_list = ['delta', 'theta', 'alpha1', 'alpha2', 'beta1', 'beta2', 'gamma']
        obj.draw_edges(band_col='band',
                       band=band_list,
                       chan_col='chan_pair',
                       filter_by=filter_by,
                       color_col='freq_repr_sz_25',
                       width_col='abs_eff_size',
                       cmap=cm.jet,
                       normalize_values=False,
                       sign='same')
        plt.show()
        self.assertTrue(True)
