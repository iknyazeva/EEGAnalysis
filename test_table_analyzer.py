from unittest import TestCase
import time
from table_analyzer import SynchronizationTable, bootstrap_effect_size
from table_analyzer import SubsampleTable
from table_analyzer import bool_dice, pairwise_bool_dice, vec_2_arr_bool_dice
import pandas as pd
import numpy as np
from multipy.fwer import bonferroni, holm_bonferroni, hochberg, sidak
from multipy.fdr import lsu, abh


class TestSynchronizationTable(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
        cls.table = SynchronizationTable(df)

    def test_compute_individual_ttests(self):
        p_val_dict, t_stat_dict = self.table.compute_individual_ttests()
        self.assertTrue(set(p_val_dict.keys()) ==
                        {'alpha1', 'beta1', 'delta', 'beta2', 'alpha2', 'theta', 'gamma'})


    def test_compute_individual_ttests_df(self):

        t_df = self.table.compute_individual_ttests(return_df=True)
        self.assertIsInstance(t_df, pd.DataFrame)

    def test_compute_significant_pval(self):
        p_val_dict, t_stat_dict = self.table.compute_individual_ttests()
        p_df = self.table.compute_significant_pval(p_val_dict, lsu, alpha=0.05, ih_weights=None)
        self.assertTrue(True)

    def test_compute_significant_by_method_list(self):
        p_val_dict, t_stat_dict = self.table.compute_individual_ttests()
        p_df = self.table.compute_significant_by_method_list(p_val_dict, (lsu, holm_bonferroni, sidak),
                                                             alpha=0.05, ih_weights=None)
        self.assertTrue(True)

    def test_get_effect_size(self):
        start = time.time()
        eff_size_dict = self.table.get_effect_size(num=100)
        end = time.time()
        print(end - start)
        self.assertIsInstance(eff_size_dict, dict)

    def test_compute_stat_df(self):
        start = time.time()
        stat_df = self.table.compute_stat_df(bts_num=100)
        end = time.time()
        print(end - start)
        self.assertIsInstance(stat_df, pd.DataFrame)


class TestSubsampleTable(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
        cls.table = SynchronizationTable(df)
    def test_get_subgroups(self):
        size = 70
        subs_table = SubsampleTable(self.table)
        subs_table.get_subgroups(size=size, overlay=size//10)
        self.assertTrue(len(subs_table.list_of_idxs) > 0)
        self.assertTrue()

    def test_compute_subgroup_stats(self):
        size = 60
        subs_table = SubsampleTable(self.table)
        df_repr, dice_dict = subs_table.compute_subgroup_stats(size=size, overlay=0,
                                                               bts_num=100, eff_thrs=(0.05, 0.1, 0.15))
        self.assertTrue(True)

    def test_repeat_n_subgroup_stats(self):
        size = 40
        subs_table = SubsampleTable(self.table)
        merged_dict, merged_df = subs_table.repeat_n_subgroup_stats(n=3, size=size, bts_num=1000)

        self.assertTrue(True)



class Test(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.diff_arr = np.random.randn(100)

    def test_bootstrap_effect_size(self):
        for j in range(3):
            start = time.time()
            eff_size, (low, upper) = bootstrap_effect_size(self.diff_arr, num=1000)
            end = time.time()
            print(end - start)
        self.assertTrue(eff_size < 0.5)
        self.assertTrue(low < upper)

    def test_bool_dice(self):
        u1 = np.random.choice([0, 1], size=5)
        v1 = np.random.choice([0, 1], size=5)
        u = np.array([1, 0, 1])
        v = np.array([1, 1, 1])
        coef = bool_dice(u, v)
        coef1 = bool_dice(u1, v1)
        self.assertTrue(bool_dice(u, u) == 1)

    def test_pairwise_bool_dice(self):
        start = time.time()
        arr = np.random.choice([0, 1], size=(200, 40), p=[0.9, 0.1])
        dice_list = pairwise_bool_dice(arr)
        end = time.time()
        print(end - start)
        self.assertTrue(True)
