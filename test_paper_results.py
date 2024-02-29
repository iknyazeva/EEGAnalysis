from unittest import TestCase
from paper_results import full_table_stats, create_df_from_eeg_res, subsample_table_stats
from paper_results import create_df_from_eeg_res_with_sbsmpls
import pandas as pd
import numpy as np


class Test(TestCase):
    def test_full_table_stats(self):
        path_to_data = 'eeg_dataframe_nansfilled.csv'
        correction = 'bonferroni'
        eeg_res = full_table_stats(path_to_data,
                                   correction=correction,
                                   agg='wmean',
                                   per_num=10000,
                                   bs_num=5000)
        alpha = 0.05
        final = create_df_from_eeg_res(eeg_res, alpha=alpha,
                                       save_path='./repr_results')
        self.assertTrue(True)

    def test_subsample_table_stats(self):
        path_to_data = 'eeg_dataframe_nansfilled.csv'
        final_list = []
        for correction in ['np']:
            print(f"Compute method {correction}")
            eeg_res = subsample_table_stats(path_to_data=path_to_data,
                                            sample_size=[20, 25, 30, 40, 50, 70],
                                            num_samples=500,
                                            correction=correction,
                                            per_num=10000,
                                            agg='max',
                                            bs_num=1000,
                                            alpha=0.05
                                            )
            final = create_df_from_eeg_res_with_sbsmpls(eeg_res, alpha=0.05,
                                                        save_path='./repr_results')
            final_list.append(final)
        self.assertTrue(True)
