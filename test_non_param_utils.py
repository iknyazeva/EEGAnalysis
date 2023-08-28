from unittest import TestCase
import time
import numpy as np
from non_param_utils import permute_null_dist, get_diff_perm_3d_sample
from non_param_utils import nb_2d_stat, nb_2d_max_stat, nb_permute_null_dist, np_2d_stat
from non_param_utils import np_compute_p_val_from_null, nb_compute_p_val_from_null
from non_param_utils import mass_univariate_2d_testing, non_parametric_2d_testing
from scipy import stats

class Test(TestCase):

    def setUp(self) -> None:
        self.arr1 = np.random.randn(20, 6, 4)
        self.arr2 = 1 + np.random.randn(20, 6, 4)
        self.data = 0.5 + np.random.randn(20, 171, 7)


    def test_nb_2d_stat(self):
        res = nb_2d_stat(self.arr1, type_stat='t_stat')
        self.assertEqual(res.shape, self.arr1[0].shape)


    def test_nb_2d_max_stat(self):
        res = nb_2d_stat(self.arr1)
        max_1, max_2 = nb_2d_max_stat(res)
        self.assertEqual(max_1.size, res.shape[0])
        self.assertEqual(max_2.size, res.shape[1])
        self.assertEqual(max_1.ndim, 1)
        self.assertEqual(max_2.ndim, 1)

    def test_get_diff_perm_3d_sample(self):
        perm_sample, sig_perm = get_diff_perm_3d_sample(self.arr2-self.arr1)
        if sig_perm[0] == 1:
            self.assertEqual((self.arr2[0]-self.arr1[0]).mean(), perm_sample[0].mean())
        else:
            self.assertEqual((self.arr1[0]-self.arr2[0]).mean(), perm_sample[0].mean())

    def test_permute_null_dist(self):
        type_stat = 'eff_size'
        emp_stat = np_2d_stat(self.data, type_stat=type_stat)
        null_stat_1dim, null_stat_2dim = permute_null_dist(self.data, num=100,
                                                           type_stat=type_stat, return_dist=True)

        self.assertEqual(null_stat_1dim.shape[1], self.data.shape[1])
        self.assertEqual(null_stat_2dim.shape[1], self.data.shape[2])

    def test_np_compute_p_val_from_null(self):
        type_stat = 'eff_size'
        stats_2dim = np_2d_stat(self.data, type_stat=type_stat)
        null_stat_1dim, null_stat_2dim = permute_null_dist(self.data, num=10000,
                                                           type_stat=type_stat, return_dist=True)
        p_vals = np_compute_p_val_from_null(stats_2dim, null_stat_1dim, null_stat_2dim)

        self.assertTrue(True)

    def test_nb_permute_null_dist(self):
        type_stat = 'eff_size'
        start_time = time.time()
        stats_2dim = nb_2d_stat(self.data, type_stat=type_stat)
        _,_ = nb_permute_null_dist(self.data, num=10)
        next_time = time.time()
        print(f"Compilation time: {next_time-start_time}")
        null_stat_1dim, null_stat_2dim = nb_permute_null_dist(self.data, num=1000)
        print(f"Computation time: {time.time()-next_time}")
        self.assertEqual(null_stat_1dim.shape[1], self.data.shape[1])
        self.assertEqual(null_stat_2dim.shape[1], self.data.shape[2])

    def test_non_parametric_2d_testing(self):
        p_vals = non_parametric_2d_testing(self.data, num=1000)
        sh_rejected = np.mean(p_vals < 0.05)
        self.assertTrue((p_vals <= 1).all())
    def test_mass_univariate_2d_testing(self):
        p_vals = mass_univariate_2d_testing(self.data, correction=None)
        sh_rejected = np.mean(p_vals < 0.05)
        p_vals_bonf = mass_univariate_2d_testing(self.data, correction='bonferroni')
        sh_rejected_bonf = np.mean(p_vals_bonf < 0.05)
        p_vals_fdr = mass_univariate_2d_testing(self.data, correction='fdr_by')
        sh_rejected_fdr= np.mean(p_vals_fdr < 0.05)
        self.assertTrue((p_vals <= 1).all())

