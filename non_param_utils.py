import numpy as np
from numba import jit, prange
import numba as nb
from scipy import stats
from statsmodels.stats.multitest import multipletests


@jit(nopython=True, cache=True)
def bootstrap_effect_size(diff_array, num=100):
    conf_int = nb.typed.List([2.5, 97.5])
    effect_sizes = np.zeros(num)
    diff_array = diff_array.copy()
    for i in range(num):
        bs_array = np.random.choice(diff_array, len(diff_array))
        effect_sizes[i] = np.mean(bs_array) / np.std(bs_array)
    return np.mean(effect_sizes), np.percentile(effect_sizes, nb.typed.List(conf_int))


@jit(nopython=True, cache=True)
def nb_get_diff_perm_3d_sample(diff_array):
    n = diff_array.shape[0]
    sig_perms = np.expand_dims(np.random.choice(np.array([1, -1]), n), axis=1)
    sig_perms_3d = np.expand_dims(sig_perms, axis=1)
    perm_diff = sig_perms_3d * diff_array
    return perm_diff, sig_perms_3d


def get_diff_perm_3d_sample(diff_array):
    n = diff_array.shape[0]
    sig_perms = np.random.choice(np.array([1, -1]), n).reshape(-1, 1, 1)
    perm_diff = sig_perms * diff_array
    return perm_diff, sig_perms


def np_2d_stat(arr, type_stat='eff_size'):
    if type_stat == 'eff_size':
        stats_2dim = np.abs(arr.mean(axis=0)) / arr.std(axis=0)
    elif type_stat == 't_stat':
        stats_2dim = np.sqrt(arr.shape[0]) * np.abs(arr.mean(axis=0)) / arr.std(axis=0)
    else:
        raise NotImplementedError('Only eff_size and t_stat are implemented')
    return stats_2dim


@jit(parallel=True, nopython=True, cache=True)
def nb_2d_stat(arr, type_stat='eff_size'):
    res = np.zeros((arr.shape[1], arr.shape[2]))

    for i in prange(arr.shape[1]):
        for j in prange(arr.shape[2]):
            if type_stat == 'eff_size':
                res[i, j] = abs(arr[:, i, j].mean()) / arr[:, i, j].std()
            elif type_stat == 't_stat':
                res[i, j] = np.sqrt(arr.shape[0]) * abs(arr[:, i, j].mean()) / arr[:, i, j].std()
            else:
                raise NotImplementedError('Only eff_size and t_stat are implemented')

    return res


@jit(parallel=True, nopython=True, cache=True)
def nb_2d_max_stat(arr):
    max_1 = np.zeros(arr.shape[0])
    max_2 = np.zeros(arr.shape[1])
    for i in prange(arr.shape[0]):
        max_1[i] = arr[i, :].max()
    for j in prange(arr.shape[1]):
        max_2[j] = arr[:, j].max()

    return max_1, max_2


@jit(nopython=True, cache=True)
def nb_permute_null_dist(diff_array,
                         num=100,
                         type_stat='eff_size'):
    null_stat_1dim = np.zeros((num, diff_array.shape[1]))
    null_stat_2dim = np.zeros((num, diff_array.shape[2]))
    for i in range(num):
        perm_diff, _ = nb_get_diff_perm_3d_sample(diff_array)

        stats_2dim = nb_2d_stat(perm_diff, type_stat=type_stat)
        max_1, max_2 = nb_2d_max_stat(stats_2dim)
        null_stat_1dim[i] = max_1
        null_stat_2dim[i] = max_2
    return null_stat_1dim, null_stat_2dim


def permute_null_dist(diff_array,
                      num=100,
                      conf_int=(95, 97.5, 99),
                      type_stat='eff_size',
                      return_dist=True):
    if diff_array.ndim != 3:
        raise NotImplementedError('only 2-dimensional features space is implemented')
    null_stat_1dim = np.zeros((num, diff_array.shape[1]))
    null_stat_2dim = np.zeros((num, diff_array.shape[2]))

    for i in range(num):
        perm_diff, _ = get_diff_perm_3d_sample(diff_array)

        stats_2dim = np_2d_stat(perm_diff, type_stat=type_stat)
        null_stat_1dim[i] = np.max(stats_2dim, axis=1)
        null_stat_2dim[i] = np.max(stats_2dim, axis=0)
    if return_dist:
        return null_stat_1dim, null_stat_2dim
    else:
        return np.percentile(null_stat_1dim, conf_int), np.percentile(null_stat_2dim, conf_int)


def np_compute_p_val_from_null(stats_2dim,
                               null_stat_1dim,
                               null_stat_2dim,
                               agg='wmean'):
    p_vals = np.zeros_like(stats_2dim)
    for i in range(stats_2dim.shape[0]):
        for j in range(stats_2dim.shape[1]):
            p1 = np.mean(null_stat_1dim[:, i] >= stats_2dim[i, j])
            p2 = np.mean(null_stat_2dim[:, j] >= stats_2dim[i, j])
            if agg == 'max':
                p_vals[i, j] = max(p1, p2)
            elif agg == 'min':
                p_vals[i, j] = min(p1, p2)
            elif agg == 'wmean':
                p_vals[i, j] = 0.8 * p1 + 0.2 * p2
            else:
                raise NotImplementedError('Only min, max, mean')
    return p_vals


@jit(parallel=True, nopython=True, cache=True)
def nb_compute_p_val_from_null(stats_2dim,
                               null_stat_1dim,
                               null_stat_2dim):
    p_vals = np.zeros((stats_2dim.shape[0], stats_2dim.shape[1]))
    for i in prange(stats_2dim.shape[0]):
        for j in prange(stats_2dim.shape[1]):
            p1 = np.mean(stats_2dim[i, j] >= null_stat_1dim[:, i])
            p2 = np.mean(stats_2dim[i, j] >= null_stat_2dim[:, j])
            p_vals[i, j] = max(p1, p2)
    return p_vals


def non_parametric_2d_testing(diff_arr, num=100, is_numba=False, type_stat='eff_size', agg='max'):
    if is_numba:
        stats_2dim = nb_2d_stat(diff_arr, type_stat=type_stat)
        null_stat_1dim, null_stat_2dim = nb_permute_null_dist(diff_arr, num=num, type_stat=type_stat)
        p_vals = nb_compute_p_val_from_null(stats_2dim, null_stat_1dim, null_stat_2dim)
    else:
        stats_2dim = np_2d_stat(diff_arr, type_stat=type_stat)
        null_stat_1dim, null_stat_2dim = permute_null_dist(diff_arr, num=num,
                                                           type_stat=type_stat, return_dist=True)
        p_vals = np_compute_p_val_from_null(stats_2dim, null_stat_1dim, null_stat_2dim, agg=agg)
    return p_vals


def mass_univariate_2d_testing(diff_arr,
                               correction='uncorr',
                               alpha=0.05):
    p_vals = stats.ttest_1samp(diff_arr, popmean=0, axis=0).pvalue
    assert correction in ['uncorr', 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by',
                          'fdr_tsbh'], 'Method is not recognized '
    if correction == 'uncorr':
        return p_vals
    else:

        _, p_vals_corr, _, _ = multipletests(p_vals.flatten(), alpha=alpha, method=correction)
        return p_vals_corr.reshape(*p_vals.shape)
