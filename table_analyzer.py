import numpy as np
from tqdm import tqdm
from numba import jit
from collections import Counter
from itertools import combinations, product
import numpy.typing as npt
import pandas as pd
from scipy import stats
from typing import TypeVar, Iterable, Tuple, List, Callable, Optional
from multipy.fdr import lsu, abh
from multipy.fwer import bonferroni, holm_bonferroni, hochberg, sidak
from scipy.spatial.distance import dice

PDict = TypeVar('PDict')


class SynchronizationTable:
    """ Class for identification of significant pattern
    and their uncertainty for EEG synchronization results
    """
    key_bands = {1: 'delta', 2: 'theta', 3: 'alpha1', 4: 'alpha2', 5: 'beta1', 6: 'beta2', 7: 'gamma'}
    open_postfix = 'fo'
    close_postfix = 'fz'

    ihw_coeffs = {'delta': {'theta': 0.8,
                            'alpha1': 0.4,
                            'alpha2': 0.2,
                            'beta1': 0,
                            'beta2': 0,
                            'gamma': 0},
                  'theta': {'delta': 0.8,
                            'alpha1': 0.8,
                            'alpha2': 0.4,
                            'beta1': 0.2,
                            'beta2': 0,
                            'gamma': 0},
                  'alpha1': {'delta': 0.4,
                             'theta': 0.8,
                             'alpha2': 0.8,
                             'beta1': 0.4,
                             'beta2': 0.2,
                             'gamma': 0},

                  'alpha2': {'delta': 0.2,
                             'theta': 0.4,
                             'alpha1': 0.8,
                             'beta1': 0.8,
                             'beta2': 0.4,
                             'gamma': 0.2},
                  'beta1': {'delta': 0,
                            'theta': 0.2,
                            'alpha1': 0.4,
                            'alpha2': 0.8,
                            'beta2': 0.8,
                            'gamma': 0.4},
                  'beta2': {'delta': 0,
                            'theta': 0,
                            'alpha1': 0.2,
                            'alpha2': 0.4,
                            'beta1': 0.8,
                            'gamma': 0.8},
                  'gamma': {'delta': 0,
                            'theta': 0,
                            'alpha1': 0,
                            'alpha2': 0.2,
                            'beta1': 0.4,
                            'beta2': 0.8},
                  }

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        # list of channel pairs taken from df columns
        self.df.columns = [el.strip() for el in self.df.columns]

        self.channel_bivar: List[str] = list(set([el.split('_')[0] for el in list(self.df.columns)]))

    def compute_individual_ttests(self,
                                  return_df=False) -> Tuple[PDict, PDict]:
        p_val_dict = dict()
        t_stat_dict = dict()
        for key, value in self.key_bands.items():
            p_val_dict[value] = dict()
            t_stat_dict[value] = dict()
            for chan_pair in self.channel_bivar:
                t_stat, p_val = stats.ttest_rel(self.df[f"{chan_pair}_{key}_{self.open_postfix}"].values,
                                                self.df[f"{chan_pair}_{key}_{self.close_postfix}"].values)
                p_val_dict[value][chan_pair] = p_val
                t_stat_dict[value][chan_pair] = t_stat
        if return_df:
            pval_df = pd.DataFrame(p_val_dict).reset_index().rename(columns={'index': 'chan_pair'}).melt(
                id_vars='chan_pair',
                var_name='band',
                value_name='p_val')
            tstat_df = pd.DataFrame(t_stat_dict).reset_index().rename(columns={'index': 'chan_pair'}).melt(
                id_vars='chan_pair',
                var_name='band',
                value_name='t_stat')
            pval_df = pval_df.merge(tstat_df)
            return pval_df
        else:
            return p_val_dict, t_stat_dict

    def compute_stat_df(self, alpha: float = 0.05,
                        bts_num: int = 100):
        p_val_dict, t_stat_dict = self.compute_individual_ttests(return_df=False)
        stat_df = self.compute_significant_by_method_list(p_val_dict, alpha=alpha)
        eff_size_dict = self.get_effect_size(conf_int=(2.5, 97.5), num=bts_num)

        eff_df = pd.DataFrame(eff_size_dict).reset_index()
        eff_df = eff_df.rename(columns={'index': 'chan_pair'}).melt(id_vars='chan_pair',
                                                                    var_name='band',
                                                                    value_name='eff_size')
        eff_df['mean_eff_size'] = eff_df[['eff_size']].applymap(lambda x: x[0])
        eff_df['low_eff_size'] = eff_df[['eff_size']].applymap(lambda x: x[1][0])
        eff_df['upper_eff_size'] = eff_df[['eff_size']].applymap(lambda x: x[1][1])
        eff_df.drop('eff_size', axis=1, inplace=True)
        stat_df = stat_df.merge(eff_df)
        return stat_df

    def compute_significant_pval(self, p_val_dict: PDict,
                                 method: Callable = lsu,
                                 alpha: float = 0.05,
                                 ih_weights: Optional[PDict] = None,
                                 return_df: bool = True) -> pd.DataFrame:
        """ Compute corrections for p-val for multiple comparisons

        Parameters
        ----------
        p_val_dict: PDict
                    dictionary with uncorrected p-values for each channel and band
        method: Callable
                method for multiple comparisons,
                available methods: bonferroni, hochberg, holm_bonferroni, sidak, lsu, abh
        alpha: float in [0,1]
                The desired critical level.
        ih_weights: PDict
            optional parameter for independent hypothesis weighting
        return_df: bool
            if return in pandas DataFrame


        Returns
        -------
            dict with significant channels
        """

        supported_methods = ['bonferroni', 'hochberg', 'holm_bonferroni', 'sidak', 'lsu', 'abh']
        if (method.__name__ not in supported_methods):
            raise Exception('The method %s is not supported!' % method.__name__)

        p_df = pd.DataFrame(p_val_dict).reset_index()
        p_df = p_df.rename(columns={'index': 'chan_pair'}).melt(id_vars='chan_pair',
                                                                var_name='band',
                                                                value_name='p_val')

        pvals = p_df['p_val'].values

        if ih_weights is None:
            significant = method(pvals, alpha)
        else:
            weights = np.abs(ih_weights)
            weights = weights / np.mean(weights)
            pvals = pvals / weights
            significant = method(pvals, alpha)
        if return_df:
            p_df['significant'] = significant
            return p_df
        else:
            return significant

    def compute_significant_by_method_list(self, p_val_dict: PDict,
                                           method_list: Tuple[Callable] = (
                                                   holm_bonferroni, sidak, bonferroni, lsu),
                                           alpha: float = 0.05,
                                           ih_weights: Optional[PDict] = None) -> pd.DataFrame:

        p_df = pd.DataFrame(p_val_dict).reset_index().rename(columns={'index':
                                                                          'chan_pair'}).melt(id_vars='chan_pair',
                                                                                             var_name='band',
                                                                                             value_name='p_val')
        for method in method_list:
            sign_p = self.compute_significant_pval(p_val_dict, method,
                                                   alpha, None, return_df=False)
            p_df['sign_' + method.__name__] = sign_p
        if ih_weights:
            for method in method_list:
                sign_p = self.compute_significant_pval(p_val_dict, method,
                                                       alpha, ih_weights, return_df=False)
                p_df['sign_ihw_' + method.__name__] = sign_p
        return p_df

    def get_effect_size(self, conf_int=(2.5, 97.5), num=100) -> dict:
        """
        Bootstrap effect size for selected bands, where sign_df is the dataFrame with chan_pairs, band and significance
        Parameters
        ----------
        conf_int: tuple of float
        num: int
            Number of bootstrap samples
        Returns
        -------
        dict for each band and channel with mean effect size and percentiles
        """

        eff_size_dict = dict()
        for key, value in self.key_bands.items():
            eff_size_dict[value] = dict()
            for chan_pair in self.channel_bivar:
                diff_array = self.df[f"{chan_pair}_{key}_{self.open_postfix}"].values - \
                             self.df[f"{chan_pair}_{key}_{self.close_postfix}"].values
                eff_size_dict[value][chan_pair] = bootstrap_effect_size(diff_array, num=num, conf_int=conf_int)
        return eff_size_dict


class SubsampleTable:
    # todo посчитать частоту воспроизводимости для всех каналов  и dice
    def __init__(self, obj: SynchronizationTable):
        self.big_table = obj
        self.full_index = obj.df.index
        self.list_of_idxs = None

    def get_subgroups(self, size: int = 30,
                      overlay: int = 0):

        idx_to_split = list(self.full_index)
        full_len = len(idx_to_split)
        assert size < full_len // 2 + overlay, 'Not enough data for splitting'
        np.random.shuffle(idx_to_split)
        list_of_idxs = []
        for ndx in range(0, full_len, size):
            if full_len - ndx >= size:
                list_of_idxs.append(idx_to_split[ndx: ndx + size])
            elif full_len - ndx < size <= full_len - ndx + overlay:
                list_of_idxs.append(idx_to_split[ndx - overlay: ndx - overlay + size])
        self.list_of_idxs = list_of_idxs

    def compute_one_subroup_stats(self, subgroup_ids, uncorr_levels=(0.01, 0.05),
                              ground_true: np.ndarray = None, bts_num=1000,
                              eff_thrs=(0.1, 0.15, 0.2)):
        """
        Compute statistics of reproducibility compared to ground true for subgroups
        Parameters
        ----------
        subgroup_ids: list of string
            ids from big table index
        uncorr_levels: tuple or list of floats
            uncorrected level for p-values
        ground_true: array of Bool
            array with ground trues
        bts_num: int
            number for bootstrap
        eff_thrs: list or tuple of floats
            thresholds for effect sizes

        Returns
        -------
        statistics data frame and reproducibility metrics: dice, fdr, power
        """
        df_subgroup = self.big_table.df.loc[subgroup_ids]
        sub_table = SynchronizationTable(df_subgroup)
        stat_df = sub_table.compute_stat_df(bts_num=bts_num)
        for uncorr_level in uncorr_levels:
            stat_df[f'sign_uncorr_{uncorr_level}'] = stat_df['p_val'] < uncorr_level
        for thrs in eff_thrs:
            stat_df[f'sign_eff_{thrs}'] = (stat_df['low_eff_size'] > thrs) | (stat_df['upper_eff_size'] < -thrs)
        cols = list(stat_df.filter(regex='sign').columns)

        ground_stat = pd.DataFrame(vec_2_arr_bool_dice(ground_true, stat_df[cols].values), index=cols, columns=['dice'])
        power, fdr = vec_2_arr_bool_power_fdr(ground_true, stat_df[cols].values)
        ground_stat['fdr'] = fdr
        ground_stat['power'] = power
        ground_stat['count'] = stat_df[cols].sum()
        return stat_df[cols], ground_stat

    def compute_subgroups_stats(self, size=30, overlay=0, uncorr_levels=(0.01, 0.05),
                                ground_true_col='sign_eff_0.15',
                                bts_num=1000, eff_thrs=(0.1, 0.15, 0.2)):

        stat_df = self.big_table.compute_stat_df(bts_num=bts_num)
        for thrs in eff_thrs:
            stat_df[f'sign_eff_{thrs}'] = (stat_df['low_eff_size'] > thrs) | (stat_df['upper_eff_size'] < -thrs)
        ground_true = stat_df[ground_true_col].values
        self.get_subgroups(size, overlay)
        sign_data = []
        ground_stat = []
        for subgroup_ids in self.list_of_idxs:

            stat_df_sbg, ground_stat_sbg = self.compute_one_subroup_stats(subgroup_ids, uncorr_levels=uncorr_levels,
                                                                        ground_true=ground_true, bts_num=bts_num,
                                                                        eff_thrs=eff_thrs)
            cols = stat_df_sbg.columns
            sign_data.append(stat_df_sbg.values)
            ground_stat.append(ground_stat_sbg)

        dice_dict = dict()
        for i, col in enumerate(cols):
            dice_dict[col] = pairwise_bool_dice(np.array(sign_data)[:, :, i].T)
        df_repr = pd.DataFrame(data=np.array(sign_data).mean(axis=0), columns=cols)
        df_repr['chan_pair'] = stat_df['chan_pair']
        df_repr['band'] = stat_df['band']
        ground_stat_df = pd.concat(ground_stat, axis=0).reset_index()

        return df_repr, ground_stat_df, dice_dict

    def repeat_n_subgroup_stats(self, n=5, size=30, overlay=10, ground_true_col='sign_eff_0.15',
                                uncorr_levels=(0.01, 0.05),
                                bts_num=1000, eff_thrs=(0.05, 0.1, 0.15)):
        _reprod_values = []
        _stat_dfs_list = []
        dice_within_dict = dict()
        for i in tqdm(range(n)):
            df_repr, ground_stat_df, dice_dict = self.compute_subgroups_stats(size=size, overlay=overlay,
                                                                    uncorr_levels=uncorr_levels,
                                                                    ground_true_col=ground_true_col,
                                                                    bts_num=bts_num, eff_thrs=eff_thrs)

            if i == 0:
                dice_within_dict = dice_dict
            else:
                for k,v in dice_within_dict.items():
                    v.extend(dice_dict[k])

            _stat_dfs_list.append(ground_stat_df)
            _reprod_values.append(df_repr.filter(regex='sign').values)
        merged_df = pd.DataFrame(data=np.array(_reprod_values).mean(axis=0),
                                 columns=list(df_repr.filter(regex='sign').columns))

        merged_df['chan_pair'] = df_repr['chan_pair']
        merged_df['band'] = df_repr['band']
        ground_stat_dfs = pd.concat(_stat_dfs_list)
        ground_stat_dfs['sample_size'] = size
        dice_df = pd.DataFrame(dice_within_dict)
        dice_within_df = dice_df.melt(value_name='dice_coeff', var_name='corr_method')
        dice_within_df['corr_method'] = dice_within_df[['corr_method']].applymap(lambda x: x[5:])
        dice_within_df['sample_size'] = size
        return ground_stat_dfs, merged_df, dice_within_df


@jit(nopython=True, cache=True)
def bootstrap_effect_size(diff_array, num=100,
                          conf_int=(2.5, 97.5)):
    effect_sizes = np.zeros(num)
    diff_array = diff_array.copy()
    for i in range(num):
        bs_array = np.random.choice(diff_array, len(diff_array))
        effect_sizes[i] = np.mean(bs_array) / np.std(bs_array)
    return np.mean(effect_sizes), np.percentile(effect_sizes, conf_int)


@jit(nopython=True, cache=True)
def bool_dice(u, v):
    not_u = ~u
    not_v = ~v
    nft = (not_u & v).sum()
    ntf = (u & not_v).sum()
    ntt = (u & v).sum()
    return float((2.0 * ntt) / np.array(2.0 * ntt + ntf + nft))


@jit(nopython=True, cache=True)
def bool_power_fdr(u, v):
    not_u = ~u
    fdr = float((not_u & v).sum() / v.sum()) if v.sum() > 0 else 0
    power = float((u & v).sum() / u.sum()) if u.sum() > 0 else 0
    return power, fdr


@jit(nopython=True, cache=True)
def vec_2_arr_bool_dice(v, arr):
    dice_list = []
    n = arr.shape[1]
    for i in range(n):
        dice_list.append(bool_dice(v, arr[:, i]))
    return dice_list


@jit(nopython=True, cache=True)
def vec_2_arr_bool_power_fdr(v, arr):
    power = []
    fdr = []
    n = arr.shape[1]
    for i in range(n):
        ntt, nft = bool_power_fdr(v, arr[:, i])
        power.append(ntt)
        fdr.append(nft)
    return power, fdr


@jit(nopython=True, cache=True)
def pairwise_bool_dice(arr):
    dice_list = []
    n = arr.shape[1]
    for i in range(n - 1):
        for j in range(i + 1, n):
            dice_list.append(bool_dice(arr[:, i], arr[:, j]))
    return dice_list
