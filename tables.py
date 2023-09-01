import numpy as np
from eeg_data_class import PairsElectrodes, Electrodes, Bands
import numpy.typing as npt
import pandas as pd
from non_param_utils import mass_univariate_2d_testing, non_parametric_2d_testing
from typing import TypeVar, Iterable, Tuple, List, Callable, Optional, Union
from reproducibility_utils import bool_dice, bool_power_fdr, pairwise_bool_dice
from reproducibility_utils import vec_2_arr_bool_dice, vec_2_arr_bool_power_fdr
import pickle


class DataTable:
    """Basic class for all the data
    """

    def __init__(self, data: npt.NDArray,
                 subj_list: List[str]):
        """

        Parameters
        ----------
        data (np.ndarray) with shape[0] equal to subject list and others
        subj_list (list of str): list of subjects
        """
        assert data.shape[0] == len(subj_list), "First dimension size should be equal to number of subjects"
        self.data = data
        self.subj_list = subj_list

    def __sub__(self, other):
        if isinstance(other, DataTable):
            return DataTable(self.data - other.data, self.subj_list)
        else:
            raise NotImplementedError('Only difference between DataTables is supported')

    def __eq__(self, other):
        if isinstance(other, DataTable):
            return (self.data == other.data).all() and (self.subj_list == other.subj_list)
        else:
            raise NotImplementedError('Only eq between DataTables is supported')

    def get_subj_subsamples(self, size:int, type_subs: str='bs', num:int=10,
                            replace:bool=False, overlay:int=10):
        assert size <= len(self.subj_list), f'Could not sample size {size} from smaller dataset'
        from operator import itemgetter
        n = len(self.subj_list)
        subjs_list = []
        if type_subs == 'bs':
            for i in range(num):
                idxs = np.random.choice(range(n), size=size, replace=replace)
                subjs_list.append(list(itemgetter(*idxs)(self.subj_list)))
        elif type_subs == 'perm':
            for i in range(num):
                idx_to_split = list(self.subj_list)
                full_len = len(idx_to_split)
                assert size < full_len // 2 + overlay, 'Not enough data for splitting'
                np.random.shuffle(idx_to_split)
                list_of_idxs = []
                for ndx in range(0, full_len, size):
                    if full_len - ndx >= size:
                        list_of_idxs.append(idx_to_split[ndx: ndx + size])
                    elif full_len - ndx < size <= full_len - ndx + overlay:
                        list_of_idxs.append(idx_to_split[ndx - overlay: ndx - overlay + size])
                subjs_list.append(list_of_idxs)
        return subjs_list

    def get_subtable_by_subjs(self, subj_list):

        try:
            sub_ids = [self.subj_list.index(sub) for sub in subj_list]
            new_data = self.data[sub_ids]
            return DataTable(new_data, subj_list)
        except:
            print('Not all subjects in index')




    def compute_stat(self, type_stat: str = 'mean'):
        # assert type_stat in ['mean', 'eff_size', 't_stat'], 'Only mean, eff_size and t_stat implemented'
        if type_stat == 'mean':
            return self.data.mean(axis=0)
        elif type_stat == 'eff_size':
            return np.abs(self.data.mean(axis=0)) / self.data.std(axis=0)
        elif type_stat == 't_stat':
            return np.sqrt(len(self.subj_list)) * np.abs(self.data.mean(axis=0)) / self.data.std(axis=0)
        else:
            raise NotImplementedError('Only three statistics implemented: "mean", "eff_size", "t_stat"')

    def test_zero_effect_non_parametric(self, num=1000,
                                        alpha=0.05,
                                        type_stat='eff_size',
                                        is_numba=False,
                                        agg='max'):
        p_vals = non_parametric_2d_testing(self.data, num=num, type_stat=type_stat, is_numba=is_numba, agg=agg)
        rejected = p_vals < alpha
        return rejected, p_vals

    def test_zero_effect_parametric(self, correction='uncorr', alpha=0.05):
        p_vals = mass_univariate_2d_testing(self.data, correction=correction)
        rejected = p_vals < alpha
        return rejected, p_vals


class EEGSynchronizationTable(DataTable):
    """ Class for EEG synchronization table, in a 3-dimensional array, where first dimension is subjects,
        second dimension is Electrode pairs, and third is key bands
    """

    def __init__(self, data: npt.NDArray, subj_list: List[str], el_pairs_list: Optional[List[Tuple[str]]],
                 bands: Union[Optional[List[str]], Bands]):
        '''

        Parameters
        ----------
        data (npt.NDArray):  array with synchronization data
        subj_list: list of subject, ordered according to first dimension
        el_list: list of pairs electrodes
        bands: list of bands
        '''
        super().__init__(data, subj_list)

        self.el_pairs_list = el_pairs_list
        if isinstance(bands, type(Bands)) or bands is None:
            self.bands = [b.name for b in Bands]
        else:
            self.bands = bands

    @classmethod
    def read_from_eeg_dataframe(cls, path_to_df,
                                cond_prefix='fo',
                                band_list=None):
        if band_list is None:
            band_list = [1,2,3,4,5,6,7]
            bands = Bands
        df = pd.read_csv(path_to_df, index_col=0)
        subj_list = list(df.index)
        els = list(map(lambda x: x.name, Electrodes))
        pairs = PairsElectrodes(Electrodes)
        pairs_list = list(df.columns)
        data = []
        for b in band_list:
            pairs_dict = pairs.create_pairs_dict(pairs_list, filter_by=[cond_prefix, f'_{b}_'])
            columns = [col[0] for col in list(pairs_dict.values())]
            data.append(df[columns].values)
        data = np.array(data).swapaxes(0, 1).swapaxes(1, 2)
        return cls(data, subj_list, list(pairs_dict.keys()), bands)

    def get_subtable_by_subjs(self, subj_list):

        try:
            sub_ids = [self.subj_list.index(sub) for sub in subj_list]
            new_data = self.data[sub_ids]
            return EEGSynchronizationTable(new_data, subj_list, self.el_pairs_list, self.bands)
        except:
            print('Not all subjects in index')


    def get_2dim_data(self,
                      band: Optional[str],
                      el_pairs: Optional[Union[Tuple[str], str]]):
        if isinstance(band, str):
            assert band in self.bands, 'Band should be in band list'
            if not (isinstance(el_pairs, str) or el_pairs is None):
                raise TypeError('Electrodes pairs should be an element from el_pairs_list or None')

        if isinstance(el_pairs, tuple):
            assert el_pairs in self.el_pairs_list, 'Electrode pairs should be in el_pairs_list'
            if not (isinstance(band, str) or band is None):
                raise TypeError('Band should be an element from band or None')

        if band is None:
            subset_data = self.data[:, self.el_pairs_list.index(el_pairs), :].squeeze()
            return pd.DataFrame(subset_data, columns=self.bands, index=self.subj_list)
        if el_pairs is None:
            subset_data = self.data[:, :, self.bands.index(band)].squeeze()
            return pd.DataFrame(subset_data, columns=self.el_pairs_list, index=self.subj_list)
        if isinstance(el_pairs, str) and isinstance(band, str):
            subset_data = self.data[:, self.el_pairs_list.index(el_pairs), self.bands.index(band)].squeeze()
            return pd.DataFrame(subset_data, columns=[el_pairs + '_' + band], index=self.subj_list)


class PairedNonParametric:

    def __init__(self, table1: DataTable,
                 table2: DataTable,
                 type_stat: str = 'eff_size'):
        assert table1.subj_list == table2.subj_list, 'Should be equal subject list'
        self.table1 = table1
        self.table2 = table2
        self.type_stat = type_stat



    def get_emp_diff(self):
        return self.table2 - self.table1

    def get_empirical_stat(self, diff: DataTable):
        return diff.compute_stat(type_stat=self.type_stat)

    def get_zero_eff_perm_non_parametric(self, num=1000, alpha=0.05, agg='max'):

        diff_table = self.get_emp_diff()
        rejected, p_vals = diff_table.test_zero_effect_non_parametric(num=num, alpha=alpha, type_stat=self.type_stat, agg=agg)

        return rejected, p_vals

    def get_zero_eff_parametric(self, correction=None, alpha=0.05):
        """

        Parameters
        ----------
        correction: str
                    available corrections: 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by',
                              'fdr_tsbh'
        alpha

        Returns
        -------

        """
        diff_table = self.get_emp_diff()
        rejected, p_vals = diff_table.test_zero_effect_parametric(correction, alpha)

        return rejected, p_vals

class Reproducibility:

    def __init__(self, table1, table2, ground_true=False):
        self.table1 = table1
        self.table2 = table2

    @staticmethod
    def compare_pairwise(p_vals1: npt.NDArray, p_vals2: npt.NDArray):

        return bool_dice(p_vals1.flatten(), p_vals2.flatten())

    def bootstrap_reproducibility(self, sample_size=30, num=10,
                                  correction='np',
                                  per_num=1000, alpha=0.05):
        """
        Compute dice coefficient each to each for subsample with fixed size.
        At first from the whole set permuted subsamples with no repetiotion (except overlay)
        is computed, then in this group each2each dice is computed
        Parameters
        ----------
        sample_size (int):
        interested sample size

        Returns
        -------

        """

        subj_lists = self.table1.get_subj_subsamples(sample_size, type_subs='perm', num=num)
        dice_list = []
        for subj_list in subj_lists:
            rejected_arr = []
            for subjs in subj_list:
                stable1 = self.table1.get_subtable_by_subjs(subjs)
                stable2 = self.table2.get_subtable_by_subjs(subjs)
                rejected, _ = self._compute_p_vals(stable1, stable2, correction, per_num, alpha)
                rejected_arr.append(rejected.flatten())
            dice_list.extend(pairwise_bool_dice(np.array(rejected_arr).T))
        return dice_list

    def dice_by_methods(self, subjs, perm_num=1000,
                        method_to_comp='np',
                        methods=None, alpha=0.05):

        stable1 = self.table1.get_subtable_by_subjs(subjs)
        stable2 = self.table2.get_subtable_by_subjs(subjs)
        rejected_arr = []
        if methods is None:
            methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by',
                              'fdr_tsbh', 'np', None]

        for correction in methods:

            rejected, _ = self._compute_p_vals(stable1, stable2, correction, perm_num, alpha)
            rejected_arr.append(rejected.flatten())
            if correction == method_to_comp:
                rejected_to_comp = rejected.flatten()
        dice_list = vec_2_arr_bool_dice(rejected_to_comp, np.array(rejected_arr).T)
        return dice_list

    def power_fdr_rel_full(self, sample_size=30,
                           bs_num=10,
                           perm_num=1000,
                           alpha=0.05,
                           correction=None,
                           full_correction='np',
                           agg='max'):

        rej_full, _ = self._compute_p_vals(self.table1, self.table2,
                                           full_correction, perm_num, alpha, agg='max')

        subj_lists = self.table1.get_subj_subsamples(sample_size, type_subs='bs', num=bs_num)
        rejected_arr = []
        count = [np.sum(rej_full)]
        for subjs in subj_lists:
            stable1 = self.table1.get_subtable_by_subjs(subjs)
            stable2 = self.table2.get_subtable_by_subjs(subjs)
            rejected, _ = self._compute_p_vals(stable1, stable2, correction, perm_num, alpha, agg=agg)
            count.append(np.sum(rejected))
            rejected_arr.append(rejected.flatten())


        power, fdr = vec_2_arr_bool_power_fdr(rej_full.flatten(), np.array(rejected_arr).T)
        return power, fdr, count





    def _compute_p_vals(self,
                        table1,
                        table2,
                        correction='np',
                        per_num=1000, alpha=0.05, agg='max'):

        assert correction in ['uncorr','bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by',
                              'fdr_tsbh', 'np'], f'{correction} not available'

        pt = PairedNonParametric(table1, table2)
        if correction == 'np':
            rejected, p_vals = pt.get_zero_eff_perm_non_parametric(num=per_num, alpha=alpha, agg=agg)
        else:
            rejected, p_vals = pt.get_zero_eff_parametric(correction=correction, alpha=alpha)
        return rejected, p_vals

    def compare_btsp_to_full(self, sample_size=30, full_correction='fdr_by',
                               bs_num=5, perm_num=5000, agg='max', save_path='./repr_results'):
        power_d, fdr_d, count_d = {}, {}, {}
        for correction in ['uncorr', 'bonferroni', 'sidak', 'holm', 'fdr_bh', 'fdr_by', 'fdr_by', 'fdr_tsbh', 'np']:
            power, fdr, count = self.power_fdr_rel_full(sample_size=sample_size, correction=correction, perm_num=perm_num,
                                                       full_correction='fdr_by', bs_num=bs_num, agg=agg)
            power_d[correction] = power
            fdr_d[correction] = fdr
            count_d[correction] = count[1:]
        res = {'sample_size': sample_size, 'full_correction': full_correction, 'full_count': count[0],
                'perm_num': perm_num, 'power': power_d, 'fdr': fdr_d, 'count': count_d}
        if save_path:
            with open(f'{save_path}/repr_res{sample_size}_full_{full_correction}_agg_{agg}_perm_num_{perm_num}.pkl', 'wb') as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        return res