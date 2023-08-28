import numpy as np
from eeg_data_class import PairsElectrodes, Electrodes, Bands
import numpy.typing as npt
import pandas as pd
from non_param_utils import mass_univariate_2d_testing, non_parametric_2d_testing
from typing import TypeVar, Iterable, Tuple, List, Callable, Optional, Union


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
                                        is_numba=False):
        p_vals = non_parametric_2d_testing(self.data, num=num, type_stat=type_stat, is_numba=is_numba)
        rejected = p_vals < alpha
        return rejected, p_vals

    def test_zero_effect_parametric(self, correction=None, alpha=0.05):
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

    def get_zero_eff_perm_non_parametric(self, num=1000, alpha=0.05):

        diff_table = self.get_emp_diff()
        rejected, p_vals = diff_table.test_zero_effect_non_parametric(num=num, alpha=alpha, type_stat=self.type_stat)

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

