import numpy as np
from eeg_data_class import PairsElectrodes, Electrodes, Bands
from tqdm import tqdm
from numba import jit
from collections import Counter
from itertools import combinations, product
import numpy.typing as npt
import pandas as pd
from scipy import stats
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
            return (self.data == other.data) and (self.subj_list == other.subj_list)
        else:
            raise NotImplementedError('Only eq between DataTables is supported')


    def compute_stat(self, type_stat: str = 'mean'):
        assert type_stat in ['mean', 'eff_size', 't_stat'], 'Only mean, eff_size and t_stat implemented'
        if type_stat == 'mean':
            return self.data.mean(axis=0)
        elif type_stat == 'eff_size':
            return np.abs(self.data.mean(axis=0)) / self.data.std(axis=0)
        elif type_stat == 't_stat':
            return np.sqrt(len(self.subj_list)) * np.abs(self.data.mean(axis=0)) / self.data.std(axis=0)
        else:
            raise NotImplementedError('Only three statistics implemented: "mean", "eff_size", "t_stat"')


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

    def get_emp_diff(self):
        return DataTable(self.table1-self.table2, self.table1.subj_list)

    def get_perm_diff(self):
        n = len(self.table1.subj_list)
        diff_table = self.get_emp_diff()
        return DataTable(np.random.choice([1, -1], n)*diff_table.data, self.table1.subj_list)

