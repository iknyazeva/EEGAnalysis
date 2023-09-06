from unittest import TestCase
from eeg_data_class import Electrodes, PairsElectrodes, EEGdata, Bands, EEGdata1020
from itertools import combinations
import pandas as pd
import numpy as np


class TestPairsElectrodes(TestCase):

    def setUp(self) -> None:
        df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
        self.pairs_list = list(df.columns)

    def test_electrode_pairs(self):
        els = map(lambda x: x.name, Electrodes)
        res = list(combinations(els, 2))
        pairs = PairsElectrodes(Electrodes)
        self.assertTrue(True)
        #self.fail()

    def test_create_pairs_dict(self):
        pairs = PairsElectrodes(Electrodes)
        pairs_dict = pairs.create_pairs_dict(self.pairs_list)
        pairs_dict = pairs.create_pairs_dict(self.pairs_list, filter_by=['fo', '_1_'])
        self.assertTrue(True)


class TestEEGdata(TestCase):
    def setUp(self) -> None:
        self.data = np.random.randn(19, 7)
        self.data_corr = np.random.randn(171, 7)
        self.electrodes = Electrodes
        self.pairs = PairsElectrodes(Electrodes)

    def test_set_values_to_electrodes(self):
        eeg = EEGdata(Electrodes,  Bands, self.pairs.electrode_pairs)
        eeg.set_values_to_electrodes('repr_freq', self.data)
        self.assertTrue(True)

    def test_set_values_to_pairs(self):
        eeg = EEGdata(Electrodes, Bands, self.pairs.electrode_pairs)
        eeg.set_values_to_pairs('repr_freq', self.data_corr)
        self.assertTrue(True)


class TestEEGdata1020(TestCase):

    def setUp(self) -> None:
        self.data = np.random.randn(19, 7)
        self.data_corr = np.random.randn(171, 7)

    def test_init(self):
        eeg = EEGdata1020()

        self.assertTrue(True)

