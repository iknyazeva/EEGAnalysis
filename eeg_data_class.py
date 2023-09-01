from dataclasses import dataclass, field
from enum import Enum
from typing import List
from itertools import combinations


class Bands(Enum):
    delta = 1
    theta = 2
    alpha1 = 3
    alpha2 = 4
    beta1 = 5
    beta2 = 6
    gamma = 7


class Electrodes(Enum):
    Fp1 = 1
    Fp2 = 2
    F7 = 3
    F3 = 4
    Fz = 5
    F4 = 6
    F8 = 7
    T3 = 8
    T4 = 9
    T5 = 10
    T6 = 11
    O1 = 12
    O2 = 13
    C3 = 14
    Cz = 15
    C4 = 16
    P3 = 17
    Pz = 18
    P4 = 19

# @dataclass()
# class IHWdefaults:
#     delta: tuple = (('theta', 0.8),
#                       ('alpha1', 0.4),
#                       ('alpha2', 0.2),
#                       ('beta1', 0),
#                       ('beta2', 0),
#                       ('gamma', 0))
#     theta: tuple = (('delta', 0.8),
#                     ('alpha1', 0.8),
#                     ('alpha2', 0.4),
#                     ('beta1', 0.2),
#                     ('beta2', 0),
#                     ('gamma', 0))
#     alpha1: tuple = (('delta': 0.4,
#                              'theta': 0.8,
#                              'alpha2': 0.8,
#                              'beta1': 0.4,
#                              'beta2': 0.2,
#                              'gamma': 0))
#
#     alpha2:tuple = (('delta': 0.2,
#                              'theta': 0.4,
#                              'alpha1': 0.8,
#                              'beta1': 0.8,
#                              'beta2': 0.4,
#                              'gamma': 0.2))
#     beta1:tuple =  (('delta': 0,
#                             'theta': 0.2,
#                             'alpha1': 0.4,
#                             'alpha2': 0.8,
#                             'beta2': 0.8,
#                             'gamma': 0.4))
#     beta2:tuple =  (('delta': 0,
#                             'theta': 0,
#                             'alpha1': 0.2,
#                             'alpha2': 0.4,
#                             'beta1': 0.8,
#                             'gamma': 0.8))
#     gamma: tuple = (('delta', 0,)
#                      ('theta', 0),
#                      ('alpha1', 0),
#                      ('alpha2', 0.2),
#                      ('beta1', 0.4),
#                      ('beta2', 0.8))
#


class PairsElectrodes:
    def __init__(self, electrodes: Electrodes):
        self.electrodes = electrodes
        self.nearest = [('Fp1', 'Fp2'),
                        ('Fp1', 'Fz'),
                        ('Fp2', 'Fz'),
                        ('Fp1', 'F3'),
                        ('Fp1', 'F7'),
                        ('Fp2', 'F4'),
                        ('Fp2', 'F8'),
                        ('F7', 'T3'),
                        ('F7', 'C3'),
                        ('F7', 'F3'),
                        ('F3', 'C3'),
                        ('F3', 'Cz'),
                        ('F3', 'Fz'),
                        ('Fz', 'C3'),
                        ('Fz', 'C4'),
                        ('Fz', 'Cz'),
                        ('Fz', 'F4'),
                        ('F4', 'Cz'),
                        ('F4', 'C4'),
                        ('F4', 'T4'),
                        ('F4', 'F8'),
                        ('F8', 'T4'),
                        ('F8', 'C4'),
                        ('T3', 'T5'),
                        ('T3', 'C3'),
                        ('T3', 'P3'),
                        ('C3', 'P3'),
                        ('C3', 'Cz'),
                        ('C3', 'Pz'),
                        ('C3', 'T5'),
                        ]

    @property
    def electrode_pairs(self):
        els = list(map(lambda x: x.name, self.electrodes))
        return list(combinations(els, 2))


    def create_pairs_dict(self, pairs_list, filter_by = None):
        pairs_dict = dict()
        p_list = pairs_list.copy()
        els = list(map(lambda x: x.name, self.electrodes))
        if filter_by:
            for opt in filter_by:
                p_list = [pair for pair in p_list if opt in pair]
        for i, el1 in enumerate(els):
            el1_p_list = [pair for pair in p_list if el1 in pair]
            for el2 in els[i+1:]:
                pairs_dict[(el1, el2)] = [pair for pair in el1_p_list if el2 in pair]
        return pairs_dict



