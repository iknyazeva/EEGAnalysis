import numpy as np
from numba import jit


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
    ''' U - ground true, so power is how many elements from v in u, and fdr how many elements from v not in u'''
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
