from tables import EEGSynchronizationTable, PairedNonParametric, DataTable, Reproducibility
from eeg_data_class import PairsElectrodes, Bands, Electrodes, EEGdata1020
import pickle
import numpy as np
import pandas as pd


# Final results: table with non-parametric stats

def get_stat_data(path_to_data=None):
    if path_to_data is None:
        path_to_data = 'eeg_dataframe_nansfilled.csv'
    stable_fo = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_data, cond_prefix='fo')
    stable_fz = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_data, cond_prefix='fz')
    stat = PairedNonParametric(stable_fz, stable_fo)
    return stat



def full_table_stats(path_to_data,
                     correction='np',
                     per_num=100,
                     agg='wmean',
                     bs_num=100,
                     alpha=0.05):
    #corrections is 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh', 'fdr_by','fdr_tsbh'

    stat = get_stat_data(path_to_data)
    if correction == 'np':
        rejected, p_vals = stat.get_zero_eff_perm_non_parametric(num=per_num, alpha=alpha, agg=agg)
    else:
        rejected, p_vals = stat.get_zero_eff_parametric(correction=correction, alpha=alpha)
    eff_size = stat.get_eff_size_dist(bs_num=bs_num)
    eeg_res: EEGdata1020 = EEGdata1020()
    if correction=='np':
        method = f'{correction}_{agg}_{per_num}'
    else:
        method=correction

    eeg_res.set_values_to_pairs(f'p_vals_full', p_vals)
    eeg_res.set_values_to_pairs('eff_size_full_mean', eff_size[:, :, 0])
    eeg_res.set_values_to_pairs('eff_size_full_low', eff_size[:, :, 1])
    eeg_res.set_values_to_pairs('eff_size_full_upper', eff_size[:, :, 2])
    eeg_res.method = method
    eeg_res.eff_bs_num = bs_num
    return eeg_res

def dice_stat(path_to_data, per_num=5000, save_path = './repr_results'):
    stat = get_stat_data(path_to_data)
    rpr = Reproducibility(stat.stable_fz, stat.stable_fo)
    save_path = './repr_results'
    per_num = 5000
    for sample_size in [20, 25, 30, 40, 50, 60]:
        print(sample_size)
        dice_res = dict()
        for correction in ['uncorr', 'bonferroni', 'sidak', 'holm', 'fdr_bh', 'fdr_by', 'np']:
            dice_list = rpr.bootstrap_reproducibility(sample_size=sample_size, num=10, per_num=per_num,
                                                      correction=correction,
                                                      agg='max', alpha=0.05)

            dice_res[correction] = dice_list
        dice_list = rpr.bootstrap_reproducibility(sample_size=sample_size, num=10, per_num=per_num, correction='np',
                                                  agg='wmean', alpha=0.05)
        dice_res['np_wmean'] = dice_list
        with open(f'{save_path}/dice_{sample_size}_in_group.pkl',
                  'wb') as f:
            pickle.dump(dice_res, f, protocol=pickle.HIGHEST_PROTOCOL)
def create_df_from_eeg_res(eeg_res: EEGdata1020,
                           alpha=0.05,
                           save_path='./repr_results'):
    dfs = []
    for band in eeg_res.bands.get_values():
        df = pd.DataFrame(eeg_res.p_vals_full[:, band - 1], index=eeg_res.el_pairs_list,
                          columns=['p_val'])
        df['mean_eff_size'] = eeg_res.eff_size_full_mean[:, band - 1]
        df['abs_eff_size'] = np.abs(df['mean_eff_size'])
        df['low_eff_size'] = eeg_res.eff_size_full_low[:, band - 1]
        df['upper_eff_size'] = eeg_res.eff_size_full_upper[:, band - 1]
        df['band'] = eeg_res.bands.get_name_by_id(band)
        df = df[df['p_val'] <= alpha]
        dfs.append(df)
    final = pd.concat(dfs)
    final.sort_values(by='abs_eff_size', inplace=True, ascending=False)
    method = getattr(eeg_res, 'method', 'unc')
    if save_path:
        final.to_csv(f'{save_path}/stats_full_{method}_eff_bs_num_{eeg_res.eff_bs_num}.csv')
    return final


def subsample_table_stats(path_to_data=None,
                          sample_size=20,
                          num_samples=10,
                          correction='np',
                          per_num=100,
                          agg='wmean',
                          bs_num=100,
                          alpha=0.05
                          ):
    if path_to_data is None:
        path_to_data = 'eeg_dataframe_nansfilled.csv'

    eeg_res = full_table_stats(path_to_data,
                     correction=correction,
                     per_num=per_num,
                     agg=agg,
                     bs_num=bs_num,
                     alpha=alpha)

    stable_fo = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_data, cond_prefix='fo')
    stable_fz = EEGSynchronizationTable.read_from_eeg_dataframe(path_to_data, cond_prefix='fz')
    rpr = Reproducibility(stable_fz, stable_fo)

    if isinstance(sample_size, int):
        sample_size_list = [sample_size]
    elif isinstance(sample_size, list):
        sample_size_list = sample_size

    for sz in sample_size_list:
        subj_lists = stable_fo.get_subj_subsamples(sz,
                                                 type_subs='bs', num=num_samples, replace=False)

        p_vals_list = []
        eff_size_list = []
        for subjs in  subj_lists:
            stable1 = rpr.table1.get_subtable_by_subjs(subjs)
            stable2 = rpr.table2.get_subtable_by_subjs(subjs)
            p_vals, eff_size = rpr._compute_p_vals_eff_size(stable1,
                                                        stable2,
                                                        correction=correction,
                                                        bs_num=bs_num,
                                                        per_num=per_num,
                                                        agg=agg,
                                                        alpha=alpha)
            p_vals_list.append(p_vals)
            eff_size_list.append(eff_size[:,:,0])
        eeg_res.set_values_to_pairs(f'p_vals_mean_sz_{sz}', np.stack(p_vals_list).mean(axis=0))
        eeg_res.set_values_to_pairs(f'freq_repr_sz_{sz}', (np.stack(p_vals_list) < alpha).mean(axis=0))
        eeg_res.set_values_to_pairs(f'eff_size_sz_{sz}', np.stack(eff_size_list).mean(axis=0))

    eeg_res.num_samples = num_samples
    eeg_res.sample_sizes = sample_size_list
    return eeg_res


def create_df_from_eeg_res_with_sbsmpls(eeg_res: EEGdata1020,
                           alpha=0.05,
                           save_path='./repr_results'):
    dfs = []
    sample_size_list=eeg_res.sample_sizes
    for band in eeg_res.bands.get_values():
        df = pd.DataFrame(eeg_res.p_vals_full[:, band - 1], index=eeg_res.el_pairs_list,
                          columns=['p_vals_full'])
        df['mean_eff_size'] = eeg_res.eff_size_full_mean[:, band - 1]
        df['abs_eff_size'] = np.abs(df['mean_eff_size'])
        for sz in sample_size_list:
            df[f'p_vals_sz_{sz}'] = getattr(eeg_res, f'p_vals_mean_sz_{sz}')[:, band-1]
            df[f'eff_size_sz_{sz}'] = getattr(eeg_res, f'eff_size_sz_{sz}')[:, band - 1]
            df[f'freq_repr_sz_{sz}'] = getattr(eeg_res, f'freq_repr_sz_{sz}')[:, band - 1]

        df['band'] = eeg_res.bands.get_name_by_id(band)
        df = df[df['p_vals_full'] <= alpha]
        dfs.append(df)
    final = pd.concat(dfs)
    final.sort_values(by='abs_eff_size', inplace=True, ascending=False)
    method = getattr(eeg_res, 'method', 'unc')
    if save_path:
        final.to_csv(f'{save_path}/stats_full_with_sbs_{method}_num_smpls_{eeg_res.num_samples}_eff_bs_num_{eeg_res.eff_bs_num}.csv')
    return final

