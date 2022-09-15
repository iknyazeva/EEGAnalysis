from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics import dice, jaccard
from matplotlib import cm
import numpy as np
from collections import Counter


class EEGPairedPermutationAnalyser:
    """Class for paired permutation comparisons with the stability of recovered channels estimation.
    As input we need to have two sets of pairwise parameters (for example synchronization) for each participant. Further
    for each subgroup from the big group the reproducibility of the patterns is estimated

    """

    def __init__(self, data_df=None, num_perm=1000, thres=0.001):
        """

        :param data_df (pd.DataFrame): pandas data frame with predefind columns structure, where in each column there is channel pair and condition
        :param num_perm (int): number of permutation used for paired test for mean difference
        :param thres (float): threshold for p-values
        """
        self.df = data_df.copy()
        colnames = [el.strip() for el in self.df.columns]
        self.df.columns = colnames
        self.thres = thres
        self.channel_bivar = np.unique([el.split('_')[0] for el in list(self.df.columns)])
        self.num_perm = num_perm
        self.open_name = None
        self.close_name = None
        self.subgroup_ids = list(self.df.index)
        self.size = len(self.subgroup_ids)
        self.key_bands = {1: 'delta', 2: 'theta', 3: 'alpha1', 4: 'alpha2', 5: 'beta1', 6: 'beta2', 7: 'gamma'}

    def get_subgroup(self, size=70):
        """function return subgroup from the total group with unique ids
        """
        if size is None:
            self.subgroup_ids = list(self.df.index)
        else:
            self.subgroup_ids = list(np.random.choice(self.df.index, size=size, replace=False))

    def plot_chnl_perm_test(self, emp_stat=None, perm_stats=None, ch_idx=0, band=1):
        """ Function for plotting  permutational pairwise statistics for specific channel

        :param emp_stat: float if precomuted statistics
        :param perm_stats: np.ndarray of floats permutational statistics
        :param ch_idx: int, id of channel to test
        :param band: int, eeg band
        :return: None
        """

        if (emp_stat is None) or (perm_stats is None):
            (emp_stat, _), perm_stats, = self.perm_difference_paired(band=1)
        val = plt.hist(perm_stats[:, ch_idx], density=True)
        plt.vlines(emp_stat[ch_idx], 0, max(val[0]), 'red', label='Observed stat');
        plt.vlines(np.percentile(perm_stats[:, ch_idx], 2.5), 0, max(val[0]), 'black', label='2.5 alpha level');
        plt.vlines(np.percentile(perm_stats[:, ch_idx], 97.5), 0, max(val[0]), 'black', label='97.5 alpha level');
        plt.vlines(np.percentile(perm_stats[:, ch_idx], 95), 0, max(val[0]), 'black', label='95 alpha level');
        plt.title(f"Permutation difference for channels {self.channel_bivar[ch_idx]} for band {self.key_bands[band]} ")
        plt.legend()

    def bootstrap_significant_channels(self, band=1,
                             plot=True, num_btsp=1000, conf_levels=[5, 95]):

        (emp_mean_diffs, p_val), perm_mean_diffs = self.perm_difference_paired(band=band)

        sign_connections = np.where(p_val < self.thres)[0]
        df_sbgroup = self.df.loc[self.subgroup_ids]
        size = df_sbgroup.shape[0]

        bstp_emp_diffs = []

        for i in tqdm(range(num_btsp)):
            b_df = df_sbgroup.loc[list(np.random.choice(df_sbgroup.index, size=size, replace=True))]
            b_emp_diffs = b_df[self.open_name].values - b_df[self.close_name].values
            bstp_emp_diffs.append(b_emp_diffs.mean(axis=0))

        conf_ints = np.percentile(np.array(bstp_emp_diffs), conf_levels, axis=0)
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(16, 4))

            ax.plot(range(len(sign_connections)), emp_mean_diffs[sign_connections], '*');
            ax.vlines(range(len(sign_connections)), conf_ints[0, sign_connections], conf_ints[1, sign_connections],
                      colors='gray')
            ax.set_xticks(range(len(sign_connections)))
            ax.set_xticklabels(self.channel_bivar[sign_connections], rotation=90, fontsize=10);
            ax.set_title(f'Significant edges for band {self.key_bands[band]}')
        return dict(zip(self.channel_bivar[sign_connections],
                        [(emp_mean_diffs[sign_connections], p_val[sign_connections]), conf_ints[:, sign_connections]]))

    def perm_difference_paired(self, band=1):
        """ Basic function for estimation inter-channel difference in specific for two condition

        :param band: (int) integer code for band
        :return : (tuple) empirical difference from the data (vector (size num channels)), p_val (vector (size num channels))
        and vector of simulated differences (ndarray size = (repet_time, num_channels))
        """
        self.open_name = [f'{self.channel_bivar[i]}_{band}_fo' for i in range(len(self.channel_bivar))]
        self.close_name = [f'{self.channel_bivar[i]}_{band}_fz' for i in range(len(self.channel_bivar))]
        df_sbgroup = self.df.loc[self.subgroup_ids]

        emp_diffs = df_sbgroup[self.open_name].values - df_sbgroup[self.close_name].values
        emp_mean_diffs = emp_diffs.mean(axis=0)

        perm_mean_diffs = np.zeros((self.num_perm, emp_diffs.shape[1]))

        for k in np.arange(self.num_perm):
            perm_mean_diffs[k, :] = (np.random.choice([1, -1], emp_diffs.shape[0]).reshape(-1, 1) * emp_diffs).mean(
                axis=0)
        p_val = np.mean(np.abs(perm_mean_diffs) > np.abs(emp_mean_diffs), axis=0)
        return (emp_mean_diffs, p_val), perm_mean_diffs

    def test_reproducability(self, size=70, band=1, num_reps=100):
        """ Function returned significant channels in each repetion

        :param size: (int) size of subgroup
        :param band: (int) number of band
        :param num_reps:  (int) number of repetitions
        :return: (list of dict) with number if significant channels and statistical value
        """
        self.get_subgroup(size=None)
        sign_channels = []
        for i in tqdm(range(num_reps)):
            self.get_subgroup(size=size)
            (emp_stat, p_val), _ = self.perm_difference_paired(band=band)
            sign_channel_ids = np.where(p_val < self.thres)[0]
            if sign_channel_ids.size > 0:
                sign_channels_values = emp_stat[sign_channel_ids]
                sign_channels.append(dict(zip(sign_channel_ids, sign_channels_values)))
        return sign_channels



    def pairwise_set_comparisons(self, size=70, band=1, num_reps=10, func=dice, type_='neigh'):
        """ Pairwise comparison significant channels sets received for each independent group in every repetion

        :param size: int, size of subgroup for simulation
        :param band: int, code of band
        :param num_reps: int, number of repetitions
        :param func: callable, function to compute stats, dice of jaccard
        :param type_: what to compare, all to all or neighbours
        :return: tuple, list of dict with significant channels, list with metrics results, Counter with frequency of channel pairs
        """

        assert type_ in ['neigh', 'all'], "type_ variable should be 'neigh' or 'all'"
        metric_list = []
        sign_tested = self.test_reproducability(size=size, band=band, num_reps=num_reps)
        if len(sign_tested) == 0:
            return [], ([], [])
        chns = [list(els.keys()) for els in sign_tested]
        cnt = Counter(np.hstack(chns))
        if type_ == 'neigh':
            for i in range(len(sign_tested) - 1):
                metric_list.append(func(list(sign_tested[i].keys()), list(sign_tested[i + 1].keys())))
        elif type_ == "all":
            for i, list1 in enumerate(sign_tested):
                for j, list2 in enumerate(sign_tested):
                    if i != j:
                        metric_list.append(func(list(list1.keys()), list(list2.keys())))
        else:
            raise NotImplementedError("type_ could be neigh or all")
        return sign_tested, (metric_list, cnt)

    def compute_reproducible_pattern(self, size=70, num_reps=50, factor=0.4, band=1):

        """ Compute reproducible pattern, those channels survived with the most of the repetitions
        (experiment simulations)

        :param size:  int, group size
        :param num_reps:  int, number of repetitions or psevdo experiments
        :param factor:  float from 0 to 1,
        :param band:
        :return:
        """

        assert 0 < factor < 1, "Factor variable should be from 0 to 1"
        sign_tested = self.test_reproducability(size=size, band=band, num_reps=num_reps)
        cnt = Counter(np.hstack([list(els.keys()) for els in sign_tested]))
        most_frequent = {x: count for x, count in cnt.items() if count >= num_reps*factor}
        if len(most_frequent) == 0:
            return {"channels": [], "frequency": [], "mean_diff": []}
        most_frequent_channels = list(self.channel_bivar[list(most_frequent.keys())])
        most_frequent_freqs = np.array(list(most_frequent.values()))/num_reps
        most_frequent_values = []
        for chan in most_frequent_channels:
            chan_id = list(self.channel_bivar).index(chan)
            chan_values = []
            for chan_dict in sign_tested:
                if chan_id in chan_dict.keys():
                    chan_values.append(chan_dict[chan_id])
            most_frequent_values.append(np.mean(chan_values))

        chn_dict = {"channels": most_frequent_channels, "frequency": most_frequent_freqs, "mean_diff": most_frequent_values}
        return chn_dict



