from paired_connectivity_analyzer import EEGPairedPermutationAnalyser, DrawEEG
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import cm
from metrics import dice, jaccard


def plot_reproducibility_by_frequency(size=70, band=1, num_reps=100, num_perms=1000, thres=0.05,
                                      factor=0.01, replace=False,
                                      cmap=cm.cool, figsize=(18, 4), is_param=True):
    """

    :param is_param:
    :param idxs:
    :param size:
    :param band:
    :param num_perms:
    :param thres:
    :param title:
    :param cmap:
    :param figsize:
    :return:
    """
    df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
    analyzer = EEGPairedPermutationAnalyser(data_df=df, num_perm=num_perms, thres=thres)
    draw_obj = DrawEEG()
    pattern = []
    assert isinstance(band, (int, list)), "Band should be int or list of ints"
    if isinstance(band, int):

        dict_diffs = analyzer.compute_reproducible_pattern(size=size,
                                                                  num_reps=num_reps, factor=factor, band=band,
                                                                  replace=replace, is_param=is_param)
        suptitle = f"Reproducible pairs in \"open-close\" contrast for group size {size} \n Rythm: {draw_obj.bands[band]}, num_reps: {num_reps}, frequency threshold: {factor}"

        pos_dict = {}
        pos_ = np.array(dict_diffs["mean_diff"]) >= 0
        neg_dict = {}
        neg_ = np.array(dict_diffs["mean_diff"]) < 0
        fig1, axs1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=figsize)


        for key in dict_diffs.keys():
            pos_dict[key] = np.array(dict_diffs[key])[pos_]
            neg_dict[key] = np.array(dict_diffs[key])[neg_]
        draw_obj1 = draw_edges_by_dict(pos_dict, band, keys=["channels", "mean_diff", "frequency"],
                                      title="Positive connections", cmap=cmap, vmin=0, vmax=1, is_freq=True, ax=axs1, normalize_width=False)
        draw_obj2 = draw_edges_by_dict(neg_dict, band, keys=["channels", "mean_diff", "frequency"],
                                     title="Negative connections", cmap=cmap, vmin=0, vmax=1, is_freq=True, ax=axs2, normalize_width=False)
        #draw_obj1.fig.suptitle(suptitle, fontsize=14)
        #draw_obj1.fig.subplots_adjust(top=0.8)

    elif isinstance(band, list):
        draw_obj = DrawEEG()
        title_list = [f"{draw_obj.bands[b_ - 1]}" for b_ in band]
        fig1, axs1 = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)
        fig2, axs2 = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)

        for i, b in enumerate(band):

            dict_diffs = analyzer.compute_reproducible_pattern(size=size,
                                                                  num_reps=num_reps, factor=factor, band=b,
                                                                  replace=replace, is_param=is_param)
            pos_dict = {}
            pos_ = np.array(dict_diffs["mean_diff"]) >= 0
            neg_dict = {}
            neg_ = np.array(dict_diffs["mean_diff"]) < 0
            for key in dict_diffs.keys():
                pos_dict[key] = np.array(dict_diffs[key])[pos_]
                neg_dict[key] = np.array(dict_diffs[key])[neg_]
            draw_obj1 = draw_edges_by_dict(pos_dict, b, keys=["channels", "mean_diff", "frequency"], title=title_list[i], ax=axs1[i],
                                   vmin=0, vmax=1, is_freq=True, cmap=cm.YlOrRd, normalize_width=False)
            draw_obj2 = draw_edges_by_dict(neg_dict, b, keys=["channels", "mean_diff", "frequency"], title=title_list[i], ax=axs2[i],
                                   vmin=0, vmax=1, is_freq=True,  cmap=cm.Blues, normalize_width=False)

            pattern.append(dict_diffs)
        draw_obj1.fig.suptitle(
            f"Reproducible patterns in \"open-close\" contrast  for group size {size}\n num_reps: {num_reps}(positive)",
            fontsize=16)
        draw_obj1.fig.subplots_adjust(top=0.8)
        draw_obj2.fig.suptitle(
            f"Reproducible patterns in \"open-close\" contrast  for group size {size}\n num_reps: {num_reps}(negative)",
            fontsize=16)
        draw_obj2.fig.subplots_adjust(top=0.8)
    else:
        NotImplementedError("Band should be list or list of ints")

    plt.show()
    fig1.savefig(f'positive{size}_thr_{thres}_reps{num_reps}.png', dpi=300)
    fig2.savefig(f'negative{size}_thr_{thres}_reps{num_reps}.png', dpi=300)
    return pattern, (fig1, fig2)



def plot_compute_sign_differences(idxs=None, size=70, band=1, num_perms=100, thres=0.001,
                                  title=None, cmap=cm.cool, vmin=-1.5, vmax=1.5, figsize=(18, 4)):
    """

    :param thres:
    :param figsize:
    :param cmap:
    :param title:
    :param idxs: list of ints, indexes of choosen subgroup
    :param size: int, size of group if idxs not specified
    :param band: int or list of ints
    :param num_perms: int, number of permutations
    :return: plot with differences
    """
    df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
    analyzer = EEGPairedPermutationAnalyser(data_df=df, num_perm=num_perms, thres=thres)
    assert isinstance(band, (int, list)), "Band should be int or list of ints"
    if isinstance(band, int):
        dict_diffs = analyzer.compute_sign_differences(idxs=idxs, size=size,
                                                       band=band, num_perms=num_perms, thres=thres)
        draw_obj = draw_edges_by_dict(dict_diffs, band, keys=["chan_names", "chan_diffs", "chan_pvals"],
                                      title=title, cmap=cmap, vmin=vmin, vmax=vmax)

    elif isinstance(band, list):
        draw_obj = DrawEEG()
        title_list = [f"{draw_obj.bands[b_-1]}" for b_ in band]
        fig, axs = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)

        for i, b in enumerate(band):
            dict_diffs = analyzer.compute_sign_differences(idxs=idxs, size=size,
                                                           band=b, num_perms=num_perms, thres=thres)
            draw_obj = draw_edges_by_dict(dict_diffs, b, keys=["chan_names", "chan_diffs", "chan_pvals"],
                                          title=title_list[i], ax=axs[i], cmap=cmap)
        draw_obj.fig.suptitle(f"Significant synchronization differences for \"open-close\" condition \n", fontsize=14)
        draw_obj.fig.subplots_adjust(top=0.8)
    else:
        NotImplementedError("Band should be list or list of ints")

    plt.show()
    return draw_obj


# TODO draw edges be reproducible patterns dict

def plot_reproducibility_pattern(size=70, band=1, num_perms=100, num_reps=50, factor=0.4, replace=False,
                                 thres=0.001, cmap=cm.cool,
                                 normalize_width=True, vmin=-1.5, vmax=1.5, figsize=(18, 4), is_param=True):
    df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
    analyzer = EEGPairedPermutationAnalyser(data_df=df, num_perm=num_perms, thres=thres)
    draw_obj = DrawEEG()
    pattern = []

    assert isinstance(band, (int, list)), "Band should be int or list of ints"
    if isinstance(band, int):
        dict_reproducible = analyzer.compute_reproducible_pattern(size=size,
                                                                  num_reps=num_reps, factor=factor, band=band,
                                                                  replace=replace, is_param=is_param)
        title = f"Reproducible pairs in \"open-close\" contrast for group size {size} \n Rythm: {draw_obj.bands[band]}, num_reps: {num_reps}, frequency threshold: {factor}"

        draw_obj = draw_edges_by_dict(dict_reproducible, band, keys=["channels", "mean_diff", "frequency"],
                                      cmap=cm.PRGn, title=title, vmin=vmin, vmax=vmax, normalize_width=normalize_width)
        pattern.append(dict_reproducible)

    elif isinstance(band, list):
        draw_obj = DrawEEG()
        title_list = [f"Rhythm: {draw_obj.bands[b_-1]}" for b_ in band]
        fig, axs = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)

        for i, b in enumerate(band):
            dict_reproducible = analyzer.compute_reproducible_pattern(size=size,
                                                               num_reps=num_reps, factor=factor, band=b,
                                                                      replace=replace, is_param=is_param)
            draw_obj = draw_edges_by_dict(dict_reproducible, b, keys=["channels", "mean_diff", "frequency"],
                                      title=title_list[i], ax=axs[i], cmap=cmap, vmin=vmin, vmax=vmax)
            pattern.append(dict_reproducible)
        draw_obj.fig.suptitle(
            f"Reproducible patterns in \"open-close\" contrast  for group size {size}\n num_reps: {num_reps}, frequency threshold: {factor}",
            fontsize=16)
    else:
        NotImplementedError("Band should be list or list of ints")

    draw_obj.fig.subplots_adjust(top=0.8)
    plt.show()
    return draw_obj, pattern


def draw_edges_by_dict(dict_diffs, band,
                       keys=["chan_names", "chan_diffs", "chan_pvals"], is_freq=False,
                       title=None, vmin=-1.5, vmax=1.5, ax=None, cmap=cm.cool,
                       normalize_width=True):
    """

    :param is_freq: bool: keyword from dict_diffs
    :param cmap: obj, matplotlib colormap
    :param ax: obj, axis to draw
    :param title: str, title
    :param keys:  list of string, keys for pair_names, values_color and values_width
    :param dict_diffs: dict, dictionary with eeg channel parameters
    :return: obj, draw obj
    """
    draw_obj = DrawEEG()
    pair_names = dict_diffs[keys[0]]
    if is_freq:
        values_color = np.array(dict_diffs['frequency'])
        vmin = 0
        vmax = 1
    else:
        values_color = np.array(dict_diffs[keys[1]])
    if len(pair_names) > 0:
        if keys[2] == "chan_pvals":
            values_width = 1 - dict_diffs[keys[2]]
        else:
            values_width = dict_diffs[keys[2]]
    else:
        values_width = dict_diffs[keys[2]]
    if title is None:
        title = f"Significant differences \"open-close\" for {draw_obj.bands[band-1]}"
    draw_obj.draw_edges(pair_names=pair_names, values_color=values_color,
                        values_width=values_width, normalize_values=True, normalize_width=normalize_width, vmin=vmin, vmax=vmax,
                        title=title, ax=ax, cmap=cmap)
    divider = make_axes_locatable(draw_obj.ax)
    cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
    draw_obj.fig.add_axes(cax)

    if len(pair_names) > 0:

        #cvalues = sorted([min(values_color)-0.01]+list(values_color)+[max(values_color)+0.01])
        cvalues = list(np.linspace(vmin, vmax, 20))
        cbar = matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=cvalues,
                                            orientation="horizontal")
    else:
        cvalues = list(np.linspace(vmin, vmax, 20))
        cbar = matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap,  values=cvalues,
                                                orientation="horizontal")
    if is_freq:
        cbar.set_label("Level of reproducibility")
    else:
        cbar.set_label("Fisher's Z difference")
    return draw_obj
