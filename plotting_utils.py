from paired_connectivity_analyzer import EEGPairedPermutationAnalyser, DrawEEG
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import cm
from metrics import dice, jaccard


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

def plot_reproducibility_pattern(size=70, band=1, num_perms=100, num_reps=50, factor=0.4,
                                 thres=0.001, cmap=cm.cool,  vmin=-1.5, vmax=1.5, figsize=(18, 4)):
    df = pd.read_csv('eeg_dataframe_nansfilled.csv', index_col=0)
    analyzer = EEGPairedPermutationAnalyser(data_df=df, num_perm=num_perms, thres=thres)
    draw_obj = DrawEEG()
    pattern = []

    assert isinstance(band, (int, list)), "Band should be int or list of ints"
    if isinstance(band, int):
        dict_reproducible = analyzer.compute_reproducible_pattern(size=size,
                                                                  num_reps=num_reps, factor=factor, band=band)
        title = f"Reproducible pairs in \"open-close\" contrast for group size {size} \n Rythm: {draw_obj.bands[band]}, num_reps: {num_reps}, frequency threshold: {factor}"

        draw_obj = draw_edges_by_dict(dict_reproducible, 1, keys=["channels", "mean_diff", "frequency"],
                                      cmap=cm.PRGn, title=title, vmin=vmin, vmax=vmax)
        pattern.append(dict_reproducible)

    elif isinstance(band, list):
        draw_obj = DrawEEG()
        title_list = [f"Rhythm: {draw_obj.bands[b_-1]}" for b_ in band]
        fig, axs = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)

        for i, b in enumerate(band):
            dict_reproducible = analyzer.compute_reproducible_pattern(size=size,
                                                               num_reps=num_reps, factor=factor, band=b)
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
                       keys=["chan_names", "chan_diffs", "chan_pvals"],
                       title=None, vmin=-1.5, vmax=1.5, ax=None, cmap=cm.cool):
    """

    :param cmap: obj, matplotlib colormap
    :param ax: obj, axis to draw
    :param title: str, title
    :param keys:  list of string, keys for pair_names, values_color and values_width
    :param dict_diffs: dict, dictionary with eeg channel parameters
    :return: obj, draw obj
    """
    draw_obj = DrawEEG()
    pair_names = dict_diffs[keys[0]]
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
                        values_width=values_width, normalize_values=True, vmin=vmin, vmax=vmax,
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
    cbar.set_label("Fisher's Z difference")
    return draw_obj
