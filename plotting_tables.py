import pandas as pd
import re
from typing import Union, Optional
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import cm
import numpy.typing as npt
from eeg_data_class import EEGdata1020


class DrawEEG1020:
    def __init__(self, img_source=None):

        self.eeg_obj = EEGdata1020()
        self.electrodes = self.eeg_obj.electrodes
        self.number_of_channles = len(self.electrodes)
        if img_source is None:
            self.img = img.imread('21ch_eeg.png')
        else:
            self.img = img.imread(img_source)
        self.el_centers_dict = {"Cz": (197, 181), "C3": (134, 181),
                                "C4": (261, 181), "T3": (70, 181), "T4": (324, 181),
                                "Fz": (197, 117), "F3": (146, 116), "F4": (250, 116),
                                "F7": (95, 107), "F8": (300, 107), "Fp1": (156, 61),
                                "Fp2": (239, 61), "O1": (157, 301),
                                "O2": (238, 301), "Pz": (197, 245),
                                "P3": (146, 245), "P4": (250, 245), "T5": (95, 255),
                                "T6": (300, 255)}
        self.bands = self.eeg_obj.bands
        self.ax = None

    def draw_edges(self, chan_pairs: list[(str, str)],
                   values_color: npt.NDArray[float] = None,
                   values_width: npt.NDArray[float] = None,
                   normalize_values=False,
                   normalize_width=False,
                   vmin=-1, vmax=1,
                   cmap=cm.jet,
                   title="Hey, hey!",
                   color_label = "effect_size",
                   ax=None):

        """ draw edges
        Args:
            pair_names (list of string): list of tuple of string in format '(cn1, cn2)' ('F3/C3')
            values_color (array of floats): value from 0 to 1
            values_width (array of floats): should be positive value near 1
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        else:
            self.ax = ax
            self.fig = ax.get_figure()
        if len(chan_pairs) == 0:
            self.ax.imshow(self.img);
            self.ax.set_title(title)
            self.ax.axis('off');
            #return


        #chan_pairs = [el.split('/') for el in list(pair_names)]
        if len(chan_pairs) > 0:
            if values_color is None:
                values_color = 0.9 * np.ones(len(chan_pairs))
            if values_width is None:
                values_width = 0.9 * np.ones(len(chan_pairs))
            if normalize_values:
                max_ = max(vmax, max(values_color))
                min_ = min(vmin, min(values_color))
                if max_ > min_:
                    values_color = (values_color - min_) / (max_ - min_)
            if normalize_width:
                abs_max_ = np.max(np.abs(values_width))
                values_width = np.abs(values_width) / abs_max_

            for idx in range(len(chan_pairs)):
                chan_pair = chan_pairs[idx]
                els = np.array([self.el_centers_dict[chan_pair[0]], self.el_centers_dict[chan_pair[1]]])
                col = cmap(values_color[idx])

                self.ax.plot(els[:, 0], els[:, 1], color=col, alpha=1, linewidth=values_width[idx]);
            self.ax.imshow(self.img);
            self.ax.set_title(title)
            self.ax.axis('off');
        divider = make_axes_locatable(self.ax)
        cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
        self.fig.add_axes(cax)

        if len(chan_pairs) > 0:

            # cvalues = sorted([min(values_color)-0.01]+list(values_color)+[max(values_color)+0.01])
            cvalues = list(np.linspace(vmin, vmax, 20))
            cbar = matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=cvalues,
                                                    orientation="horizontal")
        else:
            cvalues = list(np.linspace(vmin, vmax, 20))
            cbar = matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap, values=cvalues,
                                                    orientation="horizontal")

        cbar.set_label(color_label)


class DrawDfEEG:
    def __init__(self, df):
        self.df = df
        self.draw_obj = DrawEEG1020()

    def get_values_to_draw(self,
                           chan_col='chan_pair',
                           band_col='band',
                           band='alpha2',
                           sign='pos',
                           filter_by=None,
                           color_col='mean_eff_size',
                           width_col=None):

        assert sign in ['pos', 'neg', 'both'], 'sign should be pos, neg, both'
        if filter_by is not None:
            df = self.df[filter_by]
        else:
            df = self.df.copy()
        df_small = df[df[band_col] == band]
        if sign == 'pos':
            df_small = df_small[df_small[color_col] > 0]
        elif sign == 'neg':
            df_small = df_small[df_small[color_col] < 0]

        pair_names_list = df_small[chan_col].values
        pair_names = list(map(parse_chan_string, pair_names_list))
        values_color = df_small[color_col].values
        if width_col is None:
            values_width = None
        else:
            values_width = np.power(df_small[width_col].values+0.5, 2) #*np.exp(1)
        return pair_names, values_color, values_width

    def draw_edges(self,
                   band_col='band',
                   band: Union[str, tuple] = 'alpha1',
                   chan_col: str = 'chan_pair',
                   filter_by: Optional[str] = None,
                   color_col: str = None,
                   width_col: str = None,
                   sign: str = 'separate',
                   figsize: tuple[float, float] = (18, 4),
                   sample_size=177,
                   **kwargs
                   ):
        kwargs.setdefault('cmap', cm.seismic)
        kwargs.setdefault('vmin', -1)
        kwargs.setdefault('vmax', 1)
        kwargs.setdefault('normalize_values', True)
        kwargs.setdefault('normalize_width', False)
        kwargs.setdefault('color_label', 'effect_size')

        assert sign in ['separate', 'same'], "sign responsible for effect direction, could be separate or same"


        if isinstance(band, str):
            if sign == 'same':
                figs, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
                kwargs.setdefault('title', f'Significant channels for {band} rythm')

                pair_names, values_color, values_width \
                    = self.get_values_to_draw(chan_col=chan_col,
                                              band_col=band_col,
                                              band=band,
                                              sign='both',
                                              filter_by=filter_by,
                                              color_col=color_col,
                                              width_col=width_col)
                self.draw_obj.draw_edges(pair_names, values_color, values_width, **kwargs, ax=axs)
            elif sign == 'separate':
                figs, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
                effects = ['open > close', 'open < close']
                _sign = ['pos', 'neg']
                for i in range(2):
                    kwargs['title'] = f'Significant channels for {band} rythm\n {effects[i]} '

                    pair_names, values_color, values_width \
                        = self.get_values_to_draw(chan_col=chan_col,
                                                  band_col=band_col,
                                                  band=band,
                                                  sign=_sign[i],
                                                  filter_by=filter_by,
                                                  color_col=color_col,
                                                  width_col=width_col)
                    self.draw_obj.draw_edges(pair_names, values_color, values_width, **kwargs, ax=axs[i])

        elif isinstance(band, (list, tuple)):
            if sign == 'same':
                figs, axs = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)
                for i, b in enumerate(band):
                    kwargs['title'] = f' {b} rythm'

                    pair_names, values_color, values_width \
                        = self.get_values_to_draw(chan_col=chan_col,
                                                  band_col=band_col,
                                                  band=b,
                                                  sign='both',
                                                  filter_by=filter_by,
                                                  color_col=color_col,
                                                  width_col=width_col)
                    self.draw_obj.draw_edges(pair_names, values_color, values_width, **kwargs, ax=axs[i])
                figs.suptitle(f"Significant differences for Open/Close conditions", fontsize=16)
                figs.subplots_adjust(top=0.8)
            elif sign == 'separate':
                effects = ['Open > Close', 'Open < Close']
                fig1, axs1 = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)
                fig2, axs2 = plt.subplots(nrows=1, ncols=len(band), figsize=figsize)
                _sign = ['pos', 'neg']
                axs = [axs1, axs2]
                figs = [fig1, fig2]
                for i in range(2):
                    for j, b in enumerate(band):
                        kwargs['title'] = f'{b} rythm'

                        pair_names, values_color, values_width \
                            = self.get_values_to_draw(chan_col=chan_col,
                                                      band_col=band_col,
                                                      band=b,
                                                      sign=_sign[i],
                                                      filter_by=filter_by,
                                                      color_col=color_col,
                                                      width_col=width_col)
                        self.draw_obj.draw_edges(pair_names, values_color, values_width, **kwargs, ax=axs[i][j])
                    figs[i].suptitle(f"Significant differences {effects[i]}, sample size = {sample_size}, filtered with {filter_by}", fontsize=16)
                    figs[i].subplots_adjust(top=0.8)
        return figs



def parse_chan_string(chan_str: str):
    res = re.search(r"^\('(\w{2,})', '(\w{2,})'\)", chan_str)
    return res.group(1), res.group(2)


