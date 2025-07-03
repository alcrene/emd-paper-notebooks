# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (emdcmp-dev)
#     language: python
#     name: emdcmp-dev
# ---

# %% [markdown]
# This is a port of the more recent `viz` module in `emdcmp`, which was updated after most experiments were run.
#
# Because we changed the name of Bconf to Bepis, updating this project’s `emdcmp`
# dependency would either force us to re-run all experiments or do some
# shenanigans to reuse the already computed results.
#
# To avoid this, we reproduce the new `viz` module here so that notebook can use the new methods.

# %% editable=true slideshow={"slide_type": ""}
from __future__ import annotations

# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import holoviews as hv

# %%
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# %% editable=true raw_mimetype="" slideshow={"slide_type": ""} tags=["skip-execution"]
from emdcmp import config, utils

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
# from config import config
# import utils

# %%
dash_patterns = ["dotted", "dashed", "solid"]


# %% [markdown]
# # Plotting functions

# %%
@dataclass
class CalibrationPlotElements:
    """
    `bin_idcs`: Dictionary indicating which experiment index were assigned to
       each bin. Use in conjunction with the EpistemicDistribution iterator
       to reconstruct specific experiments.
    """
    calibration_curves: hv.HoloMap
    prohibited_areas   : hv.Area
    discouraged_areas : hv.Overlay
    bin_idcs: Dict[float,List[np.ndarray[int]]]
    Bemd_hist_data:  Dict[float,Tuple[np.ndarray[int], np.ndarray[float]]]
    Bepis_hist_data: Dict[float,Tuple[np.ndarray[int], np.ndarray[float]]]

    def __iter__(self):
        yield self.calibration_curves
        yield self.prohibited_areas
        yield self.discouraged_areas

    ## Plotting functions ##

    @property
    def opts(self):
        return (
            hv.opts.Curve(**config.viz.calibration_curves),
            hv.opts.Area("Overconfident_area", **config.viz.prohibited_area),
            hv.opts.Area("Undershoot_area", **config.viz.discouraged_area)
            )
    @property
    def scatter_opts(self):
        scatter_opts = [hv.opts.Curve(color="#888888"),
                        hv.opts.Scatter(color=config.viz.calibration_curves["color"])
                        ]
        if "matplotlib" in hv.Store.renderers:
            scatter_opts += [hv.opts.Curve(linestyle="dotted", linewidth=1, backend="matplotlib"),
                             hv.opts.Scatter(s=10, backend="matplotlib")]
        if "bokeh" in hv.Store.renderers:
            scatter_opts += [hv.opts.Curve(line_dash="dotted", line_width=1, backend="bokeh")]
        return scatter_opts
    @property
    def hist_opts(self):
        hist_opts = []
        if "matplotlib" in hv.Store.renderers:
            hist_opts.append(
                hv.opts.Histogram(backend="bokeh",
                    line_color=None, alpha=0.75,
                    color=config.viz.calibration_curves["color"])
                )
        if "bokeh" in hv.Store.renderers:
            hist_opts.append(
                hv.opts.Histogram(backend="matplotlib",
                    color="none", edgecolor="none", alpha=0.75,
                    facecolor=config.viz.calibration_curves["color"])
                )
        return hist_opts

    @property
    def lines(self) -> hv.HoloMap:
        """
        Plot the calibration curves as solid lines joining the (Bemd, Bepis) tuples.
        """
        # We use .clone() to prevent contamination with different view options
        # It must be applied to the Curves themselves; cloning their containing HoloMap is not sufficient
        return hv.HoloMap({c: self.prohibited_areas * self.discouraged_areas
                              * curve.clone()
                           for c, curve in self.calibration_curves.items()}
               ).opts(*self.opts)

    @property
    def overlayed_lines(self) -> hv.Overlay:
        """
        Plot the calibration curves as solid lines joining the (Bemd, Bepis) tuples.
        """
        # We use .clone() to prevent contamination with different view options
        # It must be applied to the Curves themselves; cloning their containing HoloMap is not sufficient
        return (self.prohibited_areas * self.discouraged_areas
                * hv.Overlay([curve.clone() for curve in self.calibration_curves])
               ).opts(*self.opts)

    @property
    def scatters(self) -> hv.HoloMap:
        """
        Plot the calibration (Bemd, Bepis) tuples as a scatter plot.
        Points are joined with grey dotted lines to allow to make it easier to
        see which points come from the same `c` value and what curve they form.

        When possible, this should be the preferred way of reporting calibration
        curves: by showing where the bins fall, it is much easier to identify
        an excess concentration of points on the edge.
        With many curves however, they are easier to differentiate with the
        solid `lines` format.
        """
        scatters = {c: curve.to.scatter() for c, curve in self.calibration_curves.items()}
        return hv.HoloMap({c: self.prohibited_areas * self.discouraged_areas
                              * self.calibration_curves[c].clone() * scatters[c]
                           for c in scatters},
                          kdims=["c"]
               ).opts(*self.opts, *self.scatter_opts)
    @property
    def overlayed_scatters(self) -> hv.Overlay:
        """
        Same as `scatters`, except all curve+scatter plots are overlayed
        into one figure.
        """
        scatters = {c: curve.to.scatter() for c, curve in self.calibration_curves.items()}
        return (self.prohibited_areas * self.discouraged_areas
                * hv.Overlay([curve.clone() for curve in self.calibration_curves.values()])
                * hv.Overlay(list(scatters.values()))
                ).opts(*self.opts, *self.scatter_opts)

    @property
    def Bemd_hists(self) -> hv.HoloMap:
        frames = {c: hv.Histogram(data, kdims=["Bemd"], label="Bemd")
                  for c, data in self.Bemd_hist_data.items()}
        return hv.HoloMap(frames, kdims=["c"], group="Bemd_hists").opts(*self.hist_opts)

    @property
    def Bepis_hists(self) -> hv.HoloMap:
        frames = {c: hv.Histogram(data, kdims=["Bepis"], label="Bepis")
                  for c, data in self.Bepis_hist_data.items()}
        return hv.HoloMap(frames, kdims=["c"], group="Bepis_hists").opts(*self.hist_opts)

    def _repr_mimebundle_(self, *args, **kwds):
        return self.scatters._repr_mimebundle_(*args, **kwds)


# %%
def calibration_bins(calib_results: CalibrateResult,
                     target_bin_size: Optional[int]=None):
    """Return the bin edges for the histograms produced by `calibration_plot`.
    
    .. Note:: These are generally *not* good bin edges for plotting a histogram
    of calibration results: by design, they will produce an almost
    flat histogram.
    """
    bin_edges = {}
    for c, data in calib_results.items():
        i = 0
        Bemd = np.sort(data["Bemd"])
        edges = [Bemd[0]]
        for w in utils.get_bin_sizes(len(Bemd), target_bin_size)[:-1]:
            i += w
            edges.append(Bemd[i:i+2].mean())
        edges.append(Bemd[-1])
        bin_edges[c] = edges
    return bin_edges


# %%
def calibration_hists(calib_results: CalibrateResult,
                      target_bin_size: Optional[int]=None
                    ) -> CalibrationPlotElements:
    """
    Convert Calibration Results into the (Bemd, Bepis) pairs needed for
    calibration plots.
    Recall that on any one experiment, Bepis is either True or False. So to
    estimate the probability P(E[R_A] < E[R_B] | Bemd), we histogram the data
    points into equal-sized bins according to Bemd, then average the value of
    Bepis within each bin. Representing each bin by its midpoint Bemd then
    produces the desired list of (Bemd, Bepis) pairs.
    Note that bins are equal in the number of experiments (and so have the
    equal statistical power), rather than equal in width. The resulting points
    are therefore not equally spaced along the Bemd axis, but will concentrate
    in locations where there are more data.

    When designing calibration experiments, it is important to ensure that there
    are ambiguous cases which probe the middle of the calibration plot – the part
    we actually care about. Otherwise we can end up with all points being
    concentrated in the top right and bottom left corners: while this shows
    a strong correlated, it does not actually tell us whether the probability
    assigned by Bemd is any good, because only clear-cut cases were considered.

    Parameters
    ----------
    calib_results: The calibration results to plot. The typical way to obtain
       these is to create and run `Calibrate` task:
       >>> task = emdcmp.tasks.Calibrate(...)
       >>> calib_results = task.unpack_results(task.run())
    target_bin_size: Each point on the calibration curve is an average over
       some number of calibration experiments; this parameter sets that number.
       (The actual number may vary a bit, if `target_bin_size` does not exactly
       divide the total number of samples.)
       Larger bin sizes result in fewer but more accurate curve points.
       The default is to aim for the largest bin size possible which results
       in 16 curve points, with some limits in case the number of results is
       very small or very large.

    Returns
    -------
    curve_data: {c: [(Bemd, Bepis), ...]}
        Dictionary where each entry is a list of data points defining a
        calibration curve for a different c value.
    experiment_idcs: {c: [[int,...], ...]}
        Lists of experiment indices used in each Bemd bin.
    """
    ## 
    curve_data = {}
    experiment_idcs = {}
    for c, data in calib_results.items():
        # # We don’t do the following because it uses the Bepis data to break ties.
        # # If there are a lot equal values (typically happens with a too small c),
        # # then those will get sorted and we get an artificial jump from 0 to 1
        # data.sort(order="Bemd")
        Bemd = data["Bemd"] if "Bemd" in data.dtype.names else data["BQ"]
        σ = np.argsort(Bemd)  # This will only use Bemd data; order within ties remains random
        Bemd = Bemd[σ]
        Bepis = data["Bepis"][σ] if "Bepis" in data.dtype.names else data["Bconf"][σ]

        curve_points = []
        bin_idcs = []
        i = 0
        for w in utils.get_bin_sizes(len(data), target_bin_size):
            curve_points.append((Bemd[i:i+w].mean(),
                                 Bepis[i:i+w].mean()))
            bin_idcs.append(σ[i:i+w])
            i += w
        curve_data[c] = curve_points
        experiment_idcs[c] = bin_idcs

    return curve_data, experiment_idcs

# %%
def calibration_plot(calib_results: CalibrateResult,
                     target_bin_size: Optional[int]=None
                    ) -> CalibrationPlotElements:
    """Create a calibration plot from the results of calibration experiments.
    Calls `calibration_hists` to compute the plot data

    Parameters
    ----------
    calib_results: The calibration results to plot. The typical way to obtain
       these is to create and run `Calibrate` task:
       >>> task = emdcmp.tasks.Calibrate(...)
       >>> calib_results = task.unpack_results(task.run())
    target_bin_size: Each point on the calibration curve is an average over
       some number of calibration experiments; this parameter sets that number.
       (The actual number may vary a bit, if `target_bin_size` does not exactly
       divide the total number of samples.)
       Larger bin sizes result in fewer but more accurate curve points.
       The default is to aim for the largest bin size possible which results
       in 16 curve points, with some limits in case the number of results is
       very small or very large.

    See also
    --------
    - `calibration_hists`
    """

    ## Calibration curves ##
    curve_data, experiment_idcs = calibration_hists(calib_results, target_bin_size)

    calib_curves = {}
    for c, points in curve_data.items():
        curve = hv.Curve(points, kdims="Bemd", vdims="Bepis", label=f"{c=}")
        curve = curve.redim.range(Bemd=(0,1), Bepis=(0,1))
        # curve.opts(hv.opts.Curve(**config.viz.calibration_curves))
        calib_curves[c] = curve
    calib_hmap = hv.HoloMap(calib_curves, kdims=["c"])

    ## Precompute histograms, so that we can have .*_hist methods to CalibratePlotElements ##
    Bemd_hists = {}
    Bepis_hists = {}
    for c, res in calib_results.items():
        res_Bemd = res["Bemd"] if "Bemd" in res.dtype.names else res["BQ"]
        res_Bepis = res["Bepis"] if "Bepis" in res.dtype.names else res["Bconf"]
        Bemd_hists[c]  = np.histogram(res_Bemd,              bins="auto", density=True)
        Bepis_hists[c] = np.histogram(res_Bepis.astype(int), bins="auto", density=True)

    ## Prohibited & discouraged areas ##
    # Prohibited area
    prohibited_areas = hv.Area([(x, x, 1-x) for x in np.linspace(0, 1, 32)],
                              kdims=["Bemd"], vdims=["Bepis", "Bepis2"],
                              group="overconfident area")

    # Discouraged areas
    discouraged_area_1 = hv.Area([(x, 1-x, 1) for x in np.linspace(0, 0.5, 16)],
                         kdims=["Bemd"], vdims=["Bepis", "Bepis2"],
                         group="undershoot area")
    discouraged_area_2 = hv.Area([(x, 0, 1-x) for x in np.linspace(0.5, 1, 16)],
                         kdims=["Bemd"], vdims=["Bepis", "Bepis2"],
                         group="undershoot area")

    prohibited_areas = prohibited_areas.redim.range(Bemd=(0,1), Bepis=(0,1))
    discouraged_area_1 = discouraged_area_1.redim.range(Bemd=(0,1), Bepis=(0,1))
    discouraged_area_2 = discouraged_area_2.redim.range(Bemd=(0,1), Bepis=(0,1))

    # prohibited_areas.opts(hv.opts.Area(**config.viz.prohibited_area))
    # discouraged_area_1.opts(hv.opts.Area(**config.viz.discouraged_area))
    # discouraged_area_2.opts(hv.opts.Area(**config.viz.discouraged_area))

    ## Combine & return ##
    return CalibrationPlotElements(
        calib_hmap, prohibited_areas, discouraged_area_1*discouraged_area_2,
        experiment_idcs, Bemd_hists, Bepis_hists)

