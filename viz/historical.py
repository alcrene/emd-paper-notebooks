"""
These are plotting functions that we used at some point, but no longer.
We keep them in case since some elements become of use in the future.
"""

# from __future__ import annotations

# # %%
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import seaborn as sns
# import holoviews as hv

# # %%
# from dataclasses import dataclass
# from typing import Optional, Dict, List

# # %% editable=true raw_mimetype="" slideshow={"slide_type": ""} tags=["skip-execution"]
# from .config import config
# from . import utils

# # %%
# dash_patterns = ["dotted", "dashed", "solid"]

# %% [markdown]
# sanitize = hv.core.util.sanitize_identifier_fn

# %% [markdown]
# def plot_R_bars(df_emd_cond: pd.DataFrame, data_label: str,
#                    colors: list[str], size_dim: str, xformatter=None
#     ) -> hv.Overlay:
#     """
#     Create a plot of vertical marks, with each mark corresponding to a value
#     in `df_emd_cond`.
#     The span of marks corresponding to the same dataset size is show by a
#     horizontal bar above them. The values of dataset sizes should be given in
#     of the columns of the DataFrame `df_emd_cond`; `size_dim` indicates the
#     name of this column.

# %% [markdown]
#     `colors` must be a list at least as long as the number of rows in `df_emd_cond`.
#     `size_dim` must match the name of the index level in the DataFrame
#     used to indicate the dataset size.
#     """
#     size_labels = pd.Series(df_emd_cond.index.get_level_values(size_dim), index=df_emd_cond.index)
#     size_marker_heights = pd.Series((1.3 + np.arange(len(size_labels))*0.7)[::-1],  # [::-1] places markers for smaller data sets higher
#                                     index=df_emd_cond.index)
#     logL_dim = hv.Dimension("logL", label=data_label)
#     y_dim = hv.Dimension("y", label=" ", range=(-0.5, max(size_marker_heights)))
#     size_dim = hv.Dimension("data_size", label=size_dim)

# %% [markdown]
#     ## Construct the actual lines marking the log likelihood ##
#     vlines = [hv.Path([[(logL, -0.5), (logL, 1)]], kdims=[logL_dim, y_dim],
#                       group=size, label=model_lbl)
#                   .opts(color=c)
#                   .opts(line_dash=dash_pattern, backend="bokeh")
#                   .opts(linestyle=dash_pattern, backend="matplotlib")
#               for (size, dash_pattern) in zip(df_emd_cond.index, dash_patterns)
#               for (model_lbl, logL), c in zip(df_emd_cond.loc[size].items(), colors)]

# %% [markdown]
#     # Legend proxies (Required because Path elements are not included in the legend)
#     legend_proxies = [hv.Curve([(0,0)], label=f"model: {model_lbl}")
#                           .opts(color=c)
#                           .opts(linewidth=3, backend="matplotlib")
#                       for model_lbl, c in zip(df_emd_cond.columns, colors)]

# %% [markdown]
#     ## Construct the markers indicating data set sizes ##
#     # These are composed of a horizontal segment above the log L markers, and a label describing the data set size
#     logp_centers = (df_emd_cond.max(axis="columns") + df_emd_cond.min(axis="columns"))/2
#     df_size_labels = pd.DataFrame(
#         (logp_centers, size_marker_heights, size_labels),
#         index=["x", "y", size_dim.name]
#     ).T

# %% [markdown]
#     size_labels_labels = hv.Labels(df_size_labels, kdims=["x", "y"], vdims=size_dim)

# %% [markdown]
#     size_markers = hv.Segments(dict(logL=df_emd_cond.min(axis="columns"),
#                                     logL_right=df_emd_cond.max(axis="columns"),
#                                     y0=size_marker_heights.to_numpy(),
#                                     y1=size_marker_heights.to_numpy()),
#                                [logL_dim, "y0", "logL_right", "y1"])

# %% [markdown]
#     ## Assemble and configure options ##
#     ov = hv.Overlay([*vlines, *legend_proxies, size_markers, size_labels_labels])

# %% [markdown]
#     # For some reason, applying these opts separately is more reliable with matplotlib backend
#     size_markers.opts( 
#         hv.opts.Segments(color="black"),
#         hv.opts.Segments(line_width=1, backend="bokeh"),
#         hv.opts.Segments(linewidth=1, backend="matplotlib")
#     )
#     size_labels_labels.opts(
#         hv.opts.Labels(yoffset=0.2),
#         hv.opts.Labels(yoffset=0.2, text_font_size="8pt", text_align="center", backend="bokeh"),
#         hv.opts.Labels(yoffset=0.4, size=10, verticalalignment="top", backend="matplotlib")
#     )
#     ov.opts(
#         hv.opts.Path(yaxis="bare", show_frame=False),
#         # *(hv.opts.Path(f"{sanitize(size)}.{model_lbl}", color=c)
#         #   for model_lbl, c in zip(df_emd_cond.columns, colors.curve) for size in size_labels),
#         # *(hv.opts.Path(sanitize(size), line_dash=dashpattern, backend="bokeh")
#         #   for size, dashpattern in zip(
#         #       size_labels, dash_patterns)),
#         hv.opts.Path(line_width=3, backend="bokeh"),
#         hv.opts.Path(linewidth=3, backend="matplotlib"),
#         hv.opts.Overlay(yaxis="bare", show_frame=False, padding=0.175,
#                         show_legend=True, legend_position="bottom"),
#         hv.opts.Overlay(width=600, height=200,
#                         legend_spacing=30, backend="bokeh")
#     )
#     if xformatter:
#         ov.opts(xformatter=xformatter, backend="bokeh")
#     ov.opts(hooks=[no_spine_hook("left")], backend="matplotlib")

# %% [markdown]
#     return ov
