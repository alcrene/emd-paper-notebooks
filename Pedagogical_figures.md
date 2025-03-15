---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python (emd-paper)
  language: python
  name: emd-paper
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Pedagogical figures

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import Ex_UV
import Ex_Prinz2004
import emdcmp as emd
import matplotlib as mpl
import numpy as np
import holoviews as hv
from math import sqrt, log2
from dataclasses import dataclass, replace
from scipy import stats
from more_itertools import nth
from num2words import num2words
#from myst_nb import glue
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Project imports

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import utils
import viz
from config import config
from viz import glue
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Configuration

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
hv.extension("matplotlib", "bokeh")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
dims = viz.dims.matplotlib
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
viz.save.update_figure_files = True
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
sanitize = hv.core.util.sanitize_identifier_fn
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@dataclass
class colors(viz.ColorScheme):
    mixed : str = config.figures.colors["bright"].green
    synth : str = config.figures.colors["bright"].purple
    δemd  : str = config.figures.colors["pale"].yellow
    qhat  : str = config.figures.colors["bright"].grey
    data  : str = config.figures.colors["bright"].grey
    interp_ppf: str = config.figures.colors["light"]["light cyan"]
    marginal  : str = config.figures.colors["light"]["light cyan"]
    prestep   : str = config.figures.colors["light"].orange
    poststep  : str = config.figures.colors["light"]["light blue"]
    refine_lines: hv.Cycle = hv.Cycle([hv.Cycle("YlOrBr").values[i] for i in (5, 4, 2, 0)])
    #refine_points: str = "#666666"
    refine_points: hv.Cycle = hv.Cycle(["#444444", "#444444", "#444444", "#CCCCCC"])
@dataclass
class color_labels:
    mixed : str = "green"
    synth : str = "red"
    δemd  : str = "yellow"
    qhat  : str = "grey"
    data  : str = "data"
    interp_ppf: str = "cyan"
    marginal: str = "cyan"
    prestep : str = "orange"
    poststep: str = "blue"
colors
```

```{code-cell} ipython3
mpl.style.use(["config/base.mplstyle", "config/publish.mplstyle"])
#mpl.rcParams["font.size"] = 11.0  # Default was 9.
mpl.rcParams["axes.labelsize"] = 9.0
mpl.rcParams["axes.titlesize"] = 10.  # Base font size is 9
#mpl.rcParams["font.size"] = 9.0
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_pedag-path-sampling)=
## $δ^{\mathrm{EMD}}$ and path sampling

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
datasets = Ex_UV.panel_param_iterator(Ex_UV.L_med, "PPF sampling illustration", progbar=False)

D = nth(datasets, 5)
mixed_ppfs, synth_ppfs = Ex_UV.get_ppfs(D)

#phys_model = "Rayleigh-Jeans"
phys_model = "Planck"
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
c=Ex_UV.c_chosen
#c = .25
M_marginals = 10_000
```

We plot distributions at a particular refinement step $n$,
where the intermediate points $Φ$ which are shared by an $(x_1, x_2)$:

```{code-cell} ipython3
n = 4
ΔΦ = 2**-n
Φarr = ((np.arange(2**(n-1))*2+1)*ΔΦ)
Φplot = ((np.arange(2**(n-1))*2+1)*ΔΦ)[[0, 3, 6]]
print(f"All split Φ at refinement step {n}: {Φarr}")
print(f"Those we plot:                    {Φplot}")
```

```{code-cell} ipython3
mixed_ppf = mixed_ppfs[phys_model]
synth_ppf = synth_ppfs[phys_model]
def δemd(Φarr): return abs(synth_ppf(Φarr) - mixed_ppf(Φarr))
```

```{code-cell} ipython3
Φarr = np.linspace(0, 1, 192*8)  # *8 is to resolve the curve in the [0.97, 1] zoom
mixed_curve = hv.Curve(zip(Φarr, mixed_ppf(Φarr)),
                       kdims=[dims.Φ], vdims=[dims.q], label=r"mixed PPF ($q^*$)")
synth_curve = hv.Curve(zip(Φarr, synth_ppf(Φarr)),
                       kdims=[dims.Φ], vdims=[dims.q], label=r"synth PPF ($\tilde{q}$)")
area = hv.Area((Φarr, mixed_ppf(Φarr) - sqrt(c)*δemd(Φarr), mixed_ppf(Φarr) + sqrt(c)*δemd(Φarr)),
               kdims=[dims.Φ], vdims=[dims.q, "q2"], label="δemd")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} Higher resolution
:class: note margin
We normally don’t use a `res` greater than 8, because the increased precision in the integral decreases sharply beyond that point (and the computational cost *increases* sharply). However for purpose of illustration, since we zoom into the region $Φ \in [0.97, 1]$, we use `res=10` to better resolve the sampled paths.
:::

:::{admonition} Selection of RNG seed
For pedagogical reasons, we want to show only a handful of PPF traces. Unfortunately, there is quite a lot of variability between traces, and at least our initial draw (with keys `"pedag", "qpaths", 1` gave the wrong impression that all draws were biased below $q^*$, the ostensible mean.
We tried four seeds (incrementing 1 to 4) until we found one which returns a set of six PPFs which we feel represents the ensemble well.

Drawing a larger number (e.g. 40) suggests that the distribution is skewed, with excursions above $q^*$ occurring slightly less frequently but being more pronounced.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
rng = utils.get_rng("pedag", "qpaths", 4)
qpaths = emd.path_sampling.generate_quantile_paths(mixed_ppf, δemd, c=c, M=6, res=10, rng=rng)
qhat_curves = [hv.Curve(zip(Φhat, qhat), label=r"PPF realization ($\hat{q}$)",
                        kdims=[dims.Φ], vdims=[dims.q])
               .opts(color=colors.lighten(rng.uniform(-0.2, +0.1)).qhat)
               .opts(color=colors.lighten(rng.uniform(-0.2, +0.1)).qhat, backend="bokeh")
               for Φhat, qhat in qpaths]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig_with_qpaths = area * hv.Overlay(qhat_curves) * mixed_curve * synth_curve

ticks = dict(yticks=[-7, -6, -5, -4, -3, -2, -1],
             yformatter=lambda y: str(y) if y in {-7, -1} else "")
fig_with_qpaths.opts(
    hv.opts.Curve(**ticks), hv.opts.Curve(**ticks, backend="bokeh"),
    hv.opts.Area(**ticks), hv.opts.Area(**ticks, backend="bokeh"),
    hv.opts.Overlay(legend_opts={"loc": "upper left", "bbox_to_anchor":(0.05, 0.85)}),
    hv.opts.Overlay(legend_position="top_left", backend="bokeh"),
    hv.opts.Curve(f"Curve.{sanitize(mixed_curve.label)}", color=colors.mixed),
    hv.opts.Curve(f"Curve.{sanitize(mixed_curve.label)}", color=colors.mixed, backend="bokeh"),
    hv.opts.Curve(f"Curve.{sanitize(synth_curve.label)}", color=colors.synth),
    hv.opts.Curve(f"Curve.{sanitize(synth_curve.label)}", color=colors.synth, backend="bokeh"),
    hv.opts.Area(facecolor=colors.δemd, edgecolor="none", color="none", backend="matplotlib"),
    hv.opts.Area(fill_color=colors.δemd, line_color=None, backend="bokeh")
)
fig_with_qpaths = fig_with_qpaths.redim.range(q=(-7,-1))
```

```{code-cell} ipython3
fig_with_qpaths.redim.range(q=(-7, -6))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, hide-input]
---
# Remove text descriptions for the published plot
fig = fig_with_qpaths
fig_with_qpaths = fig.redim(Φ=hv.Dimension('Φ', label='$\\Phi$', range=fig.get_dimension("Φ").range),
                            q=hv.Dimension('q', label='$q$', range=fig.get_dimension("q").range))


# Zoom near 0
ticks = dict(#yticks=[-6.9, -6.8, -6.7, -6.6, -6.5],
             #xticks=[0, 0.175, 0.35, 0.525, 0.7],
             yticks=[-6.8, -6.6, -6.4, -6.2, -6.],
             xticks=[0, 0.2, 0.4, 0.6, 0.8],
             yformatter = lambda y: str(y) if y in {-6.8, -6.} else "",
             xformatter = lambda x: str(x) if x in {0, 0.8} else "")
#hooks=[viz.despine_hook, viz.xlabel_shift_hook(), viz.ylabel_shift_hook(2)]
hooks=[viz.xlabel_shift_hook(), viz.ylabel_shift_hook(1.75)]
#zoom_initial = fig_with_qpaths.clone().redim.range(q=(-6.9,-6.45), Φ=(0, 0.72))
zoom_initial = fig_with_qpaths.clone().redim.range(q=(-6.9,-6), Φ=(0, 0.85))
zoom_initial.opts(hv.opts.Curve(**ticks, hooks=hooks, xlabel="", backend="matplotlib"),
                  hv.opts.Area (**ticks, hooks=hooks),
                  hv.opts.Overlay(show_legend=False), hv.opts.Overlay(show_legend=False, backend="bokeh"))

# Zoom near 1
ticks = dict(yticks=[-5, 0, 5, 10, 15, 20, 25], xticks=[0.97, 0.98, 0.99, 1],
             yformatter = lambda y: str(y) if y in {-5, 25} else "",
             xformatter = lambda x: str(x) if x in {0.97, 1} else "")
hooks=[viz.xlabel_shift_hook(), viz.ylabel_shift_hook(1)]
zoom_final = fig_with_qpaths.clone().redim.range(q=(-5, 25), Φ=(0.97, 1))
zoom_final.opts(hv.opts.Curve(**ticks, hooks=hooks, backend="matplotlib"),
                hv.opts.Area (**ticks, hooks=hooks),
                hv.opts.Overlay(show_legend=False), hv.opts.Overlay(show_legend=False, backend="bokeh"))

# Rectangles showing zooms in first panel
def get_rect_points(panel):
    x0, x1 = panel.get_dimension("Φ").range
    y0, y1 = panel.get_dimension("q").range
    return (x0, y0, x1, y1)
zoom_rects = hv.Rectangles([get_rect_points(zoom_initial), get_rect_points(zoom_final)],
                           kdims=[fig.get_dimension("Φ"), fig.get_dimension("q"), "Φ2", "q2"])

# Lines showing where we take the statistical slices
marginal_markers = hv.VLines(Φplot, label="marginals")

ticks = dict(xticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
             xformatter = lambda x: str(x) if x in {0, 1} else "")
hooks=[viz.xlabel_shift_hook(), viz.ylabel_shift_hook(.5)]
panelA = fig_with_qpaths.redim.range(q=(-7.2, -0.8), Φ=(-0.025,1.025))*zoom_rects
panelA.opts(hv.opts.Curve(**ticks, hooks=hooks, xlabel="", backend="matplotlib"),
            hv.opts.Area(**ticks, hooks=hooks),
            hv.opts.Overlay(legend_opts={"loc": "upper left",
                                         "bbox_to_anchor": (0.0, 0.9),
                                         "labelspacing": .5,
                                         "borderpad": 0.1,
                                         "fontsize": 8})
           )

panelB = zoom_initial*marginal_markers
panelB.opts(show_legend=False)

layout = panelA + panelB + zoom_final
# Set plot options
layout.opts(hv.opts.Layout(shared_axes=False, fig_inches=2.2, hspace=0.3,
                           sublabel_format="", sublabel_position=(0.4, 0.85), fontscale=1.3),
            hv.opts.Curve(fontscale=1.3),
            hv.opts.Area(fontscale=1.3),
            hv.opts.Rectangles(color="none", facecolor="none", edgecolor="#222222"),
            hv.opts.Rectangles(fill_color="none", line_color="#888888", backend="bokeh"),
            hv.opts.VLines(color=colors.lighten(.1).marginal, linewidth=2),
            hv.opts.Overlay(aspect=1.6)
            #hv.opts.Curve(f"Curve.{sanitize(mixed_curve.label)}", color=colors.mixed),
            #hv.opts.Curve(f"Curve.{sanitize(mixed_curve.label)}", color=colors.mixed, backend="bokeh"),
            #hv.opts.Curve(f"Curve.{sanitize(synth_curve.label)}", color=colors.synth),
            #hv.opts.Curve(f"Curve.{sanitize(synth_curve.label)}", color=colors.synth, backend="bokeh"),
            #hv.opts.Area(facecolor=colors.δemd, edgecolor="none", color="none", backend="matplotlib"),
            #hv.opts.Area(fill_color=colors.δemd, line_color=None, backend="bokeh")
        )
layout.cols(1)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
viz.save(layout.cols(1), config.paths.figures/"pedag_qhat-paths_raw.svg")
#viz.save(layout.opts(clone=True, fig_inches=4).cols(3), config.paths.figures/"pedag_qhat-paths.svg")
#viz.save(layout.cols(3), config.paths.figures/"pedag_qhat-paths.pdf")
```

Finish in Inkscape:
- Remove the bits of axis in the first panel which go beyond [0,1].
- Widen the zoom boxes in panel A so they are easier to see.
- Make the sublabels bold as in other plots.
- Save PDF.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

Older column version for PDF layout:
```python
# For the pdf figure we compact it a bit further since paper space is precious
layout.cols(1).opts(
    hv.opts.Overlay(aspect=2.7),
    hv.opts.Layout(sublabel_position=(0.4, 0.8))
)
hooks = [viz.despine_hook, viz.ylabel_shift_hook(.9, .75)]
layout[0].opts(xlabel="").opts(hv.opts.Curve(hooks=hooks), hv.opts.Area(hooks=hooks))
hooks = [viz.despine_hook, viz.ylabel_shift_hook(1.7, .5)]
layout[1].opts(xlabel="").opts(hv.opts.Curve(hooks=hooks), hv.opts.Area(hooks=hooks))
layout

viz.save(layout.cols(1), config.paths.figures/"pedag_qhat-paths.pdf")
```

+++

### Visual evidence that we match the target statistics

```{code-cell} ipython3
import pandas as pd
```

```{code-cell} ipython3
rng = utils.get_rng("pedag", "qpaths", 4)
qpaths = emd.path_sampling.generate_quantile_paths(mixed_ppf, δemd, c=c, M=M_marginals, res=4, rng=rng)
```

```{code-cell} ipython3
series = [pd.Series(qhat[1], name=f"q{i+1}", index=pd.Index(qhat[0], name="Φ"))
          for i, qhat in enumerate(qpaths)]
```

```{code-cell} ipython3
df = pd.DataFrame(series)
df.index.name = "qi"
```

NB: Pandas’ `diff` aligns results with the rightmost value, while in our theory we used the leftmost.
Therefore below we drop the first column of data and the last entry of `columns`, thus shifting the $Φ$ index back by one.

```{code-cell} ipython3
Δdf = df.diff(axis="columns").iloc[:,1:]
Δdf.columns = df.columns[0:-1]
Δdf;
```

```{code-cell} ipython3
assert n == int(log2(len(Δdf.columns))), "`n` should be constant throughout the notebook"
```

```{code-cell} ipython3
def get_text_pos(hists, xfrac=0.85, yfrac=0.9):
    xmax = max(hist.data[hist.kdims[0].name].max() for hist in hists)
    xmin = min(hist.data[hist.kdims[0].name].min() for hist in hists)
    ymax = max(hist.data[hist.vdims[0].name].max() for hist in hists)
    ymin = min(hist.data[hist.vdims[0].name].min() for hist in hists)
    Δx = xmax - xmin
    Δy = ymax - ymin
    return xmin + xfrac*Δx, ymin + yfrac*Δy
```

:::{important}
I don’t use a `Distribution` below because I actually find it more difficult to interpret.
With an `Area`, I can more easily distinguish sampling noise from actual features of the distribution.

While it is true that with 10,000 samples, the KDE used by `Distribution` does not distort the distribution much,
with so many samples we also can resolve finer details – details which are smudged by the KDE.
:::

```{code-cell} ipython3
hists = {Φ: hv.Histogram(np.histogram(df[Φ], bins='auto', density=True), kdims=[dims.q], label=rf"$q_«2^«-{n}»» = {i}\cdot 2^«{n}»$".replace('«','{').replace('»','}'))
         for i, Φ in enumerate(Δdf.columns)}
# If we have a lot of samples, the histograms are effectively distributions. Plotting them directly is OK, but the large number of bars
# is not friendly to either SVG or PDF, often producing artifacts.
# Better to convert them to solid areas, which won’t produce such artifacts.
hists = {Φ: hist.to.area() for Φ, hist in hists.items()}
# Remove text description for publication
dimq = hv.Dimension("q", label=r"$q$")
hists = {Φ: hist.redim(q=dimq)
         for Φ, hist in hists.items()}
layout_marginals = hv.Layout([(hv.VSpan(EΦ-cδ, EΦ+cδ) * hists[Φ] * hv.VLine(EΦ)
                               * hv.Text(*get_text_pos((hists[Φ],), 1, 0.95),
                                         rf"$\Phi = {int(Φ/2**-n)}"+rf"\cdot 2^{{{{-{n}}}}}$",
                                         halign="right", fontsize=10)
                              ).redim(x=dimq)
                              for Φ, EΦ, cδ in zip(Φplot, mixed_ppf(Φplot), np.sqrt(c)*δemd(Φplot))])
                              #for Φ in (np.arange(2**(n-1))*2+1)*ΔΦ])
                             #for Φ in [0.25, 0.5, 0.75, 0.9375]])
layout_marginals.opts(
    hv.opts.Layout(fig_inches=2, shared_axes=False, sublabel_format="", vspace=.2),
    hv.opts.Overlay(show_title=False, fontscale=1.3, show_legend=False),
    hv.opts.VLine(color=colors.mixed, linewidth=2),
    hv.opts.VSpan(edgecolor=None, facecolor=colors.δemd),
    hv.opts.Histogram(edgecolor=None, facecolor=colors.marginal,
                      ylabel="Density", yaxis=None),
    hv.opts.Area(facecolor=colors.marginal, edgecolor=colors.marginal, color="none", linewidth=2, backend="matplotlib",
                 alpha=0.7, ylabel="Density", yaxis=None)
    #hv.opts.Histogram("Histogram.x1", alpha=0.8, facecolor=colors.prestep),
    #hv.opts.Histogram("Histogram.x2", alpha=0.5, facecolor=colors.poststep),
)

layout_marginals[0].opts(xlim=(-6.8878, -6.883), xticks=[-6.887, -6.886, -6.885, -6.884], xlabel="")
layout_marginals[0].opts(hv.opts.VSpan(xformatter=lambda x: f"${x}$" if x in {-6.887, -6.884} else ""))
layout_marginals[1].opts(xlim=(-6.91, -6.64), xticks=[-6.9, -6.85, -6.8, -6.75, -6.7], xlabel="")
layout_marginals[1].opts(hv.opts.VSpan(xformatter=lambda x: f"${x:.2f}$" if x in {-6.9, -6.7} else ""))
layout_marginals[2].opts(xlim=(-7.25, -4.35), xticks=[-7, -6.5, -6.0, -5.5, -5.0, -4.5])
layout_marginals[2].opts(hv.opts.VSpan(xformatter=lambda x: f"${x}$" if x in {-7.0, -4.5} else ""))
layout_marginals.cols(1)
```

```{code-cell} ipython3
viz.save(layout_marginals.cols(1), config.paths.figures/"pedag_qhat-marginals_raw.svg")
```

:::{admonition} Some issues with the code below
- `.relabel` does not work when we convert Histograms to Areas.
- Even if it did work, overlays don’t include Areas in their legend.
- So for the final plot, we produce two versions of this, with Histogram and Area,
  and combine the legend of the former with the distributions of the latter.
- The non-working labels also means that we need to change the selectors, which become the default names `X_1` and `X_2`.
:::

```{code-cell} ipython3
use_area = True
hists = {Φ: hv.Histogram(np.histogram(Δdf[Φ], bins='auto', density=True), kdims=[dims.Δq], label=rf"$\Delta q_«2^«-{n}»» = {i}\cdot 2^«{n}»$".replace('«','{').replace('»','}'))
         for i, Φ in enumerate(Δdf.columns)}
# If we have a lot of samples, the histograms are effectively distributions. Plotting them directly is OK, but the large number of bars
# is not friendly to either SVG or PDF, often producing artifacts.
# Better to convert them to solid areas, which won’t produce such artifacts.
if use_area:
    hists = {Φ: hist.to.area() for Φ, hist in hists.items()}
# Remove text description for publication
hists = {Φ: hist.redim(Δq=hv.Dimension("Δq", label=r"$\Delta q$"))
         for Φ, hist in hists.items()}
layout_incr_marginals = hv.Layout([hists[Φ-ΔΦ].relabel(label="$x_1$") * hists[Φ].relabel(label="$x_2$")   # Relabel does not work when we do Hist->Area
                                   * hv.Text(*get_text_pos((hists[Φ-ΔΦ], hists[Φ]), 1, 0.9),
                                             rf"$\Phi = {int(Φ/2**-n)}"+rf"\cdot 2^{{{{-{n}}}}}$",
                                             halign="right", valign="top", fontsize=10)
                              #.opts(title=rf"$\Phi = {int(Φ/2**-n)}"+rf"\cdot 2^{{{{-{n}}}}}$")
                              #for Φ in (np.arange(2**(n-1))*2+1)*ΔΦ])
                              for Φ in Φplot])
layout_incr_marginals.opts(
    hv.opts.Layout(fig_inches=2, shared_axes=False, sublabel_format="", vspace=.1),
    hv.opts.Histogram(edgecolor=None, ylabel="Density", yaxis=None, fontscale=1.3),
    #hv.opts.Area     (edgecolor="none", color="none", backend="matplotlib"),
    hv.opts.Area(ylabel="Density", yaxis=None, fontscale=1.3),
    hv.opts.Histogram(f"Histogram.{sanitize('$x_1$')}", alpha=0.8, facecolor=colors.prestep),
    hv.opts.Histogram(f"Histogram.{sanitize('$x_2$')}", alpha=0.5, facecolor=colors.poststep),
)
if use_area:
    for panel in layout_incr_marginals:
        # Workaround for Area which ignores styling otherwise
        panel.data[('Area', 'X_1')].opts(hv.opts.Area(alpha=0.7, facecolor=colors.prestep, edgecolor=colors.prestep, color="none", linewidth=2, backend="matplotlib"))
        panel.data[('Area', 'X_2')].opts(hv.opts.Area(alpha=0.3, facecolor=colors.poststep, edgecolor=colors.poststep, color="none", linewidth=2, backend="matplotlib"))
        # Workaround for Area which does not add padding
        x1, x2 = panel.range("Δq"); w = x2-x1
        panel.opts(xlim=(x1-0.1*w, x2+0.1*w))
# The very last step is better without transparency, because the distributions are so extreme (one sharp, the other shallow)
if (1 - 2**-n) in Φplot:
    i = Φplot.tolist().index(1 - 2**-n)
    layout_marginals[i].opts(
        hv.opts.Histogram(f"Histogram.{sanitize('$x_1$')}", alpha=1),
        hv.opts.Histogram(f"Histogram.{sanitize('$x_2$')}", alpha=1),
        #hv.opts.Area(f"Area.X_1", alpha=1),  # With Area we probably need to use a workaround as above
        #hv.opts.Area(f"Area.X_2", alpha=1),
    )
for i in (0, 1):
    layout_incr_marginals[i].opts(xlabel="", show_legend=False)
layout_incr_marginals[2] \
    .opts(hv.opts.Overlay(legend_opts={'loc': "center right"}, show_legend=True),
          hv.opts.Histogram(xticks=[0, 0.5, 1, 1.5], xformatter=lambda x: f"${x}$" if x in {0, 1.5} else ""),
          hv.opts.Area(xticks=[0, 0.5, 1, 1.5], xformatter=lambda x: f"${x}$" if x in {0, 1.5} else "")
         )
layout_incr_marginals.cols(1)
```

```{code-cell} ipython3
viz.save(layout_incr_marginals.cols(1), config.paths.figures/"pedag_qhat-incr-marginals_raw.svg")
```

Drawing curves instead of histograms is not as nice:

```{code-cell} ipython3
hv.Layout([hv.Overlay([hist.to.curve() for hist in ov.values() if isinstance(hist, (hv.Histogram, hv.Area))])
           for ov in layout_incr_marginals]).opts(
    hv.opts.Layout(fig_inches=3, shared_axes=False),
    hv.opts.Curve("Curve.x1", color=colors.prestep),
    hv.opts.Curve("Curve.x2", color=colors.poststep),
).cols(2)
```

### Illustration of the refinement of a quantile path (self-consistency)

+++

To help better differentiate the paths:
- we increase $c$ to 16.
- we limit ourselves to the window (0.5, 0.75). So effectively we start after the first two refinement steps ($2^{-2} = 0.25$).

```{code-cell} ipython3
from emdcmp.path_sampling import generate_path_hierarchical_beta, draw_from_beta
```

```{code-cell} ipython3
c_refine = 16
```

```{code-cell} ipython3
import math
import scipy
from emdcmp.path_sampling import f_mid, f
def get_beta_rv(r: float, v: float) -> tuple[float]:
    """
    Return α and β corresponding to `r` and `v`.
    This function is copied from the emd package’s documentation code.
    """
    # Special cases for extreme values of r
    if r < 1e-12:
        return scipy.stats.bernoulli(0)  # Dirac delta at 0
    elif r > 1e12:
        return scipy.stats.bernoulli(1)  # Dirac delta at 1
    # Special cases for extreme values of v
    elif v < 1e-8:
        return get_beta_rv(r, 1e-8)
    elif v > 1e4:
        # (Actual draw function replaces beta by a Bernoulli in this case)
        return scipy.stats.bernoulli(1/(r+1))
    
    # Improve initialization by first solving r=1 <=> α=β
    x0 = scipy.optimize.brentq(f_mid, -5, 20, args=(v,))
    x0 = (x0, x0)
    res = scipy.optimize.root(f, x0, args=[math.log(r), v])
    if not res.success:
        logger.error("Failed to determine α & β parameters for beta distribution. "
                     f"Conditions were:\n  {r=}\n{v=}")
    α, β = np.exp(res.x)
    return scipy.stats.beta(α, β)
```

```{code-cell} ipython3
#end_points = qhat_curves[0].data.query("Φ == 0.5 or Φ == 0.75")  # All six curves have very similar values at 0.5 and 0.75
end_points = qhat_curves[0].data.query("Φ == 0 or Φ == 0.5")
qstart, qend = end_points["q"]                                   # so it doesn’t really matter which one we pick
Φstart, Φend = end_points["Φ"]
curves = []
densities = []
curves.append(hv.Curve([(Φstart, qstart), (Φend, qend)], label=r"$\Delta \Phi = 2^{{-1}}$", kdims=[dims.Φ], vdims=[dims.q]))
for res in [1, 2, 3]:#, 4, 6]:
    # Draw a realisation q at this refinement step
    rng = utils.get_rng("pedag", "qpaths", 1)
    Φ, q = generate_path_hierarchical_beta(
        mixed_ppf, δemd, c_refine, qstart, qend, res, rng, Phistart=Φstart, Phiend=Φend)
    curves.append(hv.Curve(zip(Φ, q), label=r"$\Delta \Phi = 2^{{"+f"-{res+1}"+"}}$", kdims=[dims.Φ], vdims=[dims.q]))
    # For each of the new steps which were drawn, reconstruct the beta distribution used to sample that step
    if res <= 3:
        # (Code copied from definition of `generate_path_hierarchical_beta`)
        qsarr = mixed_ppf(Φ)
        Mvar = c_refine * δemd(Φ)**2
        i = (np.arange(2**res))[::2] + 1
        Δi = 1
        d = q[i+Δi] - q[i-Δi]
        r = (qsarr[i] - qsarr[i-Δi]) / (qsarr[i+Δi]-qsarr[i])  # Ratio of first/second increments
        v = 2*Mvar[i]
        for _i, _d, _r, _v in zip(i, d, r, v):
            rv = get_beta_rv(_r, _v)
            domain = np.linspace(0, 1, 100)
            densities.append(hv.Area((q[_i-Δi] + _d * domain, rv.pdf(domain)),
                                      label=f"{res=}, pos={_i}"))
        rng2 = utils.get_rng("pedag", "qpaths", "betas", res)
        samples = q[i-Δi] + d * draw_from_beta(r, v, rng=rng2, n_samples=10_000).T
        for _i, _s in zip(i, samples.T): # One per step
            key = f"{res=}, pos={_i}"
            #densities.append(hv.Distribution(_s, label=key))
            print("percentile of realisation:", np.searchsorted(np.sort(_s), q[_i]) / len(_s))
```

One reason we need to set $c$ quite high to separate the paths is the monotonicity constraint, which strongly limits how far $q$ can deviate.
In turn, with high $c$ values, we aren’t able to fill out the $δ^{\mathrm{EMD}}$ space due to those constraints.
The consequence is that because we set the $c$ very high to show the increments, the yellow shaded area for $\sqrt{c} δ^{\mathrm{EMD}}$ is not really relevant – it is much too large.

```{code-cell} ipython3
area = hv.Area((Φarr, mixed_ppf(Φarr) - sqrt(c)*δemd(Φarr), mixed_ppf(Φarr) + sqrt(c)*δemd(Φarr)),
               kdims=[dims.Φ], vdims=[dims.q, "q2"], label="delta_emd")
```

```{code-cell} ipython3
#for i in range(1, len(curves)):
stack = None
panels = []
for panel_i, curve_i in enumerate([0, 1, 2, 3]):
    curve = curves[curve_i]
    stack = (stack * curve) if stack else curve
    ov = stack * curve.to.scatter().opts(color=colors.refine_points.values[panel_i])
    ov = ov.relabel(group="NotBottom")
    ov = ov.redim(Φ=hv.Dimension("Φ", label=r"$\Phi$"), q=hv.Dimension("q", label=r"$q$"))
    ov.opts(title=curve.label)
    panels.append(ov)
panels[-1] = ov.relabel(group="Bottom")
    
layout_refinement = hv.Layout(panels)
xticks = np.arange(2**2 + 1) / 2**3
xformatter = lambda x: str(x) if x%2**-2==0 else ""
yticks = [-6.85, -6.8, -6.75, -6.7]
layout_refinement.opts(
    #hv.opts.Curve(color=hv.Cycle(["#CCCCCC", "#BBBBBB", "#AAAAAA", "#999999", "#888888", "#666666", "#444444", "#222222", "#000000"])),
    #hv.opts.Curve(color=hv.Palette("YlOrBr", range=(0.1, .65), reverse=True)),
    hv.opts.Curve(color=colors.refine_lines, padding=0.05, xticks=xticks, fontscale=1.3,
                  xformatter=xformatter, yformatter=lambda y:""),
    hv.opts.Overlay(show_title=False),
    hv.opts.Overlay("Bottom", yticks=yticks),
    hv.opts.Overlay("NotBottom", yticks=yticks, xaxis=None, show_legend=False),
    hv.opts.Scatter(s=12, fontscale=1.3),
    hv.opts.Area(facecolor=colors.δemd, edgecolor="none", color="none", backend="matplotlib"),
    hv.opts.Area(fill_color=colors.δemd, line_color=None, backend="bokeh"),
    hv.opts.Layout(fig_inches=1.8, backend="matplotlib"),
    hv.opts.Layout(hspace=0.2, vspace=0.1, sublabel_format="", fontscale=1.3)
)
layout_refinement[-1].opts(
    hv.opts.Curve(yformatter=lambda y: str(y) if y in {yticks[0], yticks[-1]} else "")
)
layout_refinement.cols(1)
```

```{code-cell} ipython3
viz.save(layout_refinement.cols(1), config.paths.figures/"pedag_qhat-refine_raw.svg")
```

Trying to plot the beta sampling distributions on top of the curves is a mess, especially with Holoviews’ no so solid support for violin plots.
We would probably have to do the entire thing in matplotlib.

_Much_ easier is to just plot the distributions separately, and add them ourselves in Inkscape.
The key to lining them up accurately is to reused the _yticks_ from the curves above as the _xticks_ to the beta distributions. We can then rotate and align the axes in Inkscape.

```{code-cell} ipython3
beta_dists = hv.Layout(densities).opts(
    hv.opts.Area(
                         xticks=yticks, xlim=(q[0], q[-1]), yaxis=None,
                         facecolor="#CCCCCC"),
    #hv.opts.Distribution(
    #                     xticks=yticks, xlim=(q[0], q[-1]), yaxis=None,
    #                     edgecolor=None, facecolor="#CCCCCC"),
    hv.opts.Layout(sublabel_format="", fig_inches=(3.5,1))
                   #fig_inches=(0.965, 0.42))
)
beta_dists.cols(3)
```

```{code-cell} ipython3
hv.save(beta_dists, config.paths.figures/"pedag_qhat-refine_beta-dists_raw.svg")
```

____________________________________________

+++

### Alternative version using Prinz data and Cauchy noise

+++

```python
rng = utils.get_rng("pedag_cdf_prinz")
LP_model = "LP 5"
obs_model = "Cauchy"
data_prinz = replace(Ex_Prinz2004.LP_data, obs_model="Cauchy")
σ = Ex_Prinz2004.fit_gaussian_σ(data_prinz, LP_model, obs_model)
```

+++

```python
candidate = utils.compose(Ex_Prinz2004.AdditiveNoise("Gaussian", σ),
                          Ex_Prinz2004.CandidateModel(LP_model))
```

+++

```python
Q = Ex_Prinz2004.Q(LP_model, obs_model, σ)
#qarr = np.sort(Q(data_prinz.get_data()))
#Φarr = np.arange(1, len(qarr)+1)/(len(qarr)+1)  # Remove end points to leave space for unseen data
```

+++

```python
data = data_prinz.get_data()
mixed_ppf = emd.make_empirical_risk_ppf(Q(data))
synth_ppf = emd.make_empirical_risk_ppf(Q(candidate(data)))
```

+++

For illustration, we use a larger c, so that qhat paths are more separated

+++

```python
#c=Ex_Prinz2004.c_chosen
c = 1.5
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["hide-input"]}

```python
Φarr = np.linspace(0, 1, 192*8)  # *8 is to resolve the curve in the [0.97, 1] zoom
# mixed_ppf = mixed_ppfs["Planck"]
# synth_ppf = synth_ppfs["Planck"]
#mixed_ppf = mixed_ppfs[phys_model]
#synth_ppf = synth_ppfs[phys_model]
def δemd(Φarr): return abs(synth_ppf(Φarr) - mixed_ppf(Φarr))

mixed_curve = hv.Curve(zip(Φarr, mixed_ppf(Φarr)),
                       kdims=[dims.Φ], vdims=[dims.q], label=r"mixed PPF ($q^*$)")
synth_curve = hv.Curve(zip(Φarr, synth_ppf(Φarr)),
                       kdims=[dims.Φ], vdims=[dims.q], label=r"synth PPF ($\tilde{q}$)")
area = hv.Area((Φarr, mixed_ppf(Φarr) - c*δemd(Φarr), mixed_ppf(Φarr) + c*δemd(Φarr)),
               kdims=[dims.Φ], vdims=[dims.q, "q2"], label="δemd")

fig_onlycurves = area * mixed_curve * synth_curve
ticks = dict(xticks=[0, 0.25, 0.5, 0.75, 1],
             xformatter=lambda x: str(x) if x in {0, 1} else "",
             # # Planck
             # yticks=[-6, -3, -0, 3, 6],
             # yformatter=lambda y: str(y) if y in {-6, 6} else "",
             # Prinz LP 5
             yticks=[3, 6, 9],
             yformatter=lambda y: str(y) if y in {3, 9} else "",
            )
hooks = [viz.despine_hook, viz.xlabel_shift_hook(3), viz.ylabel_shift_hook(2)]
fig_onlycurves.opts(
    hv.opts.Curve(f"Curve.{sanitize(mixed_curve.label)}", color=colors.mixed),
    hv.opts.Curve(f"Curve.{sanitize(mixed_curve.label)}", color=colors.mixed, backend="bokeh"),
    hv.opts.Curve(f"Curve.{sanitize(synth_curve.label)}", color=colors.synth),
    hv.opts.Curve(f"Curve.{sanitize(synth_curve.label)}", color=colors.synth, backend="bokeh"),
    hv.opts.Area(facecolor=colors.δemd, edgecolor="none", color="none", backend="matplotlib"),
    hv.opts.Area(fill_color=colors.δemd, line_color=None, backend="bokeh"),
    hv.opts.Overlay(legend_position="top_left", fontscale=1.3, backend="matplotlib"),
    hv.opts.Overlay(legend_position="top_left", backend="bokeh"),
    # NB: despine_hook looks at the ticks, so important to set the ticks of all elements
    hv.opts.Area(**ticks),
    hv.opts.Curve(backend="matplotlib", **ticks, hooks=hooks),
    hv.opts.Area(backend="matplotlib", hooks=[viz.despine_hook, viz.xlabel_shift_hook(), viz.ylabel_shift_hook(5)]),
)
#fig_onlycurves.redim.range(q=(-7.1,6)) # Planck
fig_onlycurves
```

+++

```python
fig_with_qpaths = area * hv.Overlay(qhat_curves) * mixed_curve * synth_curve
fig_with_qpaths.opts(
    fig_inches=4, aspect=1,
    fontscale=1.5,
    legend_position="top_left"
)
```

+++

__________________________________________

+++

(code_pedag-quantile-axis-flip)=
## Quantile axis flip

```{code-cell} ipython3
num_q = 30
```

### RJ vs Planck model

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
rng = utils.get_rng("pedag_cdf_uv")
λ, B = D.data_model(4000, rng=rng)
fitted_σT = Ex_UV.fit_gaussian_σT((λ,B))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
Q = Ex_UV.Q(phys_model, fitted_σT[phys_model].σ, Ex_UV.data_T)
qarr = np.sort(Q((λ,B)))
Φarr = np.arange(1, len(qarr)+1)/(len(qarr)+1)  # Remove end points to leave space for unseen data
```

Select 30 data points at random for the scatter plot

```{code-cell} ipython3
ilst = rng.integers(0, len(qarr), num_q); ilst.sort()
```

Create the figure elements

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
empirical_cdf_uv = hv.Scatter(zip(qarr[ilst], Φarr[ilst]), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="Black body radiation")
empirical_ppf_uv = hv.Scatter(zip(Φarr[ilst], qarr[ilst]), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="Black body radiation")
interpolated_cdf_uv = hv.Curve(zip(qarr, Φarr), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="Black body radiation")
interpolated_ppf_uv = hv.Curve(zip(Φarr, qarr), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="Black body radiation")
```

Combine figure elements and apply styles

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
qrange_uv = (-7,5.5)
qticks_uv = [-6.6, -5, -3.3, -1.6, 0, 1.6, 3.3, 5]
qform_uv = lambda x: str(x) if x in {-6.6, 5} else ""
fig_uv = (interpolated_cdf_uv.redim.range(q=qrange_uv)*empirical_cdf_uv.redim.range(q=qrange_uv)
         ).opts(title="empirical CDF") \
         + (interpolated_ppf_uv.redim.range(q=qrange_uv)*empirical_ppf_uv.redim.range(q=qrange_uv)
           ).opts(title="empirical PPF\n(quantile function)")
fig_uv.opts(title=empirical_cdf_uv.label)
fig_uv.opts(
    hv.opts.Scatter("CDF", xticks=qticks_uv, xformatter=qform_uv),
    hv.opts.Scatter("PPF", yticks=qticks_uv, yformatter=qform_uv),
    hv.opts.Curve("CDF", xticks=qticks_uv, xformatter=qform_uv),
    hv.opts.Curve("PPF", yticks=qticks_uv, yformatter=qform_uv),
)
fig_uv
```

### Prinz model - Gaussian noise

Data are generated with Gaussian observation noise.  
Loss assumes Gaussian observation noise.

```{code-cell} ipython3
rng = utils.get_rng("pedag_cdf_prinz")
LP_model = "LP 5"
obs_model = "Gaussian"
data_prinz = Ex_Prinz2004.LP_data
σ = Ex_Prinz2004.fit_gaussian_σ(data_prinz, LP_model, obs_model)
```

```{code-cell} ipython3
Q = Ex_Prinz2004.Q(LP_model, obs_model, σ)
qarr = np.sort(Q(data_prinz.get_data()))
Φarr = np.arange(1, len(qarr)+1)/(len(qarr)+1)  # Remove end points to leave space for unseen data
```

Select 30 data points at random for the scatter plot

```{code-cell} ipython3
ilst = rng.integers(0, len(qarr), num_q); ilst.sort()
```

Create the figure elements

```{code-cell} ipython3
empirical_cdf_prinz_gaussian = hv.Scatter(zip(qarr[ilst], Φarr[ilst]), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="Neuron w/ Gaussian noise")
empirical_ppf_prinz_gaussian = hv.Scatter(zip(Φarr[ilst], qarr[ilst]), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="Neuron w/ Gaussian noise")
interpolated_cdf_prinz_gaussian = hv.Curve(zip(qarr, Φarr), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="Neuron w/ Gaussian noise")
interpolated_ppf_prinz_gaussian = hv.Curve(zip(Φarr, qarr), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="Neuron w/ Gaussian noise")
```

Combine figure elements and apply styles

```{code-cell} ipython3
qrange_prinz_gaussian = (3.5, 17.5)
qticks_prinz_gaussian = [4, 6, 8, 10, 12, 14, 16]
qform_prinz_gaussian = lambda q: str(q) if q in {4, 16} else ""
fig_prinz_gaussian = \
    (interpolated_cdf_prinz_gaussian.redim.range(q=qrange_prinz_gaussian)*empirical_cdf_prinz_gaussian.redim.range(q=qrange_prinz_gaussian)
    ).opts(title="empirical CDF") \
    + (interpolated_ppf_prinz_gaussian.redim.range(q=qrange_prinz_gaussian)*empirical_ppf_prinz_gaussian.redim.range(q=qrange_prinz_gaussian)
      ).opts(title="empirical PPF\n(quantile function)")
fig_prinz_gaussian.opts(title=interpolated_cdf_prinz_gaussian.label)
fig_prinz_gaussian.opts(
    hv.opts.Scatter("CDF", xticks=qticks_prinz_gaussian, xformatter=qform_prinz_gaussian),
    hv.opts.Scatter("PPF", yticks=qticks_prinz_gaussian, yformatter=qform_prinz_gaussian),
    hv.opts.Curve("CDF", xticks=qticks_prinz_gaussian, xformatter=qform_prinz_gaussian),
    hv.opts.Curve("PPF", yticks=qticks_prinz_gaussian, yformatter=qform_prinz_gaussian),
)
fig_prinz_gaussian
```

### Prinz model – Cauchy noise

Same as above, but  
Data are generated with Cauchy observation noise.  
Loss assumes Cauchy observation noise.

```{code-cell} ipython3
rng = utils.get_rng("pedag_cdf_prinz")
LP_model = "LP 5"
obs_model = "Cauchy"
data_prinz = replace(Ex_Prinz2004.LP_data, obs_model="Cauchy")
σ = Ex_Prinz2004.fit_gaussian_σ(data_prinz, LP_model, obs_model)
```

```{code-cell} ipython3
Q = Ex_Prinz2004.Q(LP_model, obs_model, σ)
qarr = np.sort(Q(data_prinz.get_data()))
Φarr = np.arange(1, len(qarr)+1)/(len(qarr)+1)  # Remove end points to leave space for unseen data
```

Select 30 data points at random for the scatter plot

```{code-cell} ipython3
ilst = rng.integers(0, len(qarr), num_q); ilst.sort()
```

Create the figure elements

```{code-cell} ipython3
empirical_cdf_prinz_cauchy = hv.Scatter(zip(qarr[ilst], Φarr[ilst]), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="Neuron w/ Cauchy noise")
empirical_ppf_prinz_cauchy = hv.Scatter(zip(Φarr[ilst], qarr[ilst]), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="Neuron w/ Cauchy noise")
interpolated_cdf_prinz_cauchy = hv.Curve(zip(qarr, Φarr), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="Neuron w/ Cauchy noise")
interpolated_ppf_prinz_cauchy = hv.Curve(zip(Φarr, qarr), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="Neuron w/ Cauchy noise")
```

Combine figure elements and apply styles

```{code-cell} ipython3
qrange_prinz_cauchy = (3, 8.5)
qticks_prinz_cauchy = [3, 4, 5, 6, 7, 8]
qform_prinz_cauchy = lambda q: str(q) if q in {3, 8} else ""
fig_prinz_cauchy = \
    (interpolated_cdf_prinz_cauchy.redim.range(q=qrange_prinz_cauchy)*empirical_cdf_prinz_cauchy.redim.range(q=qrange_prinz_cauchy)
    ).opts(title="empirical CDF") \
    + (interpolated_ppf_prinz_cauchy.redim.range(q=qrange_prinz_cauchy)*empirical_ppf_prinz_cauchy.redim.range(q=qrange_prinz_cauchy)
      ).opts(title="empirical PPF")
fig_prinz_cauchy.opts(title=interpolated_cdf_prinz_cauchy.label)
fig_prinz_cauchy.opts(
    hv.opts.Scatter("CDF", xticks=qticks_prinz_cauchy, xformatter=qform_prinz_cauchy),
    hv.opts.Scatter("PPF", yticks=qticks_prinz_cauchy, yformatter=qform_prinz_cauchy),
    hv.opts.Curve("CDF", xticks=qticks_prinz_cauchy, xformatter=qform_prinz_cauchy),
    hv.opts.Curve("PPF", yticks=qticks_prinz_cauchy, yformatter=qform_prinz_cauchy),
)
fig_prinz_cauchy
```

### High-dimensional Gaussian

We use a 30 dimensional isotropic Gaussian both to generate the data and evaluate the loss.

```{code-cell} ipython3
rng = utils.get_rng("pedag_cdf_highd")
data_highd = stats.multivariate_normal(cov=np.eye(30))
```

```{code-cell} ipython3
Q = data_highd.logpdf
qarr = np.sort(Q(data_highd.rvs(4000, random_state=rng)))
Φarr = np.arange(1, len(qarr)+1)/(len(qarr)+1)  # Remove end points to leave space for unseen data
```

Select 30 data points at random for the scatter plot

```{code-cell} ipython3
ilst = rng.integers(0, len(qarr), num_q); ilst.sort()
```

Create the figure elements

```{code-cell} ipython3
empirical_cdf_highd = hv.Scatter(zip(qarr[ilst], Φarr[ilst]), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="High-dimensional Gaussian")
empirical_ppf_highd = hv.Scatter(zip(Φarr[ilst], qarr[ilst]), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="High-dimensional Gaussian")
interpolated_cdf_highd = hv.Curve(zip(qarr, Φarr), kdims=[dims.q], vdims=[dims.Φ], group="CDF", label="High-dimensional Gaussian")
interpolated_ppf_highd = hv.Curve(zip(Φarr, qarr), kdims=[dims.Φ], vdims=[dims.q], group="PPF", label="High-dimensional Gaussian")
```

Combine figure elements and apply styles

```{code-cell} ipython3
qrange_highd = (-60, -30)
qticks_highd = [-60, -50, -40, -30]
qform_highd = lambda q: str(q) if q in {-60, -30} else ""
fig_highd = \
    (interpolated_cdf_highd.redim.range(q=qrange_highd)*empirical_cdf_highd.redim.range(q=qrange_highd)
    ).opts(title="empirical CDF") \
    + (interpolated_ppf_highd.redim.range(q=qrange_highd)*empirical_ppf_highd.redim.range(q=qrange_highd)
      ).opts(title="empirical PPF")
fig_highd.opts(title=interpolated_cdf_highd.label)
fig_highd.opts(
    hv.opts.Scatter("CDF", xticks=qticks_highd, xformatter=qform_highd),
    hv.opts.Scatter("PPF", yticks=qticks_highd, yformatter=qform_highd),
    hv.opts.Curve("CDF", xticks=qticks_highd, xformatter=qform_highd),
    hv.opts.Curve("PPF", yticks=qticks_highd, yformatter=qform_highd),
)
fig_highd
```

### Combine cdf plots

```{code-cell} ipython3
Φticks = [0, 0.25, 0.5, 0.75, 1]
Φform = lambda x: str(x) if x in {0, 1} else ""
fig = fig_prinz_gaussian + fig_prinz_cauchy + fig_uv + fig_highd
# NB: xformatter/yformatter is not called on Overlay plots, which is why we set it on each Scatter/Curve
aspect = 1
fig.opts(hv.opts.Scatter(color=colors.data, s=12, aspect=aspect),
         hv.opts.Curve(color=colors.interp_ppf, linewidth=3, aspect=aspect),
         hv.opts.Scatter("CDF", ylim=(0,1), yticks=Φticks, yformatter=Φform),
         hv.opts.Curve  ("CDF", ylim=(0,1), yticks=Φticks, yformatter=Φform),
         hv.opts.Scatter("PPF", xlim=(0,1), xticks=Φticks, xformatter=Φform),
         hv.opts.Curve  ("PPF", xlim=(0,1), xticks=Φticks, xformatter=Φform),
         hv.opts.Overlay(show_legend=False, aspect=aspect),
         #hv.opts.Overlay("CDF",hooks=[viz.despine_hook, viz.xlabel_shift_hook(8), viz.ylabel_shift_hook(6)]),
         #hv.opts.Overlay("PPF",hooks=[viz.despine_hook, viz.xlabel_shift_hook(8), viz.ylabel_shift_hook(13,.5)]),
         hv.opts.Layout(transpose=True, shared_axes=False,
                        fig_inches=config.figures.defaults.fig_inches/2,
                        sublabel_position=(0.4, 0.9), sublabel_format="")
)
for panel in fig:
    # Titles for each column
    if panel.group == "CDF":
        # Use label for title, making it two lines. Lines are split on either the first or second space
        if panel.label == "Black body radiation": title = "Black body\nradiation"
        else: title = panel.label.replace(" ", "\n", 1)
        panel.opts(title=title)
    else:
        panel.opts(title="")
    # Only display y labels on the first column
    if panel.label == "Neuron w/ Gaussian noise" and panel.group == "CDF":
        panel.opts(hooks=[viz.despine_hook, viz.xlabel_shift_hook(8), viz.ylabel_shift_hook(7)])
    elif panel.label == "Neuron w/ Gaussian noise" and panel.group == "PPF":
        panel.opts(hooks=[viz.despine_hook, viz.xlabel_shift_hook(8), viz.ylabel_shift_hook(7, -2)])
    else:
        panel.opts(ylabel="", hooks=[viz.despine_hook, viz.xlabel_shift_hook(8)])
        pass
fig.cols(2)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
viz.save(fig, config.paths.figures/"pedag_CDF_PPF_raw.svg")
#viz.save(fig, config.paths.figures/"pedag_CDF_PPF.pdf")
#viz.save(fig.clone().cols(1).opts(vspace=.75),
#         config.paths.figures/"pedag_CDF_PPF.svg")  # Sized to insert in margin
```

Finish in Inkscape:
- Make alignment of ylabels pixel-perfect
- Add large row labels "Empirical CDF" and "Empirical PPF"
- Squeeze vertical whitespace
- Save to PDF

+++

## Exported notebook variables
These can be inserted into other pages

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
glue("color_mixed", color_labels.mixed, raw_myst=True, raw_latex=True)
glue("color_synth", color_labels.synth, raw_myst=True, raw_latex=True)
glue("color_deltaemd", color_labels.δemd, raw_myst=True, raw_latex=True)
glue("color_qhat", color_labels.qhat, raw_myst=True, raw_latex=True)
glue("color_data", color_labels.data, raw_myst=True, raw_latex=True)
glue("color_interp", color_labels.interp_ppf, raw_myst=True, raw_latex=True)
glue("color_marginal", color_labels.marginal, raw_myst=True, raw_latex=True)
glue("color_prestep", color_labels.prestep, raw_myst=True, raw_latex=True)
glue("color_poststep", color_labels.poststep, raw_myst=True, raw_latex=True)
glue("pedag_paths_c", c, raw_myst=True, raw_latex=True)
glue("pedag_paths_c_refine", c_refine, raw_myst=True, raw_latex=True)
glue("pedag_num_paths", num2words(len(qhat_curves)), raw_myst=True, raw_latex=True)
glue("pedag_num_M_marginals", f"{M_marginals:,}", raw_myst=True, raw_latex=True)  # Number of paths used to compute marginals
glue("pedag_num_q", num_q, raw_myst=True, raw_latex=True)  # Number of points along CDF to highlight

glue("pedag_uv_s", D.s.m, raw_html=viz.format_pow2(D.s.m, 'latex'), # latex b/c used inside $…$ expression
                          raw_latex=viz.format_pow2(D.s.m, 'latex'))
glue("pedag_uv_lambda_range", f"{D.λmin:.0~fP} to {D.λmax:.0~fP}",
                              raw_latex=(latexstr:=f"{D.λmin:.0~fLx}\\text{{ to }}{D.λmax:.0~fLx}"),
                              raw_myst=(latexstr:=f"{D.λmin:.0~fL}\\text{{ to }}{D.λmax:.0~fL}") )
    # In latex we use siunitx because to ensure μ is printed,
    # but in myst we can’t because MathJax doesn’t support siunitx
    # (The value is used inside a $…$ expression)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
emd.utils.GitSHA(packages=["emd-falsify"])
```

```{code-cell} ipython3

```
