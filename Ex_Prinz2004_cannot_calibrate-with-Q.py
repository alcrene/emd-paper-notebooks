# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md:myst
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# ---
# math:
#     '\Bemd' : 'B^{\mathrm{EMD}}_{#1}'
#     '\Bconf': 'B^{\mathrm{epis}}_{#1}'
#     '\BQ' : 'B^{Q}_{#1}'
#     '\nN'   : '\mathcal{N}'
#     '\Unif' : '\operatorname{Unif}'
#     '\Mtrue': '\mathcal{M}_{\mathrm{true}}'
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Rejection probability is not predicted by loss distribution
#
# {{ prolog }}
#
# %{{ startpreamble }}
# %{{ endpreamble }}

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# > **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).
#
# > **NOTE** This notebook is synced with a Python file using [Jupytext](https://jupytext.readthedocs.io/). **That file is required** to run this notebook, and it must be in the current working directory.
#

# %% [markdown] editable=true slideshow={"slide_type": ""}
# This notebook explores a similar question as [](#code_aleatoric-cannot-substitute-epistemic), namely why we cannot use the loss distribution directly and instead really need to estimate the epistemic distribution.
#
# Here we explore the question in the context of our calibration experiments. Ultimately the goal is to predict the probability that a replication of the experiment would select model $A$ over model $B$, and this is exactly what our calibration experiments test (with synthetic replications). So if we can replace $\Bemd{}$ by a hypothetical $\BQ{}$ defined solely in terms of the loss distribution and still finding a correlation with $\Bconf{}$, that would suggest that we don’t really need $\Bemd{}$ (which depends on a more complicated distribution and ad hoc parameter $c$) – we could get away with considering just the loss distribution, which is much simpler and therefore more appealing.
#
# Of course, there is no reason to expect that the loss distribution (which is a result of aleatoric uncertainty) will be related to *replication* uncertainty – indeed, that is the point of the [aforementioned notebook](#code_aleatoric-cannot-substitute-epistemic). But the question gets asked, so in addition to arguing on the basis of principle, it is good to tackle the question head-on, in the most direct way possible, with an actual experiment.
#
# One challenge is how best to define $B^Q$: since it measures aleatoric instead of replication/epistemic uncertainty, there is no ontologically obvious way to do this. So instead we will use the _mathematically_ obvious way and “hope for the best”:[^ml-research] we match the form of [](#eq_Bemd_def), replacing the $R$-distribution by the $Q$-distribution:
#
# $$\BQ{AB;c_Q} := P(Q_A < Q_B + c_Q)\,.$$ (eq_BQ_def)
#
# Note that now the probability now is over the samples of $\Mtrue$ rather than over replications.
# The parameter $c_Q$ is an additional degree of freedom; we include it so that both $\BQ{}$ and $\Bemd{}$ have the same number of free parameters, and thus make the comparison more fair. In practice we will select the $c_Q$ which maximizes the correlation with $\Bconf{}$, thus giving $\BQ{}$ the best chance to match the performance of $\Bemd{}$.
#
# The advantage of [](#eq_BQ_def) is that it is easily estimated from observed samples, with no need to define an ad hoc distribution over quantile functions.
# The disadvantage is that it does not capture the effect of misspecification on the ability of a model to replicate. As we will see below, this makes it a very poor predictor for $\Bconf{}$.
#
# Note also that appropriate values of $c_Q$ are entirely dependent on what the typical values of the loss are.
# If the loss is given in terms of a probability density, rather than a probability mass, then this is not scale independent.
# There is even less reason to expect the values of $c_Q$ therefore to generalize to other problems, in contrast to what we observed with $\Bemd{}$.
#
# Of course, given the highly nonlinear shape of loss distributions, we actually expect that the appropriate "correction factor" would not be a constant shift but a function of $Q_A$ and $Q_B$, boosting the difference in some areas and reducing it in others. Such a function however would have infinitely more degrees of freedom than the simple $c$ scaling we propose with $\Bemd{}$. In fact, if we say that our hypothesis is just that there exists _some_ functional $B(Q_A,Q_B) \to \RR$ which correlates with $\Bconf{}$, then the $\Bemd{}$ is subsumed in that hypothesis, with the infinite-dimensional functional space reduced to a single scalar parameter $c$.
# For this reason we think that $\BQ{}$ as defined in [](#eq_BQ_def) is the fairest expression of a comparison criterion which uses $Q_A$ and $Q_B$ directly, with no further assumptions.
#
#
# [^ml-research]: Incidentally, this “match the form and hope for the best” is not uncommon in the machine learning literature, and partly explains the inconsistent results with attempts at getting models to generalize.

# %% [markdown]
# ## Load the epistemic dists used in the calibration experiments

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-input"]
# from Ex_Prinz2004 import EpistemicDist, L_data, Linf, colors, dims
# from config import config

# %%
import viz
import viz.emdcmp

# %%
from functools import partial
from itertools import chain

# %% [markdown]
# Load the customized calibration task which computes $\BQ{}$ instead of $\Bemd{}$.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-input"]
# from task_bq import CalibrateBQ, calib_point_dtype

# %%
import numpy as np

# %%
import emdcmp

# %%
import holoviews as hv
hv.extension("matplotlib", "bokeh")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Execution

# %% [markdown]
# :::{admonition} How many experiments
# :class: hint margin
#
# The function `emd.viz.calibration_plot` attempts to collect results into 16 bins, so making $N$ a multiple of 16 works nicely. (With the constraint that no bin can have less than 16 points.)
#
# For an initial pilot run, we found $N=64$ or $N=128$ to be good numbers. These numbers produce respectively 4 or 8 bins, which is often enough to check that $\Bemd{}$ and $\Bconf{}$ are reasonably distributed and that the epistemic distribution is actually probing the transition from strong to equivocal evidence.
# A subsequent run with $N \in \{256, 512, 1024\}$ can then refine and smooth the curve.
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Full list of epistemic dists. This is more than is useful for testing the $\BQ{}$ variant of risk distributions.
#
# ```python
# N = 64
# #N = 512
# Ωdct = {(f"{Ω.a} vs {Ω.b}", Ω.ξ_name, Ω.σo_dist, Ω.τ_dist, Ω.σi_dist): Ω
#         for Ω in (EpistemicDist(N, a, b, ξ_name, σo_dist, τ_dist, σi_dist)
#                   for (a, b) in [("A", "B"), ("A", "D"), ("C", "D")]
#                   for ξ_name in ["Gaussian", "Cauchy"]
#                   for σo_dist in ["low noise", "high noise"]
#                   for τ_dist in ["short input correlations", "long input correlations"]
#                   for σi_dist in ["weak input", "strong input"]
#             )
#        }
# ```
#
# Instead we only run the experiments on the 6 epistemic distributions used in the main paper.

# %%
N = 64
N = 128
N = 512
N = 2048
Ωdct = {(f"{Ω.a} vs {Ω.b}", Ω.ξ_name, Ω.σo_dist, Ω.τ_dist, Ω.σi_dist): Ω
        for Ω in [
            EpistemicDist(N, "A", "D", "Gaussian", "low noise", "short input correlations", "weak input"),
            EpistemicDist(N, "C", "D", "Gaussian", "low noise", "short input correlations", "weak input"),
            EpistemicDist(N, "A", "B", "Gaussian", "low noise", "short input correlations", "weak input"),
            EpistemicDist(N, "A", "D", "Gaussian", "low noise", "short input correlations", "strong input"),
            EpistemicDist(N, "C", "D", "Gaussian", "low noise", "short input correlations", "strong input"),
            EpistemicDist(N, "A", "B", "Gaussian", "low noise", "short input correlations", "strong input"),
        ]
       }

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{hint}
# :class: margin
# `Calibrate` will iterate over the data models twice, so it is important that the iterable passed as `data_models` not be consumable.
# :::

# %% editable=true slideshow={"slide_type": ""}
c_chosen = 2**-2
c_list = [2**-6, 2**-4, 2**-2, 2**0, 2**4]

# %%
cQ_chosen = -2**-1
cQ_list=[#-2**1,  2**1,
         #-2**0,  2**0,
         -2**-1, 2**-1,
         #-2**-2, 2**-2,
         -2**-3, 2**-3,
         #0
        ]

# %% editable=true slideshow={"slide_type": ""}
tasks_normal = {}
tasks_BQ = {}
for Ωkey, Ω in Ωdct.items():
    task = CalibrateBQ(
        reason = f"BQ calibration attempt – {Ω.a} vs {Ω.b} - {Ω.ξ_name} - {Ω.σo_dist} - {Ω.τ_dist} - {Ω.σi_dist} - {N=}",
        c_list = cQ_list,
        experiments = Ω.generate(N),
        Ldata = L_data,
        Linf = Linf
    )
    tasks_BQ[Ωkey] = task

    task = emdcmp.tasks.Calibrate(
        reason = f"Prinz calibration – {Ω.a} vs {Ω.b} - {Ω.ξ_name} - {Ω.σo_dist} - {Ω.τ_dist} - {Ω.σi_dist} - {N=}",
        c_list = c_list,
        experiments = Ω.generate(N),
        Ldata = L_data,
        Linf = Linf
    )
    tasks_normal[Ωkey] = task

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The code below creates task files which can be executed from the command line with the following:
#
#     smttask run -n1 --import config <task file>

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "skip-execution"]
#     for task in chain(tasks_normal.values(),
#                       tasks_BQ.values()):
#         if not task.has_run:  # Don’t create task files for tasks which have already run
#             Ω = task.experiments
#             taskfilename = f"prinz_calibration__{Ω.a}vs{Ω.b}_{Ω.ξ_name}_{Ω.σo_dist}_{Ω.τ_dist}_{Ω.σi_dist}_N={Ω.N}_c={task.c_list}"
#             task.save(taskfilename)

# %% [markdown]
#     import smttask
#     smttask.config.record = False

# %% [markdown]
#     task = next(iter(tasks.values()))
#     task.run()

# %%
from warnings import filterwarnings, catch_warnings
with catch_warnings():
    filterwarnings("ignore", "The previous implementation of stack is deprecated")
    for task in tasks.values():
        task.run()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Analysis

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# :::{dropdown} Workaround to be able run notebook while a new calibration is running
# ```python
# # Workaround to be able run notebook while a new calibration is running:
# # Use the last finished task
# from smttask.view import RecordStoreView
# rsview = RecordStoreView()
# rsview.list
# task = emd.tasks.Calibrate.from_desc(rsview.last.parameters)
# #task = emd.tasks.Calibrate.from_desc(rsview.get('20231017-222339_1c9062').parameters)
# #task = emd.tasks.Calibrate.from_desc(rsview.get('20231024-000624_baf0b3').parameters)
# ```
# :::

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
#     task = tasks['C vs D', 'Gaussian', 'low noise', 'short input correlations', 'weak input']
#
#     assert all(task.has_run for task in tasks.values()), "Run the calibrations from the command line environment, using `smttask run`. Executing it as part of a Jupyter Book build would take a **long** time."
#
#     calib_results = task.unpack_results(task.run())

# %%
calib_bq = viz.emdcmp.calibration_plot(calib_results)

# %%
calib_bq.Bemd_hists.overlay()

# %%
calib_bq.overlayed_scatters


# %% [markdown] editable=true slideshow={"slide_type": ""}
# We can check the efficiency of sampling by plotting histograms of $\Bemd{}$ and $\Bconf{}$: ideally the distribution of $\Bemd{}$ is flat, and that of $\Bconf{}$ is equally distributed between 0 and 1. Since we need enough samples at every subinterval of $\Bemd{}$, it is the most sparsely sampled regions which determine how many calibration datasets we need to generate. (And therefore how long the computation needs to run.)
# Beyond making for shorter compute times, a flat distribution however isn’t in and of itself a good thing: more important is that the criterion is able to resolve the models when it should.

# %% editable=true slideshow={"slide_type": ""}
class CalibHists:
    def __init__(self, hists_Q=None, hists_epis=None):
        frames = {viz.format_pow2(c): hists_Q[c] * hists_epis[c] for c in hists_Q}
        self.hmap = hv.HoloMap(frames, kdims=["c"])
        self.hists_Q  = hists_Q
        self.hists_epis = hists_epis

def calib_hist(task) -> CalibHists:
    calib_results = task.unpack_results(task.run())

    hists_Q = {}
    hists_epis = {}
    for c, res in calib_results.items():
        hists_Q[c] = hv.Histogram(np.histogram(res["BQ"], bins="auto", density=False), kdims=["BQ"], label="BQ")
        hists_epis[c] = hv.Histogram(np.histogram(res["Bepis"].astype(int), bins="auto", density=False), kdims=["Bepis"], label="Bepis")
    #frames = {viz.format_pow2(c): hists_Q[c] * hists_epis[c] for c in hists_Q}
        
    #hmap = hv.HoloMap(frames, kdims=["c"])
    hists = CalibHists(hists_Q=hists_Q, hists_epis=hists_epis)
    hists.hmap.opts(
        hv.opts.Histogram(backend="bokeh",
                          line_color=None, alpha=0.75,
                          color=hv.Cycle(values=config.figures.colors.light.cycle)),
        hv.opts.Histogram(backend="matplotlib",
                          color="none", edgecolor="none", alpha=0.75,
                          facecolor=hv.Cycle(values=config.figures.colors.light.cycle)),
        hv.opts.Overlay(backend="bokeh", legend_position="top", width=400),
        hv.opts.Overlay(backend="matplotlib", legend_position="top", fig_inches=4)
    )
    #hv.output(hmap, backend="bokeh", widget_location="right")
    return hists


# %% editable=true slideshow={"slide_type": ""}
def panel_calib_hist(task, c_list: list[float]) -> hv.Overlay:
    hists_hmap = calib_hist(task)
    
    _hists_Q = {c: h.relabel(group="BQ", label=f"$c={viz.format_pow2(c, format='latex')}$")
                  for c, h in hists_hmap.hists_Q.items() if c in c_list}
    for c in c_list:
        α = 1 if c == c_chosen else 0.8
        _hists_Q[c].opts(alpha=α)
    histpanel_emd = hv.Overlay(_hists_Q.values())

    histpanel_emd.redim(Bemd=dims.Bemd, Bconf=dims.Bconf, c=dims.c)

    histpanel_emd.opts(
        hv.opts.Histogram(backend="matplotlib", color="none", edgecolor="none", facecolor=colors.calib_curves),
        hv.opts.Histogram(backend="bokeh", line_color=None, fill_color=colors.calib_curves),
        hv.opts.Overlay(backend="matplotlib",
                        legend_cols=3,
                        legend_opts={"columnspacing": .5, "alignment": "center",
                                     "loc": "upper center"},
                        hooks=[viz.yaxis_off_hook, partial(viz.despine_hook(), left=True)]),
        hv.opts.Overlay(backend="bokeh",
                        legend_cols=3),
        hv.opts.Overlay(backend="matplotlib",
                        fig_inches=config.figures.defaults.fig_inches,
                        aspect=1, fontscale=1.3,
                        xlabel=r"$B^{Q}$", ylabel=r"$B^{\mathrm{epis}}$",
                        ),
        hv.opts.Overlay(backend="bokeh",
                        width=4, height=4)
    )

    return histpanel_emd


# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{hint}
# :class: margin
#
# Default options of `calibration_plot` can be changed by updating the configuration object `config.emd.viz.matplotlib` (relevant fields are `calibration_curves`, `prohibited_areas`, `discouraged_area`).
# As with any HoloViews plot element, they can also be changed on a per-object basis by calling their [`.opts` method](https://holoviews.org/user_guide/Applying_Customizations.html).
# :::

# %% editable=true slideshow={"slide_type": ""}
def panel_calib_curve(task, c_list: list[float]) -> hv.Overlay:
    calib_results = task.unpack_results(task.run())
    calib_point_dtype.names = ("Bemd", "Bconf")  # Hack to relabel fields in the way calibration_plot expects them
    calib_plot = emdcmp.viz.calibration_plot(calib_results)
    calib_point_dtype.names = ("BQ", "Bepis")
    
    calib_curves, prohibited_areas, discouraged_areas = calib_plot
    
    for c, curve in calib_curves.items():
        calib_curves[c] = curve.relabel(label=f"$c={viz.format_pow2(c, format='latex')}$")
    
    for c in c_list:
        α = 1 #if c == c_chosen else 0.85
        #w = 3 if c == c_chosen else 2
        w = 2
        calib_curves[c].opts(alpha=α, linewidth=w)
    
    curve_panel = prohibited_areas * discouraged_areas * hv.Overlay(calib_curves.select(c=c_list).values())

    curve_panel.redim(Bemd=dims.Bemd, Bconf=dims.Bconf, c=dims.c)

    # NB: When previously set options don’t specify `backend`, setting a 'bokeh' option can unset a previously set 'matplotlib' one
    curve_panel.opts(
        hv.opts.Curve(backend="matplotlib", color=colors.calib_curves),
        hv.opts.Curve(backend="bokeh", color=colors.calib_curves, line_width=3),
        hv.opts.Area(backend="matplotlib", alpha=0.5),
        #hv.opts.Area(backend="bokeh", alpha=0.5),
        hv.opts.Overlay(backend="matplotlib", legend_position="top_left", legend_cols=1, hooks=[viz.despine_hook]),
        hv.opts.Overlay(backend="bokeh", legend_position="top_left", legend_cols=1),
        hv.opts.Overlay(backend="matplotlib",
                        fig_inches=config.figures.defaults.fig_inches,
                        aspect=1, fontscale=1.3,
                        xlabel=r"$B^{Q}$", ylabel=r"$B^{\mathrm{epis}}$",
                        ),
        hv.opts.Overlay(backend="bokeh",
                        width=4, height=4)
    )
    
    return curve_panel


# %%
c_chosen = -2**-1
c_list=[#-2**1,  2**1,
        #-2**0,  2**0,
        -2**-1, 2**-1,
        #-2**-2, 2**-2,
        -2**-3, 2**-3,
        #0
       ]

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# task = tasks['C vs D', 'Gaussian', 'low noise', 'short input correlations', 'weak input']
# assert task.has_run
# hv.output(calib_hist(task).hmap,
#           backend="bokeh", widget_location="right")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{admonition} Hint: Diagnosing $\Bemd{}$ and $\Bconf{}$ histograms
# :class: hint dropdown
#
# $\Bemd{}$ distribution which bulges around 0.5.
# ~ *May* indicate that $c$ is too large and the criterion underconfident.
# ~ *May also* indicate that the calibration distribution is generating a large number of (`data`, `modelA`, `modelB`) triples which are essentially undecidable. If neither model is a good fit to the data, then their $δ^{\mathrm{EMD}}$ discrepancies between mixed and synthetic PPFs will be large, and they will have broad distributions for the expected risk. Broad distributions overlap more, hence the skew of $\Bemd{}$ towards 0.5.
#
# $\Bemd{}$ distribution which is heavy at both ends.
# ~ *May* indicate that $c$ is too small and the criterion overconfident.
# ~ *May also* indicate that the calibration distribution is not sampling enough ambiguous conditions. In this case the answer is *not* to increase the value of $c$ but rather to tighten the calibration distribution to focus on the area with $\Bemd{}$ close to 0.5. It may be possible to simply run the calibration longer until there have enough samples everywhere, but this is generally less effective than adjusting the calibration distribution.
#
# $\Bemd{}$ distribution which is heavily skewed either towards 0 or 1.
# ~ Check that the calibration distribution is using both candidate models to generate datasets. The best is usually to use each candidate to generate half of the datasets: then each model should fit best in roughly half the cases.
# The skew need not be removed entirely – one model may just be more permissive than the other.
# ~ This can also happen when $c$ is too small.
#
# $\Bconf{}$ distribution which is almost entirely on either 0 or 1.
# ~ Again, check that the calibration distribution is using both models to generate datasets.
# ~ If each candidate is used for half the datasets, and we *still* see ueven distribution of $\Bconf{}$, then this can indicate a problem: it means that the ideal measure we are striving towards (true expected risk) is unable to identify that model used to generate the data. In this case, tweaking the $c$ value is a waste of time: the issue then is with the problem statement rather than the $\Bemd{}$ calibration. Most likely the issue is that the loss is ill-suited to the problem:
#   + It might not account for rotation/translation symmetries in images, or time dilation in time-series.
#   + One model’s loss might be lower, even on data generated with the other model. This can happen with a log posterior, when one model has more parameters: its higher dimensional prior "dilutes" the likelihood. This may be grounds to reject the more complex model on the basis of preferring simplicity, but it is *not* grounds to *falsify* that model. (Since it may still fit the data equally well.)
#
# :::

# %%
pcc = panel_calib_curve(task, c_list=c_list)
pcc

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# frames_curves = {}
# frames_hists = {}
# _c_list = [-2**-3]
# #_c_list = [-2**-1, -2**-3, 2**-3, 2**-1]
# _c_list = c_list
# for key, task in tasks.items():
#     #task = calib_tasks_to_show[0]
#     histpanel_emd = panel_calib_hist(task, _c_list) \
#                     .opts(show_legend=False) \
#                     .relabel(group="Calibhist")
#     curve_panel = panel_calib_curve(task, _c_list)
#     scatters = hv.Overlay([curve.to.scatter() for curve in curve_panel.Curve])
#     #fig = curve_panel << hv.Empty() << histpanel_emd
#     frames_curves[key[0], key[4]] = curve_panel * scatters
#     frames_hists[key[0], key[4]] = histpanel_emd
#     
# fig = hv.HoloMap(frames_curves, kdims=["candidates", "input"]) \
#       << hv.Empty() \
#       << hv.HoloMap(frames_hists, kdims=["candidates", "input"])
#
# #calib_results = task.unpack_results(task.run())
# #calib_point_dtype.names = ("Bemd", "Bconf")  # Hack to relabel fields in the way calibration_plot expects them
# #calib_plot = emdcmp.viz.calibration_plot(calib_results)  # Used below
# #calib_point_dtype.names = ("BQ", "Bepis")
#
# fig.opts(
#     hv.opts.AdjointLayout(backend="matplotlib", fig_inches=4),
#     hv.opts.Overlay(backend="matplotlib", fig_inches=5),
#     #hv.opts.Overlay("Calibhist", backend="matplotlib", aspect=0.1)
#     hv.opts.Curve(backend="matplotlib", linewidth=1, color="#888888", linestyle="dotted"),
#     hv.opts.Scatter(backend="matplotlib", s=10),
#     hv.opts.Scatter(color=colors.calib_curves)
# )

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{admonition} Hint: Diagnosing calibration curves
# :class: hint dropdown
#
# Flat calibration curve
# ~ This is the most critical issue, since it indicates that $\Bemd{}$ is actually not predictive of $\Bconf{}$ at all. The most common reason is a mistake in the definition of the calibration distribution, where some values are fixed when they shouldn’t be.
#   - Remember that the model construction pipeline used on the real data needs to be repeated in full for each experimental condition produced by the calibration distribution. For example, in `Experiment` above we refit the observation noise $σ$ for each experimental condition generated within `__iter__`.
#   - Treat any global used within `EpistemicDist` with particular suspicion, as they are likely to fix values which should be variable.
#     To minimize the risk of accidental global variables, you can define `EpistemicDist` in its own separate module.
# ~ To help investigate issues, it is often helpful to reconstruct conditions that produce the unexpected behaviour. The following code snippet recovers the first calibration dataset for which both `Bemd > 0.9` and `Bconf = False`; the recovered dataset is `D`:
#   ```python
#   Bemd = calib_results[1.0]["Bemd"]
#   Bconf = calib_results[1.0]["Bconf"]
#   i = next(iter(i for i in range(len(Bemd)) if Bemd[i] > 0.9))
#     
#   for j, D in zip(range(i+1), task.models_Qs):
#       pass
#   assert j == i
#   ```
#
# Calibration curve with shortened domain
# ~ I.e. $\Bemd{}$ values don’t reach 0 and/or 1. This is not necessarily critical: the calibration distribution we want to test may simply not allow to fully distinguish the candidate models under any condition. 
# ~ If it is acceptable to change the calibration distribution (or to add one to the test suite), then the most common way to address this is to ensure the distribution produces conditions where $\Bemd{}$ can achieve maximum confidence – for example by having conditions with negligeable observation noise.
# :::

# %%
data = [[key[0], key[1], *(spearmanr(curve.data).statistic for curve in ov.Curve)]
        for key, ov in fig.main.items()]
print(tabulate(data, headers=["candidates", "input", *_c_list]))

# %%
data = [[key[0], key[1], *(scipy.stats.pearsonr((df:=curve.data).Bemd, df.Bconf).statistic for curve in ov.Curve)]
        for key, ov in fig.main.items()]
print(tabulate(data, headers=["candidates", "input", *_c_list]))

# %% [markdown]
# (code_prinz-calib-main-text)=
# #### Calibration figure: main text

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "hide-input"]
# tasks_to_show = [tasks[key] for key in [
#     ('A vs B', 'Gaussian', 'low noise', 'short input correlations', 'weak input'),
#     ('A vs B', 'Gaussian', 'low noise', 'short input correlations', 'strong input'),
#     ('C vs D', 'Gaussian', 'low noise', 'short input correlations', 'weak input'),
#     ('C vs D', 'Gaussian', 'low noise', 'short input correlations', 'strong input'),
#     ('A vs D', 'Gaussian', 'low noise', 'short input correlations', 'weak input'),
#     ('A vs D', 'Gaussian', 'low noise', 'short input correlations', 'strong input'),
# ]]
# assert all(task.has_run for task in tasks_to_show), "Run the calibration tasks from the command line environment, using `smttask run`. Executing it as part of a Jupyter Book build would take a **long** time."

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# panels = [panel_calib_curve(task, c_list) for task in tasks_to_show]
# panels[0].opts(title="$\\mathcal{{M}}_C$ vs $\\mathcal{{M}}_D$")
# panels[2].opts(title="$\\mathcal{{M}}_A$ vs $\\mathcal{{M}}_B$")
# panels[4].opts(title="$\\mathcal{{M}}_A$ vs $\\mathcal{{M}}_D$")
# panels[0].opts(ylabel=r"weak $I_{\mathrm{ext}}$")
# panels[1].opts(ylabel=r"strong $I_{\mathrm{ext}}$")
# panels[0].opts(hv.opts.Overlay(legend_cols=5, legend_position="top_left", # Two placement options    
#                                 legend_opts={"columnspacing": 2.}))
# panels[4].opts(hv.opts.Overlay(legend_cols=1, legend_position="right"))    # for the shared legend
# for i in (1, 2, 3, 4, 5):
#     panels[i].opts(hv.opts.Overlay(show_legend=False))
# for i in (0, 1, 2, 4, 5):
#     panels[i].opts(xlabel="")
# hooks = {i: [viz.despine_hook, viz.set_xticks_hook([0, 0.5, 1]), viz.set_yticks_hook([0, 0.5, 1])] for i in range(6)}
# for i in (0, 1):
#     hooks[i].extend([viz.set_yticklabels_hook(["$0$", "$0.5$", "$1$"])])
# for i in (1, 3, 5):
#     hooks[i].extend([viz.set_xticklabels_hook(["$0$", "$0.5$", "$1$"])])
# for i, hook_lst in hooks.items():
#     panels[i].opts(hooks=hook_lst)
#     # panels[i].opts(hooks=[viz.set_yticks_hook([0, 0.5, 1]), viz.set_yticklabels_hook(["$0$", "$0.5$", "$1$"]), viz.despine_hook])
# for i in (0, 2, 4):
#     panels[i].opts(xaxis="bare")
# for i in (2, 3, 4, 5):
#     panels[i].opts(yaxis="bare")
# # for i in (1, 3, 5):
# #     panels[i].opts(hooks=[viz.set_xticks_hook([0, 0.5, 1]), viz.set_xticklabels_hook(["$0$", "$0.5$", "$1$"]), viz.despine_hook])
# fig_calibs = hv.Layout(panels)
# fig_calibs.opts(
#     hv.opts.Layout(backend="matplotlib", sublabel_format="", #sublabel_format="({alpha})",
#                    transpose=True,
#                    #fig_inches=0.4*config.figures.defaults.fig_inches,  # Each panel is 1/3 of column width. Natural width of plot is a bit more; we let LaTeX scale the image down a bit (otherwise we would need to tweak multiple values like font scales & panel spacings)
#                    fig_inches=config.figures.defaults.fig_inches,
#                    hspace=-0.25, vspace=0.2, tight=False
#                   )
# ).cols(2)
# hv.output(fig_calibs)
#
# # Print panel descriptions
# from tabulate import tabulate
# headers = ["models", "input corr", "input strength", "obs noise", "obs dist"]
# data = [(f"Panel ({lbl})",
#          f"{(Ω:=task.experiments).a} vs {Ω.b}", f"{Ω.τ_dist}", f"{Ω.σi_dist}", f"{Ω.σo_dist}", f"{Ω.ξ_name} noise")
#         for lbl, task in zip("abcdef", tasks_to_show)]
# print(tabulate(data, headers, tablefmt="simple_outline"))

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# viz.save(fig_calibs, config.paths.figures/"prinz_calibrations_raw.svg")
# #viz.save(fig_calib.opts(fig_inches=5.5/3, backend="matplotlib", clone=True),
# #         config.paths.figures/"prinz_calibrations_html_raw.svg")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Finalized with Inkscape:
# - Put the curves corresponding to `c_chosen` on top. Highlight curve with white surround (~2x curve width).
# - Move legend above plots
# - Save to PDF

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Calibration figure: Supplementary

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# fig = hv.Layout([hv.Text(0, 0, f"{σo_dist}\n{ξ_name}\n\n{σi_dist}\n{τ_dist}")
#                  for σi_dist in ["weak input", "strong input"]
#                  for τ_dist in ["short input correlations", "long input correlations"]
#                  for σo_dist in ["low noise", "high noise"]
#                  for ξ_name in ["Gaussian", "Cauchy"]])
# fig.opts(
#     hv.opts.Layout(sublabel_format="", tight=True),
#     hv.opts.Overlay(show_legend=False),
#     hv.opts.Curve(hooks=[viz.noaxis_hook])
# )

# %% [markdown]
# $\mathcal{M}_A$ vs $\mathcal{M}_B$

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# a, b = ("A", "B")#, ("A", "D"), ("C", "D")
# fig = hv.Layout([panel_calib_curve(tasks[(f"{a} vs {b}", ξ_name, σo_dist, τ_dist, σi_dist)], c_list)
#                  for σi_dist in ["weak input", "strong input"]
#                  for τ_dist in ["short input correlations", "long input correlations"]
#                  for σo_dist in ["low noise", "high noise"]
#                  for ξ_name in ["Gaussian", "Cauchy"]])
# fig.opts(
#     hv.opts.Layout(sublabel_format="", fig_inches=1.1, hspace=.1, vspace=.1),
#     hv.opts.Overlay(show_legend=False),
#     hv.opts.Curve(hooks=[viz.noaxis_hook], linewidth=2)
# )

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# viz.save(fig, config.paths.figures/f"prinz_calibrations_all_{a}vs{b}_raw.svg")

# %% [markdown]
# $\mathcal{M}_A$ vs $\mathcal{M}_D$

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# a, b = ("A", "D")#, ("C", "D")
# fig = hv.Layout([panel_calib_curve(tasks[(f"{a} vs {b}", ξ_name, σo_dist, τ_dist, σi_dist)], c_list)
#                  for σi_dist in ["weak input", "strong input"]
#                  for τ_dist in ["short input correlations", "long input correlations"]
#                  for σo_dist in ["low noise", "high noise"]
#                  for ξ_name in ["Gaussian", "Cauchy"]])
# fig.opts(
#     hv.opts.Layout(sublabel_format="", fig_inches=1.1, hspace=.1, vspace=.1),
#     hv.opts.Overlay(show_legend=False),
#     hv.opts.Curve(hooks=[viz.noaxis_hook], linewidth=2)
# )

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# viz.save(fig, config.paths.figures/f"prinz_calibrations_all_{a}vs{b}_raw.svg")

# %% [markdown]
# $\mathcal{M}_C$ vs $\mathcal{M}_D$

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# a, b = ("C", "D")
# fig = hv.Layout([panel_calib_curve(tasks[(f"{a} vs {b}", ξ_name, σo_dist, τ_dist, σi_dist)], c_list)
#                  for σi_dist in ["weak input", "strong input"]
#                  for τ_dist in ["short input correlations", "long input correlations"]
#                  for σo_dist in ["low noise", "high noise"]
#                  for ξ_name in ["Gaussian", "Cauchy"]])
# fig.opts(
#     hv.opts.Layout(sublabel_format="", fig_inches=1.1, hspace=.1, vspace=.1),
#     hv.opts.Overlay(show_legend=False),
#     hv.opts.Curve(hooks=[viz.noaxis_hook], linewidth=2)
# )

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# viz.save(fig, config.paths.figures/f"prinz_calibrations_all_{a}vs{b}_raw.svg")

# %% [markdown]
# Finalize in Inkscape:
# - Add the axes (in the same style as `GridSpace`, but GridSpace only supports one dimension per axis)
# - Add a title: $\mathcal{M}_A \text{ vs } \mathcal{M}_B$

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Diagnostics: Exploring calibration experiments

# %% editable=true slideshow={"slide_type": ""}
iω_default = 21  # One of the experiments which gives unexpected results at large c


# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# tables = {}
# for c, bins in calib_plot.bin_idcs.items():
#     tab = hv.Table({str(i_bin): sorted(i_ω_lst) for i_bin, i_ω_lst in enumerate(bins)},
#                    kdims=[str(i) for i in range(len(bins))])
#     tab.opts(title="Experiments allocated to each Bemd bin", max_rows=64, fontsize={"text": 10, "title": 15}, backend="matplotlib") \
#        .opts(title="Experiments allocated to each Bemd bin", fontsize={"title": 15}, backend="bokeh") \
#        .opts(fig_inches=8, backend="matplotlib") \
#        .opts(width=700, height=450, backend="bokeh")
#     tables[c] = tab
# table_fig = hv.HoloMap(tables, kdims=viz.dims.bokeh.c)

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# hv.output(table_fig, backend="bokeh")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{admonition} The order of $\Bemd{}$ is not preserved when we change $c$
# :class: important
#
# One might expect the order of experiments to be preserved when we change $c$: i.e. that even though the values of $\Bemd{}$ change, the experiments with the largest $\Bemd{}$ are always the same.
#
# This is in fact not the case; in fact, the rightmost bin (bin #15) in the $c=2^4$ condition shares none of its experiments with the other $c$ conditions. Only about half are shared among all four other conditions.
# :::
#
# In the display below, each rectangle corresponds to an experiment; we show only experiments used on one of the 16 $\Bemd{}$ bins. (In other words, we collect all possible numbers in *one column* from the table above, across all values of $c$.)  When a rectangle is black, it means that for that choice of $c$, that experiment was assigned to the chosen $\Bemd{}$ bin. Vertically aligned columns correspond to the same experiment.

# %% [markdown]
# :::{hint}
# To interact with the interactive figures below, please download the notebook and open it in Jupyter. The interactivity will not work in a browser.
# :::

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-input"]
# import panel as pn
# import functools

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "hide-input"]
# def viz_bin_allocation(i_bin):
#     ω_lst = []
#     lines = []
#     # First iteration: collect all ω indices
#     for bins in calib_plot.bin_idcs.values():
#         ω_lst.extend(ω for ω in bins[i_bin] if ω not in ω_lst)
#     # Second iteration: For each c, iterate over ω indices and draw black or white rect
#     for c, bins in calib_plot.bin_idcs.items():
#         bin_ωs = set(bins[i_bin])
#         rects = "".join('▮' if ω in bin_ωs else '▯' for ω in ω_lst)
#         lines.append(f"c={viz.format_pow2(c):<4}\t" + rects)
#     return "\n".join(lines)

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
# bin_slider = pn.widgets.IntSlider(name="Bin index", start=0, end=len(next(iter(calib_plot.bin_idcs.values())))-1,
#                                   value=15)
# text_widget = pn.pane.HTML("<pre>"+viz_bin_allocation(15)+"</pre>")
#
# def callback(target, event):
#     target.object = "<pre>"+viz_bin_allocation(event.new)+"</pre>"
# bin_slider.link(text_widget, callbacks={"value": callback})
#
# pn.Column(bin_slider, text_widget)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Below we plot the quantile curves for each individual experiment; experiment indices are the same as in the table above. Dashed lines correspond to synthetic PPFs, solid lines to mixed PPFs.
# (Models need to be reintegrated for each experiment, so one needs to wait a few seconds after changing experiments for the plot to update. A minimal amount of caching is used, so going back to the previous value is fast.)

# %% editable=true slideshow={"slide_type": ""}
def get_color(a):
    try:
        i = {"A": 0, "B": 1, "C": 2, "D": 3}[a]
    except KeyError:
        i = {"LP 2": 0, "LP 3": 1, "LP 4": 2, "LP 5": 3}[a]
    return colors.LP_candidates.values[i]


# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "hide-input"]
# @lru_cache(maxsize=4) # Because Experiment.dataset caches data, we keep this small
# def get_ppfs(ω: Experiment):
#     data = ω.dataset.get_data()
#     mixed_ppf = Dict({
#         ω.a: emd.make_empirical_risk_ppf(ω.QA(data)),
#         ω.b: emd.make_empirical_risk_ppf(ω.QB(data))
#     })
#     synth_ppf = Dict({
#         ω.a: emd.make_empirical_risk_ppf(ω.QA(ω.candidateA(data))),
#         ω.b: emd.make_empirical_risk_ppf(ω.QB(ω.candidateB(data)))
#     })
#     return mixed_ppf, synth_ppf
#
# def ppfs_for_experiment(ω: Experiment, show_data_model_text=True,
#                         backend:Literal["bokeh","matplotlib"]="bokeh"):
#     # For diagnostic plots we typically use bokeh, which has better interactivity
#     Φarr = np.linspace(0, 1, 1024)
#     dims = viz.dims[backend]
#     mixed_ppf, synth_ppf = get_ppfs(ω)
#     curve_mixed_a = hv.Curve(zip(Φarr, mixed_ppf[ω.a](Φarr)), kdims=dims.Φ, vdims=dims.q,
#                              group="mixed", label=ω.a)
#     curve_mixed_b = hv.Curve(zip(Φarr, mixed_ppf[ω.b](Φarr)), kdims=dims.Φ, vdims=dims.q,
#                              group="mixed", label=ω.b)
#     curve_synth_a = hv.Curve(zip(Φarr, synth_ppf[ω.a](Φarr)), kdims=dims.Φ, vdims=dims.q,
#                              group="synth", label=ω.a)
#     curve_synth_b = hv.Curve(zip(Φarr, synth_ppf[ω.b](Φarr)), kdims=dims.Φ, vdims=dims.q,
#                              group="synth", label=ω.b)
#     fig = curve_synth_a * curve_synth_b * curve_mixed_a * curve_mixed_b
#     if show_data_model_text:
#         fig *= hv.Text(0.05, 30, f"Data generated with: {get_data_model_label(ω)}", halign="left")
#     fig.opts(
#         hv.opts.Overlay(width=500, height=400, aspect=None, backend="bokeh"),
#         hv.opts.Curve("Synth", linestyle="dashed", backend="matplotlib"),
#         hv.opts.Curve("Synth", line_dash="dashed", backend="bokeh"),
#         *[hv.opts.Curve(f"{ppftype}.{a}", color=get_color(a), backend=backend)
#           for ppftype in ("Synth", "Mixed") for backend in ("bokeh", "matplotlib")
#           for a in (ω.a, ω.b)]
#     )
#     return fig

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-input"]
# def get_data_model_label(ω):
#     return {"LP 2": "A", "LP 3": "B", "LP 4": "C", "LP 5": "D"}[ω.dataset.LP_model]
# def ppf_callback(i_ω):
#     ω = Ω[int(i_ω)]
#     curves = ppfs_for_experiment(ω)#.opts(title=f"Experiment index: {i_ω} – Data generated with: {get_data_model_label(ω)}")
#     return curves
# dmap = hv.DynamicMap(ppf_callback, kdims=[hv.Dimension("iω", label="Experiment index")])
# dmap = dmap.redim.values(iω=[str(i) for i in range(len(Ω))])
# dmap = dmap.redim.default(iω=str(iω_default))
# hv.output(dmap, backend="bokeh", widget_location="right")

# %% [markdown]
# ## What happens when $c$ is too large ?

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# ω = Ω[iω_default]
# rng = utils.get_rng("prinz", "c too large", "qpaths")
# mixed_ppf, synth_ppf = get_ppfs(ω)
#
# panels = []
# for c in [2**-4, 2**0, 2**4]:
#     qhat_curves = {}
#     for a in [ω.a, ω.b]:
#         def δemd(Φarr): return abs(synth_ppf[a](Φarr) - mixed_ppf[a](Φarr))
#         qpaths = emd.path_sampling.generate_quantile_paths(mixed_ppf[a], δemd, c=c, M=6, res=10, rng=rng)
#     
#         qhat_curves[a] = [hv.Curve(zip(Φhat, qhat), group="sampled q", label=a,
#                                    kdims=[dims.Φ], vdims=[dims.q])
#                           .opts(color=get_color(a), backend="matplotlib")
#                           .opts(color=get_color(a), backend="bokeh")
#                           for Φhat, qhat in qpaths]
#
#     panel = (ppfs_for_experiment(ω, show_data_model_text=False, backend="matplotlib")
#              * hv.Overlay(qhat_curves[ω.a]) * hv.Overlay(qhat_curves[ω.b]))
#     title = f"$c={viz.format_pow2(c, 'latex')}$".replace("{", "{{").replace("}", "}}")
#     panel.opts(hv.opts.Curve("Sampled_q", alpha=0.375),
#                hv.opts.Overlay(fig_inches=config.figures.matplotlib.defaults.fig_inches,
#                                title=title)
#               )
#     panels.append(panel)

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# panels[0].opts(           hooks=[viz.set_ylim_hook(2.5, 8), viz.set_xticks_hook([0, .5, 1], labels=["0", "0.5", "1"])])
# panels[1].opts(xlabel="", hooks=[viz.set_ylim_hook(2.5, 8), viz.set_xticks_hook([0, .5, 1], labels=["0", "0.5", "1"]), viz.yaxis_off_hook], show_legend=False)
# panels[2].opts(xlabel="", hooks=[viz.set_ylim_hook(2.5, 8), viz.set_xticks_hook([0, .5, 1], labels=["0", "0.5", "1"]), viz.yaxis_off_hook], show_legend=False)
# fig = hv.Layout(panels).cols(3)
# fig = fig.redim.range(q=(2.5, 8))
# fig.opts(
#     hv.opts.Layout(sublabel_format="", hspace=0.15, fontscale=1.3,
#                    fig_inches=config.figures.matplotlib.defaults.fig_inches/3),
#     hv.opts.Overlay(legend_cols=1)
# )
# fig

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{admonition} Conclusion
# :class: important
#
# There is a upper bound on the value of $c$ we can choose: too large $c$ causes the quantile curves to constantly hit upon the monotonicity constraint, distorting the distribution.
#
# Hypothesis: The iterative nature of the hierarchical beta process may make this worse. A process based on a Dirichlet (rather than beta) distribution, where all increments are sampled at once, may mitigate this.
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# (code_prinz-model-comparison)=
# ## EMD model comparison
#
# Based on the calibration results, we choose the value $c=${glue:text}`c_chosen_prinz` (set above) to compute the $\Bemd{}$ criterion between models.
#
# First we recreate `synth_ppf` and `mixed_ppf` as we did above.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# synth_ppf = Dict({
#     a: emd.make_empirical_risk_ppf(Qrisk[a](generate_synth_samples(candidate_models[a])))
#     for a in "ABCD"
# })
# mixed_ppf = Dict({
#     a: emd.make_empirical_risk_ppf(Qrisk[a](LP_data.get_data()))
#     for a in "ABCD"
# })

# %% [markdown]
# Sample of a set of expected risks ($R$) for each candidate model.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# R_samples = Dict(
#     A = emd.draw_R_samples(mixed_ppf.A, synth_ppf.A, c=c_chosen),
#     B = emd.draw_R_samples(mixed_ppf.B, synth_ppf.B, c=c_chosen),
#     C = emd.draw_R_samples(mixed_ppf.C, synth_ppf.C, c=c_chosen),
#     D = emd.draw_R_samples(mixed_ppf.D, synth_ppf.D, c=c_chosen)
# )

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Convert the samples into a distributions using a kernel density estimate (KDE).

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# xticks = [3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7]
# fig_Rdists = viz.make_Rdist_fig(
#     R_samples,
#     colors     =colors.LP_candidates,
#     xticks     =xticks,
#     xticklabels=["", "", "4.0", "" ,"", "", "", "", "4.6", ""],
#     yticks     =[0, 2, 4, 6],
#     yticklabels=["0", "", "", "6"],
# )
# fig_Rdists

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
#     Rdists = [hv.Distribution(_Rlst, kdims=[dims.R], label=f"Model {a}")
#               for a, _Rlst in R_samples.items()]
#     Rcurves = [hv.operation.stats.univariate_kde(dist).to.curve()
#                for dist in Rdists]
#     fig_Rdists = hv.Overlay(Rdists) * hv.Overlay(Rcurves)
#     
#     xticks = [round(R,1) for R in np.arange(3, 5, 0.1) if (low:=fig_Rdists.range("R")[0]) < R < (high:=fig_Rdists.range("R")[1])]
#     xticklabels = [str(R) if R in (4.0, 4.6) else "" for R in xticks]

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input", "active-ipynb", "remove-cell"]
#     # Plot styling
#     yticks = [0, 2, 4, 6]
#     yticklabels = ["0", "", "", "6"]
#     fig_Rdists.opts(
#         hv.opts.Distribution(alpha=.3),
#         hv.opts.Distribution(facecolor=colors.LP_candidates, color="none", edgecolor="none", backend="matplotlib"),
#         hv.opts.Curve(color=colors.LP_candidates),
#         hv.opts.Curve(linestyle="solid", backend="matplotlib"),
#         hv.opts.Overlay(backend="matplotlib", fontscale=1.3,
#                         hooks=[viz.set_xticks_hook(xticks), viz.set_xticklabels_hook(xticklabels), viz.ylabel_shift_hook(5),
#                                viz.set_yticks_hook(yticks), viz.set_yticklabels_hook(yticklabels), viz.xlabel_shift_hook(7),
#                                viz.despine_hook(2)],
#                         legend_position="top_left", legend_cols=1,
#                         show_legend=False,
#                         xlim=fig_Rdists.range("R"),  # Redundant, but ensures range is not changed
#                         #fig_inches=config.figures.defaults.fig_inches)  # Previously: was 1/3 full width
#                         aspect=3
#                        )
#     )

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "hide-input"]
# # First line: means of the R f
# # Second line: expected risk computed on the original data
# #Rmeans = hv.Spikes([(Rlst.mean(), 2, a) for a, Rlst in R_samples.items()],
# Rmeans = hv.Spikes([(Qrisk[a](LP_data.get_data()).mean(), 2, a) for a in R_samples.keys()],
#                      kdims=dims.R, vdims=["height", "model"], group="back")
# # Display options
# dashstyles = {"A": (0, (4, 4)), "B": (4, (4, 4)),
#               "C": (0, (3.5, 4.5)), "D": (4, (3.5, 4.5))}
# model_colors = {a: c for a, c in zip("ABCD", colors.LP_candidates.values)}
#
# # Because the C and D models are so close, the lines are very difficult to differentiate
# # To make this easier, we overlay with interleaved dashed lines.
# # NB: Key to making this visually appealing is that we leave a gap between
# #     the dash segments
# Rmeans_front = hv.Overlay([
#     hv.Spikes([(R, h, a)], kdims=Rmeans.kdims, vdims=Rmeans.vdims,
#               group="front", label=f"Model {a}")
#     .opts(backend="matplotlib", linestyle=dashstyles[a])
#     for R, h, a in Rmeans.data.values])
# # NB: Current versions don’t seem to include Spikes in the legend.
# #     Moreover, the shifted dash style means that for B and D the line is not printed in the legend
# legend_proxies = hv.Overlay([hv.Curve([(R, 0, a)], group="proxy", label=f"Model {a}")
#                              for R, h, a in Rmeans.data.values])
# fig_Rmeans = Rmeans * Rmeans_front * legend_proxies
#
# fig_Rmeans.opts(
#     hv.opts.Spikes(color=hv.dim("model", lambda alst: np.array([model_colors[a] for a in alst]))),
#     hv.opts.Spikes("back", alpha=0.5),
#     hv.opts.Spikes(backend="matplotlib", linewidth=2, hooks=[viz.yaxis_off_hook]),
#     hv.opts.Overlay(backend="matplotlib",
#                     show_legend=True, legend_position="bottom_left",
#                     xticks=xticks,
#                     hooks=[viz.set_xticklabels_hook(""), viz.despine_hook(0), viz.yaxis_off_hook],
#                     xlim=fig_Rdists.range("R"),
#                     aspect=6)
# )

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "hide-input"]
# # Why aren’t the fontsizes consistent across panels ? No idea...
# fig = (fig_Rmeans.opts(fontscale=1.3, sublabel_position=(-.25, .4), show_legend=False, xlabel="")
#        + fig_Rdists.opts(sublabel_position=(-.25, .9), show_legend=True))
# fig.opts(shared_axes=True, tight=False, aspect_weight=True,
#          sublabel_format="", sublabel_size=12,
#          vspace=0.1,
#          fig_inches=config.figures.defaults.fig_inches)
# fig.cols(1)

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell", "active-ipynb"]
# viz.save(fig, config.paths.figures/f"prinz_Rdists_raw.svg")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Things to finish in Inkscape:
# - ~~Fix alignment of sublabels~~
# - ~~Make sublabels bold~~
# - Trim unfinished dashed lines in the R means
# - Extend lines for R means into lower panel
# - Confirm that alignment of x axes pixel perfect
# - ~~Tighten placement of xlabel~~
# - Save to PDF

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# Bigger version; appropriate for HTML or slides.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
# vlines = hv.Overlay([hv.VLine(R).opts(color=model_colors[a])
#                      for R, _, a in fig_Rmeans.data['Back', 'I'].data.values])
# fig2 = (fig_Rmeans.opts(clone=True, fontscale=2, show_legend=False, sublabel_position=(-.215, .4))
#                   .opts(hv.opts.Spikes(linewidth=4))
#        + (vlines*fig_Rdists).opts(clone=True, fontscale=2, show_legend=True, sublabel_position=(-.25, .9))
#                             .opts(hv.opts.VLine(linewidth=4, alpha=0.5)))
# fig2[1].opts(fig_Rdists.opts.get())
# fig2[1].opts(show_legend=False)
# fig2[0].opts(show_legend=True)
# fig2[0].opts(backend="matplotlib", legend_position='top_left',
#              legend_opts={'framealpha': 1, 'borderpad': 1,
#                           'labelspacing': .5,
#                           'bbox_to_anchor': (-.02, 1)})  # NB: 'loc' is ignored; use legend_position
# fig2.opts(shared_axes=True, tight=False, aspect_weight=True,
#           sublabel_format="", sublabel_size=18,
#           vspace=0,  # For some reason, negative vspace doesn’t work
#           fig_inches=5.5)
# viz.save(fig2, config.paths.figures/f"prinz_Rdists_big.svg")
# fig2.cols(1)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# EMD estimates for the probabilities $P(R_a < R_b)$ are nicely summarized in a table:

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# df = emd.utils.compare_matrix(R_samples)

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "hide-input"]
# df.index = pd.MultiIndex.from_tuples([("a", a) for a in df.index])
# df.columns = pd.MultiIndex.from_tuples([("b", b) for b in df.columns])
# df.style.format(precision=3)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# For pasting into a MyST file, we reformat as a [list-table](https://jupyterbook.org/en/stable/reference/cheatsheet.html?highlight=list-table#tables):

# %% editable=true slideshow={"slide_type": ""} tags=["skip-execution", "remove-cell", "active-ipynb"]
# print(":::{list-table}")
# model_labels = list(R_samples)
# print(":header-rows: 1")
# print(":stub-columns: 1")
# print("")
#
# print(f"* - ")
# for a in model_labels:
#     print(f"  - {a}")
# for a in model_labels:
#     print(f"* - {a}")
#     for Pab in df.loc[("a", a),:]:
#         print(f"  - {Pab:.3f}")
# print(":::")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{admonition} `compare_matrix` implementation
# :class: note dropdown
#
# The `compare_matrix` function provided by `emdcmp` simply loops through all $(a,b)$ model pairs, and counts the number of $R_a$ samples which are larger than $R_b$:
#
# ```python
# def compare_matrix(R_samples: Dict[str, ArrayLike]) -> pd.DataFrame:
#     R_keys = list(R_samples)
#     compare_data = {k: {} for k in R_keys}
#     for i, a in enumerate(R_keys):
#         for j, b in enumerate(R_keys):
#             if i == j:
#                 assert a == b
#                 compare_data[b][a] = 0.5
#             elif j < i:
#                 compare_data[b][a] = 1 - compare_data[a][b]
#             else:
#                 compare_data[b][a] = np.less.outer(R_samples[a], R_samples[b]).mean()
#     return pd.DataFrame(compare_data)
# ```
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# ## Exported notebook variables
#
# These can be inserted into other pages.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
# glue("AB_model", AB_model_label, display=True)
# glue("c_chosen_prinz", c_chosen, raw_myst=viz.format_pow2(c_chosen, format='latex'), raw_latex=viz.format_pow2(c_chosen, format='latex'))
#
# glue("N", N, display=True)
# glue("Linf", viz.format_pow2(Linf), display=True)
# glue("Ldata", L_data, display=True)
#
# #glue("noise_model", Ω.ξ_dist, display=True)
# #glue("noise_bias", Ω.μ_dist, display=True)
# #glue("noise_width", Ω.σ_dist, display=True)
#
# glue("calib_curve_palette", config.emd.viz.matplotlib.calibration_curves["color"].key, display=True)
#
# # Epistemic hyperparameters
# glue("tau_short_min"   , **viz.formatted_quantity(EpistemicDist.τ_short_min ))
# glue("tau_short_max"   , **viz.formatted_quantity(EpistemicDist.τ_short_max ))
# glue("tau_long_min"    , **viz.formatted_quantity(EpistemicDist.τ_long_min  ))
# glue("tau_long_max"    , **viz.formatted_quantity(EpistemicDist.τ_long_max  ))
# glue("sigmao_low_mean" , **viz.formatted_quantity(EpistemicDist.σo_low_mean ))
# glue("sigmao_high_mean", **viz.formatted_quantity(EpistemicDist.σo_high_mean))
# glue("sigmao_std"      , **viz.formatted_quantity(EpistemicDist.σo_std      ))
# glue("sigmai_weak_mean", **viz.formatted_quantity(EpistemicDist.σi_weak_mean ))
# glue("sigmai_strong_mean", **viz.formatted_quantity(EpistemicDist.σi_strong_mean))
# glue("sigmai_std"      , **viz.formatted_quantity(EpistemicDist.σi_std      ))

# %% [markdown]
#     'AB/PD 3'
#     0.25
#     512
#     '2¹⁴'
#     4000
#     'copper'
#     '0.1 \\mathrm{ms}'
#     '0.2 \\mathrm{ms}'
#     '1.0 \\mathrm{ms}'
#     '2.0 \\mathrm{ms}'
#     '0.0 \\mathrm{mV}'
#     '1.0 \\mathrm{mV}'
#     '0.5 \\mathrm{mV}'
#     '-15.0 \\mathrm{mV}'
#     '-10.0 \\mathrm{mV}'
#     '0.5 \\mathrm{mV}'

# %% editable=true slideshow={"slide_type": ""} tags=["remove-input"]
emd.utils.GitSHA()

# %% editable=true slideshow={"slide_type": ""}
