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
#     '\RR'   : '\mathbb{R}'
#     '\nN'   : '\mathcal{N}'
#     '\Bemd' : 'B^{\mathrm{EMD}}_{#1}'
#     '\Bconf': 'B^{\mathrm{epis}}_{#1}'
#     '\BQ' : 'B^{Q}_{#1}'
#     '\Unif' : '\operatorname{Unif}'
#     '\Mtrue': '\mathcal{M}_{\mathrm{true}}'
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# (code_calibrating-with-BQ)=
# # Rejection probability is not predicted by loss distribution
#
# {{ prolog }}
#
# %{{ startpreamble }}
# %{{ endpreamble }}

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# > **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).
#

# %% [markdown] editable=true slideshow={"slide_type": ""}
# This notebook explores a similar question as [](#code_aleatoric-cannot-substitute-epistemic), namely why we cannot use the loss distribution directly and instead really need to estimate the epistemic distribution.
#
# Here we explore the question in the context of our calibration experiments. Ultimately the goal is to predict the probability that a replication of the experiment would select model $A$ over model $B$, and this is exactly what our calibration experiments test (with synthetic replications). So if we can replace $\Bemd{}$ by a hypothetical $\BQ{}$ defined solely in terms of the loss distribution and still find a correlation with $\Bconf{}$, that would suggest that we don’t really need $\Bemd{}$ (which depends on a more complicated distribution and ad hoc parameter $c$) – we could get away with considering just the loss distribution, which is much simpler and therefore more appealing.
#
# Of course, there is no reason to expect that the loss distribution (which is a result of aleatoric uncertainty) will be related to *replication* uncertainty – indeed, that is the point of the [aforementioned notebook](#code_aleatoric-cannot-substitute-epistemic). But the question gets asked, so in addition to arguing on the basis of principle, it is good to tackle the question head-on, in the most direct way possible, with an actual experiment.
#
# One challenge is how best to define $B^Q$: since it measures aleatoric instead of replication/epistemic uncertainty, there is no ontologically obvious way to do this. So instead we will use the _mathematically_ obvious way[^arbitrary-functional] and “hope for the best”:[^ml-research] we match the form of [](#eq_Bemd_def), replacing the $R$-distribution by the $Q$-distribution:
#
# $$
# \BQ{AB;c_Q} &:= P(Q_A < Q_B + η)\,, \\
# η &\sim \nN(0, c_Q^2) \,.
# $$ 
#
# :::{margin}
# This is given {eq}`eq_BQ_def` in the [supplementary material](#sec_why-not-BQ).
# :::
#
# Note that now the probability now is over the samples of $\Mtrue$ rather than over replications.
# The parameter $c_Q$ is an additional degree of freedom; it effectively convolves the $Q$ distributions with a Gaussian, thus increasing their spread and, (similar to the effect of $c$ on $\Bemd{}$) decreasing the sensitivity of $\BQ{}$.
# We include $η$ so that both $\BQ{}$ and $\Bemd{}$ have the same number of free parameters, and thus make the comparison more fair. We will select the $c_Q$ which maximizes the correlation with $\Bconf{}$, thus giving $\BQ{}$ the best chance to match the performance of $\Bemd{}$.
#
# The advantage of [](#eq_BQ_def) is that it is easily estimated from observed samples, with no need to define an ad hoc distribution over quantile functions.
# The disadvantage is that it replaces an ad hoc distribution by an ad hoc ontology, since there is no clear link between the distribution of $Q$ values and the ability of a model to replicate.
# (The two are not unrelated, but the relation is highly confounded by the aleatoric uncertainty.)
# As we will see below, this makes it a poor predictor for $\Bconf{}$.
#
# Note also that appropriate values of $c_Q$ are entirely dependent on what the typical values of the loss are.
# If the loss is given in terms of a probability density, rather than a probability mass, then this is not scale independent.
# There is even less reason to expect the values of $c_Q$ therefore to generalize to other problems, in contrast to what we observed with $\Bemd{}$.
#
# [^arbitrary-functional]: A direct comparison of $Q_A$ and $Q_B$ may be ad hoc, but it is difficult to imagine a better criterion which doesn’t also introduce more free parameters. We can of course posit that there exists _some_ functional $B(Q_A,Q_B) \to \RR$ which correlates with $\Bconf{}$, but then the $\Bemd{}$ is subsumed in that hypothesis – with the difference that $\BQ{}$ has infinitely many free parameters, while $\Bemd{}$ has only the scalar parameter $c$.
# For this reason we think that $\BQ{}$ as defined in [](#eq_BQ_def) is the fairest expression of a comparison criterion which uses $Q_A$ and $Q_B$ directly, with no further assumptions.
#
# [^ml-research]: Incidentally, this “match the form and hope for the best” is not uncommon in the machine learning literature…

# %% [markdown]
# ## Imports

# %%
import numpy as np

# %%
from functools import partial
from itertools import chain

# %%
import holoviews as hv
hv.extension("matplotlib", "bokeh")

# %% [markdown]
# `emdcmp` library

# %%
import emdcmp

# %% [markdown]
# Project code

# %%
from config import config
import viz
import viz.emdcmp   # Backported updates to emdcmp.viz

# %% [markdown]
# Load the epistemic dists used in the calibration experiments

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-input"]
# from Ex_Prinz2004 import EpistemicDist, L_data, Linf, colors, dims

# %% [markdown]
# Load the [customized calibration task](./task_bq.md) which computes $\BQ{}$ instead of $\Bemd{}$.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-input"]
# from task_bq import CalibrateBQ, calib_point_dtype

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Execution

# %% [markdown]
# ::::{admonition} How many experiments
# :class: hint margin
#
# The function `emd.viz.calibration_plot` attempts to collect results into 16 bins,[^with-no-less-than-16-points-per-bin] so making $N$ a multiple of 16 works nicely.
#
# :::{dropdown} Numbers that worked well for us
# For an initial pilot run, we found $N=64$ or $N=128$ to be good numbers. These numbers produce respectively 4 or 8 bins, which is often enough to check that $\Bemd{}$ and $\Bconf{}$ are reasonably distributed and that the epistemic distribution is actually probing the transition from strong to equivocal evidence.
# A subsequent run with $N \in \{256, 512, 1024, 2048, 4096\}$ can then refine and smooth the curve.
# :::
# ::::
#
# [^with-no-less-than-16-points-per-bin]: With the constraint that no bin can have less than 16 points.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# We only run the experiments on the 6 epistemic distributions used in the main paper.

# %% editable=true slideshow={"slide_type": ""}
#N = 64
#N = 128
#N = 512
N = 2048
Ωdct = {(f"{Ω.a} vs {Ω.b}", Ω.ξ_name, Ω.σo_dist, Ω.τ_dist, Ω.σi_dist): Ω
        for Ω in [
            EpistemicDist(N, "C", "D", "Gaussian", "low noise", "short input correlations", "weak input"),
            EpistemicDist(N, "A", "B", "Gaussian", "low noise", "short input correlations", "weak input"),
            EpistemicDist(N, "A", "D", "Gaussian", "low noise", "short input correlations", "weak input"),
            EpistemicDist(N, "C", "D", "Gaussian", "low noise", "short input correlations", "strong input"),
            EpistemicDist(N, "A", "B", "Gaussian", "low noise", "short input correlations", "strong input"),
            EpistemicDist(N, "A", "D", "Gaussian", "low noise", "short input correlations", "strong input"),
        ]
       }

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{hint}
# :class: margin
# `Calibrate` will iterate over the data models twice, so it is important that the iterable passed as `data_models` not be consumable.
# :::

# %% editable=true slideshow={"slide_type": ""}
c_list = [2**-6, 2**-4, 2**-2, 2**0, 2**2, 2**4]
LQ = 4000 # Number of draws for the Monte Carlo estimate of normal dist addition

# %% [markdown]
#     cQ_chosen = -2**-3
#     cQ_list=[#-2**1,  2**1,
#              #-2**0,  2**0,
#              -2**-1, 2**-1,
#              #-2**-2, 2**-2,
#              -2**-3, 2**-3,
#              #0
#             ]

# %% editable=true slideshow={"slide_type": ""}
tasks_normal = {}
tasks_BQ = {}
for Ωkey, Ω in Ωdct.items():
    task = CalibrateBQ(
        reason = f"BQ calibration attempt – {Ω.a} vs {Ω.b} - {Ω.ξ_name} - {Ω.σo_dist} - {Ω.τ_dist} - {Ω.σi_dist} - {N=}",
        #c_list = cQ_list,
        c_list = c_list,
        experiments = Ω.generate(N),
        Ldata = L_data,
        Linf = Linf,
        LQ = LQ
    )
    tasks_BQ[Ωkey] = task

    task = emdcmp.tasks.Calibrate(
        reason = f"Prinz calibration – {Ω.a} vs {Ω.b} - {Ω.ξ_name} - {Ω.σo_dist} - {Ω.τ_dist} - {Ω.σi_dist} - {N=}",
        c_list = c_list,
        #c_list = [2**-6, 2**-4, 2**-2, 2**0, 2**4],
        experiments = Ω.generate(N),
        Ldata = L_data,
        Linf = Linf
    )
    tasks_normal[Ωkey] = task

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The code below creates task files for any missing tasks

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "skip-execution"]
#     for task in chain(tasks_normal.values(), tasks_BQ.values()):
#         if not task.has_run:  # Don’t create task files for tasks which have already run
#             Ω = task.experiments
#             taskfilename = f"prinz_{type(task).__qualname__}__{Ω.a}vs{Ω.b}_{Ω.ξ_name}_{Ω.σo_dist}_{Ω.τ_dist}_{Ω.σi_dist}_N={Ω.N}_c={task.c_list}"
#             task.save(taskfilename)

# %% [markdown]
# If any files were created, run those tasks from the command line with
#
#     smttask run -n1 --import config <task file>
#
# before continuing.

# %%
assert all(task.has_run for task in chain(tasks_normal.values(), tasks_BQ.values()))

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Comparison of $\Bemd{}$ and $\BQ{}$ calibration curves

# %% [markdown]
# :::{important} The basic principles for calibration
#
# Wide support
# ~ The goal of calibration is to probe that intermediate where selection of either model $A$ or $B$ is not certain.
#   It is important therefore that we obtain values $\Bemd{}$ over the entire interval $[0, 1]$.  
#   Whether this is the case will be a function of two things:
#   - the design of the design of the calibration experiments: whether it produces ambiguous selection problems;
#   - the choice of $c$: generally, a larger $c$ will concentrate $\Bemd{}$ towards 0.5, a smaller $c$ will concentrate them towards 0 and 1.
#
#   So we want the support of $\Bemd{}$ to be as large as possible.
#
# Flat distribution
# ~ As a secondary goal, we also want it to be as flat as possible, since this will lead to more efficient sampling: Since we need enough samples at every subinterval of $\Bemd{}$, it is the most sparsely sampled regions which determine how many calibration datasets we need to generate. (And therefore how long the computation needs to run.)
#
#   Beyond making for shorter compute times, a flat distribution however isn’t in and of itself a good thing: more important is that the criterion is able to resolve the models when it should.
# :::

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
calibs_normal = {key: viz.emdcmp.calibration_plot(task.unpack_results(task.run()))
                 for key, task in tasks_normal.items()}
calibs_BQ     = {key: viz.emdcmp.calibration_plot(task.unpack_results(task.run()))
                 for key, task in tasks_BQ.items()}


# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
def mkkey(taskkey): return taskkey[0], taskkey[4]
taskdims = ["models", "input"]
hm = hv.HoloMap({("EMD", "Bemd", c, *mkkey(taskkey)): hist for taskkey, calplot in calibs_normal.items() for c, hist in calplot.Bemd_hists.items()}
                | {("EMD", "Bepis", c, *mkkey(taskkey)): hist for taskkey, calplot in calibs_normal.items() for c, hist in calplot.Bepis_hists.items()}
                | {("Q", "Bemd", c, *mkkey(taskkey)): hist for taskkey, calplot in calibs_BQ.items() for c, hist in calplot.Bemd_hists.items()}
                | {("Q", "Bepis", c, *mkkey(taskkey)): hist for taskkey, calplot in calibs_BQ.items() for c, hist in calplot.Bepis_hists.items()},
                kdims=["uncertainty", "B", "c", *taskdims])
# NB: The set of c values is disjoint, so attempting to plot the two uncertainty dists together will lead to artifacts
hm.select(uncertainty="EMD").overlay("B").layout(taskdims).cols(3).opts(
    hv.opts.Layout(backend="matplotlib", fig_inches=3, tight=True),
    hv.opts.NdLayout(backend="matplotlib", fig_inches=3, tight=True),
    hv.opts.Histogram(backend="matplotlib", aspect=2)
)

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
hm.select(uncertainty="Q").overlay("B").layout(taskdims).cols(3).opts(
    hv.opts.Layout(backend="matplotlib", fig_inches=3, tight=True),
    hv.opts.NdLayout(backend="matplotlib", fig_inches=3, tight=True),
    hv.opts.Histogram(backend="matplotlib", aspect=2)
)

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
fig = hm.select(uncertainty="EMD", B="Bemd").drop_dimension(["uncertainty", "B"]).overlay("c") \
      + hm.select(uncertainty="Q", B="Bemd").drop_dimension(["uncertainty", "B"]).overlay("c")\
        .redim(Bemd="BQ").opts(title="BQ")
fig.opts(hv.opts.Layout(backend="matplotlib", fig_inches=2, sublabel_format=""),
         hv.opts.Histogram(backend="matplotlib", aspect=1.5),
         hv.opts.NdOverlay(legend_position="best")
        )

# %% editable=true slideshow={"slide_type": ""}
calibopts = (
    hv.opts.Overlay(backend="matplotlib",
                    #legend_cols=3,
                    #legend_opts={"columnspacing": .5, "alignment": "center",
                    #             "loc": "upper center"},
                    #hooks=[partial(viz.despine_hook(), left=False)],
                    #fig_inches=config.figures.defaults.fig_inches,
                    aspect=1, fontscale=1.3),
    hv.opts.Scatter(backend="matplotlib", s=20),
    #hv.opts.Scatter(color=hv.Palette("copper", range=(0., 1), reverse=True)),
    hv.opts.Layout(sublabel_format=""),
    hv.opts.Layout(backend="matplotlib", hspace=0.1, vspace=0.05,
                   fig_inches=0.65*config.figures.defaults.fig_inches)
)


# %% editable=true slideshow={"slide_type": ""}
def format_calib_curves(panels, tasks):
    assert len(panels) == 6
    top_row = panels[:3]
    bot_row = panels[3:]
    
    # Column titles
    pairs = [(task.experiments.a, task.experiments.b) for task in tasks]
    assert pairs[:3] == pairs[3:]
    pairs = pairs[:3]
    col_titles = [rf"$\mathcal{{{{M}}}}_{a}$ vs $\mathcal{{{{M}}}}_{b}$" for a,b in pairs]
    panels[0].opts(title=col_titles[0])
    panels[1].opts(title=col_titles[1])
    panels[2].opts(title=col_titles[2])

    # Row titles
    assert all(task.experiments.σi_dist == "weak input" for task in tasks[:3])
    assert all(task.experiments.σi_dist == "strong input" for task in tasks[3:])
    panels[0].opts(ylabel=r"weak $I_{\mathrm{ext}}$")
    panels[3].opts(ylabel=r"strong $I_{\mathrm{ext}}$")

    # Legend placement
    panels[0].opts(hv.opts.Overlay(legend_cols=5, legend_position="top_left", legend_opts={"columnspacing": 2.}))
    for i in (1, 2, 3, 4, 5):
        panels[i].opts(hv.opts.Overlay(show_legend=False))

    # xlabel only on the centred bottom panel
    for i in (0, 1, 2, 3, 5):
        panels[i].opts(xlabel="")

    # Despine, set axis ticks, display labels only on outer panels
    hooks = {i: [viz.despine_hook, viz.set_xticks_hook([0, 0.5, 1]), viz.set_yticks_hook([0, 0.5, 1])] for i in range(6)}
    for i in (0, 3):
        hooks[i].extend([viz.set_yticklabels_hook(["$0$", "$0.5$", "$1$"])])
    for i in (3, 4, 5):
        hooks[i].extend([viz.set_xticklabels_hook(["$0$", "$0.5$", "$1$"])])
    for i in (0, 1, 2):
        hooks[i].append(viz.xaxis_off_hook)
    for i in (1, 2, 4, 5):
        hooks[i].append(viz.yaxis_off_hook)
    for i, hook_lst in hooks.items():
        panels[i].opts(hooks=hook_lst)

    
    return panels


# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
def mkkey(taskkey): return taskkey[0], taskkey[4]
taskdims = ["models", "input"]
hm = hv.HoloMap({("EMD", "Bemd", float(scat.label[2:]), *mkkey(taskkey)): scat
                     for taskkey, calplot in calibs_normal.items()
                     for scat in calplot.overlayed_scatters.Scatter.values()}
                | {("Q", "Bemd", float(scat.label[2:]), *mkkey(taskkey)): scat
                     for taskkey, calplot in calibs_BQ.items()
                     for scat in calplot.overlayed_scatters.Scatter.values()},
                kdims=["uncertainty", "B", "c", *taskdims])
hm.select(uncertainty="EMD")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# (code_calibrating-with-BQ_plot)=

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# Invert the order of curves so that the smaller $c$ are drawn with lighter colours and on top

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
for calplot in calibs_normal.values():
    data = calplot.calibration_curves.data
    calplot.calibration_curves.data = {k: data[k] for k in reversed(data.keys())}

for calplot in calibs_BQ.values():
    data = calplot.calibration_curves.data
    calplot.calibration_curves.data = {k: data[k] for k in reversed(data.keys())}

# %% editable=true slideshow={"slide_type": ""}
fig = hv.Layout(
    format_calib_curves([calplot.overlayed_scatters.redim(Bemd=hv.Dimension("Bemd", label=r"$B^{\mathrm{EMD}}$"))
                         for calplot in calibs_normal.values()],
                        list(tasks_normal.values()))
).cols(3).opts(*calibopts, hv.opts.Scatter(backend="matplotlib", s=12),
               #hv.opts.Overlay(legend_position="left", legend_cols=1)
               hv.opts.Overlay(show_legend=False),
               hv.opts.Layout(fig_inches=1.15)
              )
display(fig)
# The legend will use Curve data, which are all grey dotted lines.
# To get a legend with coloured dots, we do another figure with only the scatter plots
# We don’t want the figure (just the legend), so we make the figure tiny, remove its axes, and let the legend overflow
#hv.Overlay(list(fig.Overlay.II.Scatter.data.values())).opts(
#    fig_inches=0.01,
#    hooks=[viz.xaxis_off_hook, viz.yaxis_off_hook])

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
hv.save(fig, config.paths.figures/"Bemd_prinz_calib-scatter_6panel_raw.svg")

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
fig = hv.Layout(
    format_calib_curves([calplot.overlayed_scatters.redim(Bemd=hv.Dimension("BQ", label="$B^Q$"))
                         for calplot in calibs_BQ.values()],
                        list(tasks_BQ.values()))
).cols(3).opts(*calibopts, hv.opts.Scatter(backend="matplotlib", s=12),
               #hv.opts.Layout(backend="matplotlib", hspace=-0.2, vspace=.1),
               hv.opts.Overlay(show_legend=False),
               hv.opts.Layout(fig_inches=1.15)
              )
display(fig)

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
hv.save(fig, config.paths.figures/"BQ_prinz_calib-scatter_6panel_raw.svg")

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
legend = hv.Overlay(
    [next(iter(plot.Scatter.values())).clone()
     for (c,), plot in next(iter(calibs_normal.values())).scatters.overlay("c", sort=False).data.items()])
c_values = hv.Table([(c, np.log2(c)) for c in c_list], kdims=["c"], vdims=["log2(c)"])
(legend + c_values).opts(
    hv.opts.Overlay(fontscale=2), # fig_inches=0.01
    hv.opts.Scatter(hooks=[viz.xaxis_off_hook, viz.yaxis_off_hook], s=40, xlim=(0,.01), ylim=(0,.01)),
    hv.opts.Layout(sublabel_format="")
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
hv.save(legend, config.paths.figures/"BQ-to-Bemd-comparison_legend_raw.svg")

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
# Print panel descriptions
from tabulate import tabulate
headers = ["models", "input corr", "input strength", "obs noise", "obs dist"]
data = [(f"Panel ({lbl})",
         f"{(Ω:=task.experiments).a} vs {Ω.b}", f"{Ω.τ_dist}", f"{Ω.σi_dist}", f"{Ω.σo_dist}", f"{Ω.ξ_name} noise")
        for lbl, task in zip("abcdef", tasks_BQ.values())]
print(tabulate(data, headers, tablefmt="simple_outline"))

# %% [markdown]
# (code_higher-res-prinz-calib)=
# ## Redraw the calibration curves in the main text
#
# Since in the course of this experiment, we redid the original six panel calibration with more experiments, we might as well update the figure in the main text with a higher-resolution one.
# The original code is found [here](#code_prinz-calib-main-text).

# %%
fig_calib_highres = hv.Layout(
    format_calib_curves([calplot.overlayed_lines.redim(Bemd=hv.Dimension("Bemd", label=r"$B^{\mathrm{EMD}}$"))
                         for calplot in calibs_normal.values()],
                        list(tasks_normal.values()))
).cols(3).opts(*calibopts,
               #hv.opts.Overlay(legend_position="left", legend_cols=1),
               hv.opts.Overlay(show_legend=False),
               hv.opts.Layout(fig_inches=1.15),
               hv.opts.Curve(color=hv.Palette("copper", range=(0., 1), reverse=True))
              )
display(fig_calib_highres)

# %% [markdown]
# The legend is created separately and assembled with Inkscape.

# %%
import math
legend_calib_highres = hv.Overlay([curve.clone().redim.range(Bemd=(0,0.1), Bepis=(0.5,0.55)).relabel(label=f"$c=2^{{{int(round(math.log2(c)))}}}$")
            for c, curve in zip(c_list, fig_calib_highres.Overlay.I.Curve)]).opts(
    hv.opts.Overlay(show_legend=True, legend_cols=6,
                    hooks=[viz.xaxis_off_hook, viz.yaxis_off_hook],
                    aspect=6)
)
legend_calib_highres

# %%
hv.save(fig_calib_highres, config.paths.figures/"prinz_calibrations_high-res_raw.svg")
hv.save(legend_calib_highres, config.paths.figures/"prinz_calibrations_high-res_legend_raw.svg")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# (code_comparison-tables_prinz)=
# ## Compute comparison tables for each $c$ value

# %%
from addict import Dict
from functools import partial
import math
import emdcmp

import utils
from Ex_Prinz2004 import LP_data, phys_models, AdditiveNoise, fit_gaussian_σ, generate_synth_samples, Q

# %%
import holoviews as hv
hv.extension("matplotlib")

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("fork")  # Currently only works with "fork", because of weird injection of emd-paper/viz into the sys.path of subprocesses otherwise

# %% [markdown]
# Recreate the synthetic and mixed PPFs as we do in [the base notebook](./Ex_Prinz2004.ipynb).

# %%
candidate_models = Dict()
Qrisk = Dict()
with mp.Pool(4) as pool:
    fitted_σs = pool.starmap(fit_gaussian_σ, [(LP_data, phys_models[a], "Gaussian") for a in "ABCD"])
for a, σ in zip("ABCD", fitted_σs):
    candidate_models[a] = utils.compose(AdditiveNoise("Gaussian", σ),
                                        phys_models[a])
    Qrisk[a] = Q(phys_model=phys_models[a], obs_model="Gaussian", σ=σ)


# %%
def get_synth_ppf(a):
    return emdcmp.make_empirical_risk_ppf(Qrisk[a](generate_synth_samples(candidate_models[a])))
def get_mixed_ppf(a):
    return emdcmp.make_empirical_risk_ppf(Qrisk[a](LP_data.get_data()))

with mp.Pool(4) as pool:
    synth_ppf = Dict(zip("ABCD", pool.map(get_synth_ppf, "ABCD")))
    mixed_ppf = Dict(zip("ABCD", pool.map(get_mixed_ppf, "ABCD")))

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Draw sets of R samples for each value of $c$.

# %%
c_list = [2**-6, 2**-4, 2**-2, 2**0, 2**2, 2**4]


# %% editable=true slideshow={"slide_type": ""}
def draw_R_samples(a, c):
    return emdcmp.draw_R_samples(mixed_ppf[a], synth_ppf[a], c=c)
with mp.Pool(4) as pool:
    R_samples = {}
    for c in c_list:
        R_samples[c] = Dict(zip("ABCD", pool.map(partial(draw_R_samples, c=c), "ABCD")))

# %% editable=true slideshow={"slide_type": ""}
hm = hv.HoloMap({int(round(math.log2(c))): hv.Table(emdcmp.utils.compare_matrix(R_samples[c]).reset_index().rename(columns={"index": "models"}))
                 for c in c_list},
                kdims=["log2(c)"])
hm

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# ## Exported variables

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
viz.glue("N_high-res", N)

# %% editable=true slideshow={"slide_type": ""}
emdcmp.utils.GitSHA(packages=["emdcmp", "pyloric-network-simulator"])
