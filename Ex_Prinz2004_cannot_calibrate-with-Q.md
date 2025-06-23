---
jupytext:
  formats: ipynb,py:percent,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python (emd-paper)
  language: python
  name: emd-paper
---

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

---
math:
    '\Bemd' : 'B^{\mathrm{EMD}}_{#1}'
    '\Bconf': 'B^{\mathrm{epis}}_{#1}'
    '\BQ' : 'B^{Q}_{#1}'
    '\nN'   : '\mathcal{N}'
    '\Unif' : '\operatorname{Unif}'
    '\Mtrue': '\mathcal{M}_{\mathrm{true}}'
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Rejection probability is not predicted by loss distribution

{{ prolog }}

%{{ startpreamble }}
%{{ endpreamble }}

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

> **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).

> **NOTE** This notebook is synced with a Python file using [Jupytext](https://jupytext.readthedocs.io/). **That file is required** to run this notebook, and it must be in the current working directory.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This notebook explores a similar question as [](#code_aleatoric-cannot-substitute-epistemic), namely why we cannot use the loss distribution directly and instead really need to estimate the epistemic distribution.

Here we explore the question in the context of our calibration experiments. Ultimately the goal is to predict the probability that a replication of the experiment would select model $A$ over model $B$, and this is exactly what our calibration experiments test (with synthetic replications). So if we can replace $\Bemd{}$ by a hypothetical $\BQ{}$ defined solely in terms of the loss distribution and still finding a correlation with $\Bconf{}$, that would suggest that we don’t really need $\Bemd{}$ (which depends on a more complicated distribution and ad hoc parameter $c$) – we could get away with considering just the loss distribution, which is much simpler and therefore more appealing.

Of course, there is no reason to expect that the loss distribution (which is a result of aleatoric uncertainty) will be related to *replication* uncertainty – indeed, that is the point of the [aforementioned notebook](#code_aleatoric-cannot-substitute-epistemic). But the question gets asked, so in addition to arguing on the basis of principle, it is good to tackle the question head-on, in the most direct way possible, with an actual experiment.

One challenge is how best to define $B^Q$: since it measures aleatoric instead of replication/epistemic uncertainty, there is no ontologically obvious way to do this. So instead we will use the _mathematically_ obvious way and “hope for the best”:[^ml-research] we match the form of [](#eq_Bemd_def), replacing the $R$-distribution by the $Q$-distribution:

$$\BQ{AB;c_Q} := P(Q_A < Q_B + c_Q)\,.$$ (eq_BQ_def)

Note that now the probability now is over the samples of $\Mtrue$ rather than over replications.
The parameter $c_Q$ is an additional degree of freedom; we include it so that both $\BQ{}$ and $\Bemd{}$ have the same number of free parameters, and thus make the comparison more fair. In practice we will select the $c_Q$ which maximizes the correlation with $\Bconf{}$, thus giving $\BQ{}$ the best chance to match the performance of $\Bemd{}$.

The advantage of [](#eq_BQ_def) is that it is easily estimated from observed samples, with no need to define an ad hoc distribution over quantile functions.
The disadvantage is that it does not capture the effect of misspecification on the ability of a model to replicate. As we will see below, this makes it a very poor predictor for $\Bconf{}$.

Note also that appropriate values of $c_Q$ are entirely dependent on what the typical values of the loss are.
If the loss is given in terms of a probability density, rather than a probability mass, then this is not scale independent.
There is even less reason to expect the values of $c_Q$ therefore to generalize to other problems, in contrast to what we observed with $\Bemd{}$.

Of course, given the highly nonlinear shape of loss distributions, we actually expect that the appropriate "correction factor" would not be a constant shift but a function of $Q_A$ and $Q_B$, boosting the difference in some areas and reducing it in others. Such a function however would have infinitely more degrees of freedom than the simple $c$ scaling we propose with $\Bemd{}$. In fact, if we say that our hypothesis is just that there exists _some_ functional $B(Q_A,Q_B) \to \RR$ which correlates with $\Bconf{}$, then the $\Bemd{}$ is subsumed in that hypothesis, with the infinite-dimensional functional space reduced to a single scalar parameter $c$.
For this reason we think that $\BQ{}$ as defined in [](#eq_BQ_def) is the fairest expression of a comparison criterion which uses $Q_A$ and $Q_B$ directly, with no further assumptions.


[^ml-research]: Incidentally, this “match the form and hope for the best” is not uncommon in the machine learning literature, and partly explains the inconsistent results with attempts at getting models to generalize.

+++

## Load the epistemic dists used in the calibration experiments

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-input]
---
from Ex_Prinz2004 import EpistemicDist, L_data, Linf, colors, dims
from config import config
```

```{code-cell} ipython3
import viz
#import viz.emdcmp
```

```{code-cell} ipython3
from functools import partial
from itertools import chain
```

Load the customized calibration task which computes $\BQ{}$ instead of $\Bemd{}$.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-input]
---
from task_bq import CalibrateBQ, calib_point_dtype
```

```{code-cell} ipython3
import numpy as np
```

```{code-cell} ipython3
import emdcmp
```

```{code-cell} ipython3
import holoviews as hv
hv.extension("matplotlib", "bokeh")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Execution

+++

:::{admonition} How many experiments
:class: hint margin

The function `emd.viz.calibration_plot` attempts to collect results into 16 bins, so making $N$ a multiple of 16 works nicely. (With the constraint that no bin can have less than 16 points.)

For an initial pilot run, we found $N=64$ or $N=128$ to be good numbers. These numbers produce respectively 4 or 8 bins, which is often enough to check that $\Bemd{}$ and $\Bconf{}$ are reasonably distributed and that the epistemic distribution is actually probing the transition from strong to equivocal evidence.
A subsequent run with $N \in \{256, 512, 1024\}$ can then refine and smooth the curve.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Full list of epistemic dists. This is more than is useful for testing the $\BQ{}$ variant of risk distributions.

```python
N = 64
#N = 512
Ωdct = {(f"{Ω.a} vs {Ω.b}", Ω.ξ_name, Ω.σo_dist, Ω.τ_dist, Ω.σi_dist): Ω
        for Ω in (EpistemicDist(N, a, b, ξ_name, σo_dist, τ_dist, σi_dist)
                  for (a, b) in [("A", "B"), ("A", "D"), ("C", "D")]
                  for ξ_name in ["Gaussian", "Cauchy"]
                  for σo_dist in ["low noise", "high noise"]
                  for τ_dist in ["short input correlations", "long input correlations"]
                  for σi_dist in ["weak input", "strong input"]
            )
       }
```

Instead we only run the experiments on the 6 epistemic distributions used in the main paper.

```{code-cell} ipython3
N = 64
N = 128
N = 512
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
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{hint}
:class: margin
`Calibrate` will iterate over the data models twice, so it is important that the iterable passed as `data_models` not be consumable.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
c_chosen = 2**-2
c_list = [2**-6, 2**-4, 2**-2, 2**0, 2**2, 2**4]
LQ = 4000 # Number of draws for the Monte Carlo estimate of normal dist addition
```

    cQ_chosen = -2**-3
    cQ_list=[#-2**1,  2**1,
             #-2**0,  2**0,
             -2**-1, 2**-1,
             #-2**-2, 2**-2,
             -2**-3, 2**-3,
             #0
            ]

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
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
        experiments = Ω.generate(N),
        Ldata = L_data,
        Linf = Linf
    )
    tasks_normal[Ωkey] = task
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The code below creates task files for any missing tasks

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, skip-execution]
---
    for task in chain(tasks_normal.values(), tasks_BQ.values()):
        if not task.has_run:  # Don’t create task files for tasks which have already run
            Ω = task.experiments
            taskfilename = f"prinz_{type(task).__qualname__}__{Ω.a}vs{Ω.b}_{Ω.ξ_name}_{Ω.σo_dist}_{Ω.τ_dist}_{Ω.σi_dist}_N={Ω.N}_c={task.c_list}"
            task.save(taskfilename)
```

If any files were created, run those tasks from the command line with

    smttask run -n1 --import config <task file>

before continuing.

```{code-cell} ipython3
assert all(task.has_run for task in chain(tasks_normal.values(), tasks_BQ.values()))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "jp-MarkdownHeadingCollapsed": true}

### Analysis

+++

:::{important} The basic principles for calibration

Wide support
~ The goal of calibration is to probe that intermediate where selection of either model $A$ or $B$ is not certain.
  It is important therefore that we obtain values $\Bemd{}$ over the entire interval $[0, 1]$.  
  Whether this is the case will be a function of two things:
  - the design of the design of the calibration experiments: whether it produces ambiguous selection problems;
  - the choice of $c$: generally, a larger $c$ will concentrate $\Bemd{}$ towards 0.5, a smaller $c$ will concentrate them towards 0 and 1.

  So we want the support of $\Bemd{}$ to be as large as possible.

Flat distribution
~ As a secondary goal, we also want it to be as flat as possible, since this will lead to more efficient sampling: Since we need enough samples at every subinterval of $\Bemd{}$, it is the most sparsely sampled regions which determine how many calibration datasets we need to generate. (And therefore how long the computation needs to run.)

  Beyond making for shorter compute times, a flat distribution however isn’t in and of itself a good thing: more important is that the criterion is able to resolve the models when it should.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} Hint: Diagnosing $\Bemd{}$ and $\Bconf{}$ histograms
:class: hint dropdown

$\Bemd{}$ distribution which bulges around 0.5.
~ *May* indicate that $c$ is too large and the criterion underconfident.
~ *May also* indicate that the calibration distribution is generating a large number of (`data`, `modelA`, `modelB`) triples which are essentially undecidable. If neither model is a good fit to the data, then their $δ^{\mathrm{EMD}}$ discrepancies between mixed and synthetic PPFs will be large, and they will have broad distributions for the expected risk. Broad distributions overlap more, hence the skew of $\Bemd{}$ towards 0.5.

$\Bemd{}$ distribution which is heavy at both ends.
~ *May* indicate that $c$ is too small and the criterion overconfident.
~ *May also* indicate that the calibration distribution is not sampling enough ambiguous conditions. In this case the answer is *not* to increase the value of $c$ but rather to tighten the calibration distribution to focus on the area with $\Bemd{}$ close to 0.5. It may be possible to simply run the calibration longer until there have enough samples everywhere, but this is generally less effective than adjusting the calibration distribution.

$\Bemd{}$ distribution which is heavily skewed either towards 0 or 1.
~ Check that the calibration distribution is using both candidate models to generate datasets. The best is usually to use each candidate to generate half of the datasets: then each model should fit best in roughly half the cases.
The skew need not be removed entirely – one model may just be more permissive than the other.
~ This can also happen when $c$ is too small.

$\Bconf{}$ distribution which is almost entirely on either 0 or 1.
~ Again, check that the calibration distribution is using both models to generate datasets.
~ If each candidate is used for half the datasets, and we *still* see ueven distribution of $\Bconf{}$, then this can indicate a problem: it means that the ideal measure we are striving towards (true expected risk) is unable to identify that model used to generate the data. In this case, tweaking the $c$ value is a waste of time: the issue then is with the problem statement rather than the $\Bemd{}$ calibration. Most likely the issue is that the loss is ill-suited to the problem:
  + It might not account for rotation/translation symmetries in images, or time dilation in time-series.
  + One model’s loss might be lower, even on data generated with the other model. This can happen with a log posterior, when one model has more parameters: its higher dimensional prior "dilutes" the likelihood. This may be grounds to reject the more complex model on the basis of preferring simplicity, but it is *not* grounds to *falsify* that model. (Since it may still fit the data equally well.)

:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} Hint: Diagnosing calibration curves
:class: hint dropdown

Flat calibration curve
~ This is the most critical issue, since it indicates that $\Bemd{}$ is actually not predictive of $\Bconf{}$ at all. The most common reason is a mistake in the definition of the calibration distribution, where some values are fixed when they shouldn’t be.
  - Remember that the model construction pipeline used on the real data needs to be repeated in full for each experimental condition produced by the calibration distribution. For example, in `Experiment` above we refit the observation noise $σ$ for each experimental condition generated within `__iter__`.
  - Treat any global used within `EpistemicDist` with particular suspicion, as they are likely to fix values which should be variable.
    To minimize the risk of accidental global variables, you can define `EpistemicDist` in its own separate module.
~ To help investigate issues, it is often helpful to reconstruct conditions that produce the unexpected behaviour. The following code snippet recovers the first calibration dataset for which both `Bemd > 0.9` and `Bconf = False`; the recovered dataset is `D`:
  ```python
  Bemd = calib_results[1.0]["Bemd"]
  Bconf = calib_results[1.0]["Bconf"]
  i = next(iter(i for i in range(len(Bemd)) if Bemd[i] > 0.9))
    
  for j, D in zip(range(i+1), task.models_Qs):
      pass
  assert j == i
  ```

Calibration curve with shortened domain
~ I.e. $\Bemd{}$ values don’t reach 0 and/or 1. This is not necessarily critical: the calibration distribution we want to test may simply not allow to fully distinguish the candidate models under any condition. 
~ If it is acceptable to change the calibration distribution (or to add one to the test suite), then the most common way to address this is to ensure the distribution produces conditions where $\Bemd{}$ can achieve maximum confidence – for example by having conditions with negligeable observation noise.
:::

```{code-cell} ipython3
calibs_normal = {key: viz.emdcmp.calibration_plot(task.unpack_results(task.run()))
                 for key, task in tasks_normal.items()}
calibs_BQ     = {key: viz.emdcmp.calibration_plot(task.unpack_results(task.run()))
                 for key, task in tasks_BQ.items()}
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
hm.select(uncertainty="Q").overlay("B").layout(taskdims).cols(3).opts(
    hv.opts.Layout(backend="matplotlib", fig_inches=3, tight=True),
    hv.opts.NdLayout(backend="matplotlib", fig_inches=3, tight=True),
    hv.opts.Histogram(backend="matplotlib", aspect=2)
)
```

```{code-cell} ipython3
fig = hm.select(uncertainty="EMD", B="Bemd").drop_dimension(["uncertainty", "B"]).overlay("c") \
      + hm.select(uncertainty="Q", B="Bemd").drop_dimension(["uncertainty", "B"]).overlay("c")\
        .redim(Bemd="BQ").opts(title="BQ")
fig.opts(hv.opts.Layout(backend="matplotlib", fig_inches=2, sublabel_format=""),
         hv.opts.Histogram(backend="matplotlib", aspect=1.5),
         hv.opts.NdOverlay(legend_position="best")
        )
```

```{code-cell} ipython3
calibopts = (
    hv.opts.Overlay(backend="matplotlib",
                    #legend_cols=3,
                    #legend_opts={"columnspacing": .5, "alignment": "center",
                    #             "loc": "upper center"},
                    #hooks=[partial(viz.despine_hook(), left=False)],
                    #fig_inches=config.figures.defaults.fig_inches,
                    aspect=1, fontscale=1.3),
    hv.opts.Scatter(backend="matplotlib", s=20),
    hv.opts.Layout(sublabel_format=""),
    hv.opts.Layout(backend="matplotlib", hspace=0.1, vspace=0.05,
                   fig_inches=0.65*config.figures.defaults.fig_inches)
)
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
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
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
fig = hv.Layout(
    format_calib_curves([calplot.overlayed_scatters for calplot in calibs_normal.values()],
                        list(tasks_normal.values()))
).cols(3).opts(*calibopts, hv.opts.Scatter(backend="matplotlib", s=20),
               #hv.opts.Overlay(legend_position="left", legend_cols=1)
               hv.opts.Overlay(show_legend=False),
              )
display(fig)
# The legend will use Curve data, which are all grey dotted lines.
# To get a legend with coloured dots, we do another figure with only the scatter plots
# We don’t want the figure (just the legend), so we make the figure tiny, remove its axes, and let the legend overflow
#hv.Overlay(list(fig.Overlay.II.Scatter.data.values())).opts(
#    fig_inches=0.01,
#    hooks=[viz.xaxis_off_hook, viz.yaxis_off_hook])
```

```{code-cell} ipython3
hv.Layout(
    format_calib_curves([calplot.overlayed_scatters for calplot in calibs_BQ.values()],
                        list(tasks_BQ.values()))
).cols(3).opts(*calibopts, hv.opts.Scatter(backend="matplotlib", s=20),
               hv.opts.Layout(backend="matplotlib", hspace=-0.1, vspace=.1))
```

```{code-cell} ipython3
hv.Layout(
    format_calib_curves([calplot.scatters[c_chosen] for calplot in calibs_normal.values()],
                        list(tasks_normal.values()))
).cols(3).opts(*calibopts, hv.opts.Scatter(backend="matplotlib", s=20),
               hv.opts.Layout(backend="matplotlib", hspace=0.1, vspace=.02))
```

```{code-cell} ipython3
hv.Layout(
    format_calib_curves([calplot.scatters[cQ_chosen] for calplot in calibs_BQ.values()],
                        list(tasks_BQ.values()))
).cols(3).opts(*calibopts, hv.opts.Scatter(backend="matplotlib", s=20),
               hv.opts.Layout(backend="matplotlib", hspace=0.1, vspace=.02))
```

```{code-cell} ipython3
# Print panel descriptions
from tabulate import tabulate
headers = ["models", "input corr", "input strength", "obs noise", "obs dist"]
data = [(f"Panel ({lbl})",
         f"{(Ω:=task.experiments).a} vs {Ω.b}", f"{Ω.τ_dist}", f"{Ω.σi_dist}", f"{Ω.σo_dist}", f"{Ω.ξ_name} noise")
        for lbl, task in zip("abcdef", tasks_BQ.values())]
print(tabulate(data, headers, tablefmt="simple_outline"))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_prinz-model-comparison)=
## EMD model comparison

Based on the calibration results, we choose the value $c=${glue:text}`c_chosen_prinz` (set above) to compute the $\Bemd{}$ criterion between models.

First we recreate `synth_ppf` and `mixed_ppf` as we did above.

```{code-cell} ipython3
from addict import Dict
```

```{code-cell} ipython3
from Ex_Prinz2004 import Q, fit_gaussian_σ, LP_data, phys_models, AdditiveNoise, generate_synth_samples
import utils
```

```{code-cell} ipython3
candidate_models = Dict()
Qrisk = Dict()
for a in "ABCD":
    fitted_σ = fit_gaussian_σ(LP_data, phys_models[a], "Gaussian")
    candidate_models[a] = utils.compose(AdditiveNoise("Gaussian", fitted_σ),
                                        phys_models[a])
    Qrisk[a] = Q(phys_model=phys_models[a], obs_model="Gaussian", σ=fitted_σ)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
synth_ppf = Dict({
    a: emdcmp.make_empirical_risk_ppf(Qrisk[a](generate_synth_samples(candidate_models[a])))
    for a in "ABCD"
})
mixed_ppf = Dict({
    a: emdcmp.make_empirical_risk_ppf(Qrisk[a](LP_data.get_data()))
    for a in "ABCD"
})
```

Sample of a set of expected risks ($R$) for each candidate model.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
R_samples = {
    c: Dict(
        A = emdcmp.draw_R_samples(mixed_ppf.A, synth_ppf.A, c=c),
        B = emdcmp.draw_R_samples(mixed_ppf.B, synth_ppf.B, c=c),
        C = emdcmp.draw_R_samples(mixed_ppf.C, synth_ppf.C, c=c),
        D = emdcmp.draw_R_samples(mixed_ppf.D, synth_ppf.D, c=c)
    )
    for c in [2**-6, 2**-4, 2**-2, 2**0, 2**2, 2**4]
}
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Convert the samples into a distributions using a kernel density estimate (KDE).

```{code-cell} ipython3
hv.Layout([
    hv.Overlay([hv.Distribution(samples) for samples in R_samples[c].values()])
    for c in R_samples
]).cols(1).opts(
    hv.opts.Overlay(aspect=3),
    hv.opts.Layout(fig_inches=3.5, shared_axes=False)
)
```

```{code-cell} ipython3
yticks = [[0, 15, 30],
          [0, 10, 20],
          [0, 4, 8],
          [0, 2, 4],
          [0, 1, 2],
          [0, 0.5, 1]]
ylims = [(0, 30), (0, 20), (0, 8), (0, 4.5), (0, 2), (0, 1)]
xticks = [[4.2, 4.3, 4.4, 4.5],
          [4.0, 4.2, 4.4, 4.6],
          [3.8, 4.0, 4.2, 4.6, 4.8, 5.0],
          [3.5, 4.0, 4.5, 5.0, 5.5],
          [3, 4, 5, 6, 7],
          [2, 4, 6, 8, 10]]
          
fig = hv.HoloMap(
    {c: viz.make_Rdist_fig(
        R_samples[c],
        colors     =colors.LP_candidates,
        xticks     =_xticks,
        #xticklabels=["2", "", "6", "" ,"10"],
        yticks     =_yticks,
        #yticklabels=["0", "", "", "6"]
        #xlabelshift=0,
        #ylabelshift=0
        ).opts(hv.opts.Overlay(ylim=_ylim))
     for c, _xticks, _yticks, _ylim in zip(R_samples, xticks, yticks, ylims)
     },
    kdims=["c"]
)
fig.opts(framewise=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
xticks = [3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7]
fig_Rdists = viz.make_Rdist_fig(
    R_samples,
    colors     =colors.LP_candidates,
    xticks     =xticks,
    xticklabels=["", "", "4.0", "" ,"", "", "", "", "4.6", ""],
    yticks     =[0, 2, 4, 6],
    yticklabels=["0", "", "", "6"],
)
fig_Rdists
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
    Rdists = [hv.Distribution(_Rlst, kdims=[dims.R], label=f"Model {a}")
              for a, _Rlst in R_samples.items()]
    Rcurves = [hv.operation.stats.univariate_kde(dist).to.curve()
               for dist in Rdists]
    fig_Rdists = hv.Overlay(Rdists) * hv.Overlay(Rcurves)
    
    xticks = [round(R,1) for R in np.arange(3, 5, 0.1) if (low:=fig_Rdists.range("R")[0]) < R < (high:=fig_Rdists.range("R")[1])]
    xticklabels = [str(R) if R in (4.0, 4.6) else "" for R in xticks]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb, remove-cell]
---
    # Plot styling
    yticks = [0, 2, 4, 6]
    yticklabels = ["0", "", "", "6"]
    fig_Rdists.opts(
        hv.opts.Distribution(alpha=.3),
        hv.opts.Distribution(facecolor=colors.LP_candidates, color="none", edgecolor="none", backend="matplotlib"),
        hv.opts.Curve(color=colors.LP_candidates),
        hv.opts.Curve(linestyle="solid", backend="matplotlib"),
        hv.opts.Overlay(backend="matplotlib", fontscale=1.3,
                        hooks=[viz.set_xticks_hook(xticks), viz.set_xticklabels_hook(xticklabels), viz.ylabel_shift_hook(5),
                               viz.set_yticks_hook(yticks), viz.set_yticklabels_hook(yticklabels), viz.xlabel_shift_hook(7),
                               viz.despine_hook(2)],
                        legend_position="top_left", legend_cols=1,
                        show_legend=False,
                        xlim=fig_Rdists.range("R"),  # Redundant, but ensures range is not changed
                        #fig_inches=config.figures.defaults.fig_inches)  # Previously: was 1/3 full width
                        aspect=3
                       )
    )
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
# First line: means of the R f
# Second line: expected risk computed on the original data
#Rmeans = hv.Spikes([(Rlst.mean(), 2, a) for a, Rlst in R_samples.items()],
Rmeans = hv.Spikes([(Qrisk[a](LP_data.get_data()).mean(), 2, a) for a in R_samples.keys()],
                     kdims=dims.R, vdims=["height", "model"], group="back")
# Display options
dashstyles = {"A": (0, (4, 4)), "B": (4, (4, 4)),
              "C": (0, (3.5, 4.5)), "D": (4, (3.5, 4.5))}
model_colors = {a: c for a, c in zip("ABCD", colors.LP_candidates.values)}

# Because the C and D models are so close, the lines are very difficult to differentiate
# To make this easier, we overlay with interleaved dashed lines.
# NB: Key to making this visually appealing is that we leave a gap between
#     the dash segments
Rmeans_front = hv.Overlay([
    hv.Spikes([(R, h, a)], kdims=Rmeans.kdims, vdims=Rmeans.vdims,
              group="front", label=f"Model {a}")
    .opts(backend="matplotlib", linestyle=dashstyles[a])
    for R, h, a in Rmeans.data.values])
# NB: Current versions don’t seem to include Spikes in the legend.
#     Moreover, the shifted dash style means that for B and D the line is not printed in the legend
legend_proxies = hv.Overlay([hv.Curve([(R, 0, a)], group="proxy", label=f"Model {a}")
                             for R, h, a in Rmeans.data.values])
fig_Rmeans = Rmeans * Rmeans_front * legend_proxies

fig_Rmeans.opts(
    hv.opts.Spikes(color=hv.dim("model", lambda alst: np.array([model_colors[a] for a in alst]))),
    hv.opts.Spikes("back", alpha=0.5),
    hv.opts.Spikes(backend="matplotlib", linewidth=2, hooks=[viz.yaxis_off_hook]),
    hv.opts.Overlay(backend="matplotlib",
                    show_legend=True, legend_position="bottom_left",
                    xticks=xticks,
                    hooks=[viz.set_xticklabels_hook(""), viz.despine_hook(0), viz.yaxis_off_hook],
                    xlim=fig_Rdists.range("R"),
                    aspect=6)
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
# Why aren’t the fontsizes consistent across panels ? No idea...
fig = (fig_Rmeans.opts(fontscale=1.3, sublabel_position=(-.25, .4), show_legend=False, xlabel="")
       + fig_Rdists.opts(sublabel_position=(-.25, .9), show_legend=True))
fig.opts(shared_axes=True, tight=False, aspect_weight=True,
         sublabel_format="", sublabel_size=12,
         vspace=0.1,
         fig_inches=config.figures.defaults.fig_inches)
fig.cols(1)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell, active-ipynb]
---
viz.save(fig, config.paths.figures/f"prinz_Rdists_raw.svg")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Things to finish in Inkscape:
- ~~Fix alignment of sublabels~~
- ~~Make sublabels bold~~
- Trim unfinished dashed lines in the R means
- Extend lines for R means into lower panel
- Confirm that alignment of x axes pixel perfect
- ~~Tighten placement of xlabel~~
- Save to PDF

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

Bigger version; appropriate for HTML or slides.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
vlines = hv.Overlay([hv.VLine(R).opts(color=model_colors[a])
                     for R, _, a in fig_Rmeans.data['Back', 'I'].data.values])
fig2 = (fig_Rmeans.opts(clone=True, fontscale=2, show_legend=False, sublabel_position=(-.215, .4))
                  .opts(hv.opts.Spikes(linewidth=4))
       + (vlines*fig_Rdists).opts(clone=True, fontscale=2, show_legend=True, sublabel_position=(-.25, .9))
                            .opts(hv.opts.VLine(linewidth=4, alpha=0.5)))
fig2[1].opts(fig_Rdists.opts.get())
fig2[1].opts(show_legend=False)
fig2[0].opts(show_legend=True)
fig2[0].opts(backend="matplotlib", legend_position='top_left',
             legend_opts={'framealpha': 1, 'borderpad': 1,
                          'labelspacing': .5,
                          'bbox_to_anchor': (-.02, 1)})  # NB: 'loc' is ignored; use legend_position
fig2.opts(shared_axes=True, tight=False, aspect_weight=True,
          sublabel_format="", sublabel_size=18,
          vspace=0,  # For some reason, negative vspace doesn’t work
          fig_inches=5.5)
viz.save(fig2, config.paths.figures/f"prinz_Rdists_big.svg")
fig2.cols(1)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

EMD estimates for the probabilities $P(R_a < R_b)$ are nicely summarized in a table:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
df = emd.utils.compare_matrix(R_samples)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
df.index = pd.MultiIndex.from_tuples([("a", a) for a in df.index])
df.columns = pd.MultiIndex.from_tuples([("b", b) for b in df.columns])
df.style.format(precision=3)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

For pasting into a MyST file, we reformat as a [list-table](https://jupyterbook.org/en/stable/reference/cheatsheet.html?highlight=list-table#tables):

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution, remove-cell, active-ipynb]
---
print(":::{list-table}")
model_labels = list(R_samples)
print(":header-rows: 1")
print(":stub-columns: 1")
print("")

print(f"* - ")
for a in model_labels:
    print(f"  - {a}")
for a in model_labels:
    print(f"* - {a}")
    for Pab in df.loc[("a", a),:]:
        print(f"  - {Pab:.3f}")
print(":::")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} `compare_matrix` implementation
:class: note dropdown

The `compare_matrix` function provided by `emdcmp` simply loops through all $(a,b)$ model pairs, and counts the number of $R_a$ samples which are larger than $R_b$:

```python
def compare_matrix(R_samples: Dict[str, ArrayLike]) -> pd.DataFrame:
    R_keys = list(R_samples)
    compare_data = {k: {} for k in R_keys}
    for i, a in enumerate(R_keys):
        for j, b in enumerate(R_keys):
            if i == j:
                assert a == b
                compare_data[b][a] = 0.5
            elif j < i:
                compare_data[b][a] = 1 - compare_data[a][b]
            else:
                compare_data[b][a] = np.less.outer(R_samples[a], R_samples[b]).mean()
    return pd.DataFrame(compare_data)
```
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"], "jp-MarkdownHeadingCollapsed": true}

## Exported notebook variables

These can be inserted into other pages.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
glue("AB_model", AB_model_label, display=True)
glue("c_chosen_prinz", c_chosen, raw_myst=viz.format_pow2(c_chosen, format='latex'), raw_latex=viz.format_pow2(c_chosen, format='latex'))

glue("N", N, display=True)
glue("Linf", viz.format_pow2(Linf), display=True)
glue("Ldata", L_data, display=True)

#glue("noise_model", Ω.ξ_dist, display=True)
#glue("noise_bias", Ω.μ_dist, display=True)
#glue("noise_width", Ω.σ_dist, display=True)

glue("calib_curve_palette", config.emd.viz.matplotlib.calibration_curves["color"].key, display=True)

# Epistemic hyperparameters
glue("tau_short_min"   , **viz.formatted_quantity(EpistemicDist.τ_short_min ))
glue("tau_short_max"   , **viz.formatted_quantity(EpistemicDist.τ_short_max ))
glue("tau_long_min"    , **viz.formatted_quantity(EpistemicDist.τ_long_min  ))
glue("tau_long_max"    , **viz.formatted_quantity(EpistemicDist.τ_long_max  ))
glue("sigmao_low_mean" , **viz.formatted_quantity(EpistemicDist.σo_low_mean ))
glue("sigmao_high_mean", **viz.formatted_quantity(EpistemicDist.σo_high_mean))
glue("sigmao_std"      , **viz.formatted_quantity(EpistemicDist.σo_std      ))
glue("sigmai_weak_mean", **viz.formatted_quantity(EpistemicDist.σi_weak_mean ))
glue("sigmai_strong_mean", **viz.formatted_quantity(EpistemicDist.σi_strong_mean))
glue("sigmai_std"      , **viz.formatted_quantity(EpistemicDist.σi_std      ))
```

    'AB/PD 3'
    0.25
    512
    '2¹⁴'
    4000
    'copper'
    '0.1 \\mathrm{ms}'
    '0.2 \\mathrm{ms}'
    '1.0 \\mathrm{ms}'
    '2.0 \\mathrm{ms}'
    '0.0 \\mathrm{mV}'
    '1.0 \\mathrm{mV}'
    '0.5 \\mathrm{mV}'
    '-15.0 \\mathrm{mV}'
    '-10.0 \\mathrm{mV}'
    '0.5 \\mathrm{mV}'

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
emd.utils.GitSHA()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```
