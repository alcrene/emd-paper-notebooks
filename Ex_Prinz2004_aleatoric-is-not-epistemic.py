# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     formats: ipynb,md:myst,py:percent
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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# (code_aleatoric-cannot-substitute-epistemic)=
# # Aleatoric vs Sampling vs Replication uncertainty
#
# {{ prolog }}
#
# %{{ startpreamble }}
# %{{ endpreamble }}
#
# Some readers may ask why go through the trouble of compute the $\Bemd{}$, when we could just use a bootstrap to estimate the uncertainty on the expected risk. The simple answer is that
#
# 1. a bootstrap estimate of the variance measures the _aleatoric_ uncertainty, not the _epistemic_ uncertainty;
# 2. the aleatoric uncertainty on the expected risk vanishes in the limit of large data. 
#
# Indeed, the aleatoric uncertainty amounts to the standard error, which typically decreases as $L^{-1/2}$ with increasing dataset size.
# A bootstrap estimate of the aleatoric uncertainty therefore also contributes to the spread of $R$-distribution, _in addition_ to the epistemic contribution estimated with $δ^\EMD$ and $\qproc$. With the dataset size $L=4000$ we used in this example, a bootstrap estimate adds about 0.07 to the spread of the $R$-distribution (measured as the difference between the 95% and 5% percentiles). In contrast, the $R$-distributions shown in the figure have spreads around 0.5.
#
# To illustrate this, we compare $R$-distributions with
# - distributions of the loss itself over the dataset;
# - re-samples of the dataset. (What one typically estimates with a bootstrap approach.)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# > **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).
#
# > **NOTE** This notebook is synced with a Python file using [Jupytext](https://jupytext.readthedocs.io/). **That file is required** to run this notebook, and it must be in the current working directory.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# Poor man’s parallelization: By passing two arguments (the total number $n$ of processes and a process index $i < n$), we execute only every $n$-th loop iteration. To execute every iteration, start $n$ processes with process indices ranging from $0$ to $n - 1$.

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
import sys
if "ipykernel_launcher.py" in sys.argv[0]:
    # Running in a notebook
    num_processes = 1
    process_idx = 0  # Must start at 0.
elif len(sys.argv) == 1:
    # Running without args: execute every loop iteration
    num_processes = 1
    process_idx = 0
elif len(sys.argv) == 3:
    # Running as one of n processes: execute only every n-th loop iteration
    process_idx = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    if num_processes < 1:
        import warnings
        warnings.warn("The number of processes cannot be 0.")
    if process_idx > num_processes - 1:
        import warnings
        warnings.warn("The process index is greater than the (# processes - 1): parallelized loops will be SKIPPED entirely.\n"
                      "Note that processes are indexed starting from 0.\n")
else:
    raise TypeError("This script takes either no arguments (no parallelization) or exactly two arguments (the process index and the total number of processes).")

# %% editable=true slideshow={"slide_type": ""}
from Ex_Prinz2004 import *
from tqdm.auto import trange
import itertools

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
# from viz import glue

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The block below is copied from `Ex_Prinz2004` because it is only executed when that file is run as a notebook.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# candidate_models = Dict()
# Qrisk = Dict()
# for a in "ABCD":
#     fitted_σ = fit_gaussian_σ(LP_data, phys_models[a], "Gaussian")
#     candidate_models[a] = utils.compose(AdditiveNoise("Gaussian", fitted_σ),
#                                         phys_models[a])
#     Qrisk[a] = Q(phys_model=phys_models[a], obs_model="Gaussian", σ=fitted_σ)

# %%
hv.extension("matplotlib")

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
#memory = utils.NotebookMemory("./synced-joblib-data", cache_name="aleatoric-is-not-epistemic", verbose=0)
memory = utils.NotebookMemory("./data/aleatoric-is-not-epistemic", cache_name="aleatoric-is-not-epistemic", verbose=0)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# (The `NotebookMemory` class specializes `joblib.Memory` to make the cache is portable across machines.)
# We want to sync this cache across machines, so the cache location should not be excluded from sync; we could eventually define it in `config.paths`

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# precomputing = False  # Only executed in notebook

# %% editable=true raw_mimetype="" slideshow={"slide_type": ""} tags=["active-py"]
precomputing = True   # Only executed in a script

# %% [markdown]
# Generating long 40000 sample datasets means integrating the ODE for 40000 time steps, which takes a long time.
# So we use a memoized function to avoid redoing this every time.

# %% editable=true slideshow={"slide_type": ""}
LP_data_big = replace(LP_data, L=40_000)  # Reuse the same datasets for each dataset size
                                          # (The last 16 calls to fit_gaussian_dataset are cached, with the dataset as an arg)
                                          # (Each size is still done many times.)


# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# :::{warning}
# `memory.cache` will invalidate the cache if _anything_ changes in the function definitions; for example:
# - changing the memory object name from `memory` to `mem`;
# - adding or removing white space;
# - adding or removing comments;
# will invalidate the cache.
# :::

# %% editable=true slideshow={"slide_type": ""}
@memory.cache
def get_data(LP_model="LP 1", seed=0):
    dataset = replace(LP_data_big, LP_model=LP_model)
    return dataset.get_data(rng=utils.get_rng("prinz", "aleatoric-comparison", seed))


# %% editable=true slideshow={"slide_type": ""}
@memory.cache
def get_Qarr(candidate: Literal["a","b","c","d"], L: int,
             LP_model="LP 1", seed=0):
    dataset = replace(LP_data_big, LP_model=LP_model)
    fitted_σ = fit_gaussian_σ(dataset, phys_models[a], "Gaussian")
    
    data = get_data(LP_model, seed).iloc[:L]
    return Q(phys_model=phys_models[a], obs_model="Gaussian", σ=fitted_σ)(data)


# %% [markdown] editable=true slideshow={"slide_type": ""}
#     # Temporary hack to invalidate just a subset of the cache
#     def is_valid(metadata):
#         return int(metadata["input_args"]["L"]) <= 40_000
#     get_Qarr.cache_validation_callback = is_valid

# %% [markdown]
# ## Bootstrapping comparison

# %% [markdown]
# :::{margin}
# **Bootstrapped aleatoric uncertainty**
# :::

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input", "active-ipynb"]
# n_bootstrap_resamples = 1200
# Ls_bs = [400, 4000, 40_000]  # Dataset sizes for bootstrapping illustration
# rng_bs = utils.get_rng("prinz", "aleatoric", "bootstrap", "subsampling")  # Used to create bootstrap samples by subsampling the dataset
#
# assert LP_data_big.L >= max(Ls_bs)  
#
# models = "ABCD"
# R_bootstrap_samples = {L_bs: {a: [] for a in models}
#                        for L_bs in Ls_bs}
# for L_bs, a in tqdm(itertools.product(Ls_bs, models), desc="Computing bootstrap samples", total=len(Ls_bs)*len(models)):
#     Qarr = get_Qarr(a, L=L_bs)
#     for subsample in range(n_bootstrap_resamples):
#         R_bootstrap_samples[L_bs][a].append(Qarr[rng_bs.integers(L_bs, size=L_bs)].mean())

# %% [markdown] editable=true slideshow={"slide_type": ""}
#     ipdb>  [func_id, args_id]
#     ['HOME-Notebooks-emd-paper-<aleatoric-is-not-epistemic>/get_Qarr', '00959773ada2421cc85a707784aac832']
#     ipdb>  p args
#     ('A',)
#     ipdb>  p kwargs
#     {'L': 400}

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input", "active-ipynb"]
# index = pd.MultiIndex.from_product((Ls_bs, R_bootstrap_samples[L_bs].keys()))
# bootstrap_stats = pd.DataFrame(
#     [(q05:=np.quantile(Rvals, 0.05), q95:=np.quantile(Rvals, 0.95), q95-q05)
#      for L_bs in Ls_bs
#      for Rvals in R_bootstrap_samples[L_bs].values()],
#     index=index, columns=["5% quantile ($q_{05}$)", "95% quantile ($q_{95}$)", "$q_{95} - q_{05}$"])
# bootstrap_stats.unstack(level=0)#sort=False).swaplevel(axis="columns").sort_index(axis="columns")

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# figs = []
# for L_bs, samples in R_bootstrap_samples.items():
#     fig = viz.make_Rdist_fig(R_bootstrap_samples[L_bs],
#                              colors=colors.LP_candidates,
#                              xticks = [4, 4.2, 4.4, 4.6, 4.8],
#                              xticklabels = ["4", "", "", "", "4.8"],
#                              yticks = [])
#     ymax = fig.range("Density")[1]
#     _opts = fig.opts  # Adding Text replaces the Overlay, so we need to transfer opts over
#     fig = fig * hv.Text(4.8, 0.9*ymax, f"L = {L_bs:<5}", halign="left", valign="top",
#                         kdims=[dims.R, "R_density"], fontsize=10)
#     figs.append(fig.opts(_opts.get()))
# hooks = figs[-1].opts.get().options.get("hooks", [])
# for panel in figs[:-1]:
#     panel.opts(hv.opts.Overlay(hooks=hooks+[viz.yaxis_off_hook, viz.xaxis_off_hook]))
# figs[-1].opts(hooks = hooks+[viz.yaxis_off_hook])
# fig_bootstrap = hv.Layout(figs)
# fig_bootstrap.opts(
#     hv.opts.Overlay(xlim=(4, 5), fontscale=1.3, fig_inches=4,
#                     aspect=4, show_legend=False, sublabel_format=""),
#     hv.opts.Layout(vspace=0.0, shared_axes=False)
# )
# fig_bootstrap.cols(1)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Loss distribution

# %% [markdown] editable=true slideshow={"slide_type": ""}
# What about simply taking the distribution of losses computed on all the samples? Apart from again measuring only _aleatoric_, not _epistemic_, uncertainty, it would make for a very underpowered criterion:
#
# > Having the loss distributions of two models overlap does not mean that they cannot be compared.
#
# What matters when comparing models is their _expected_ risk.
# Of courses, if the distributions of losses are so different that they don’t overlap, then it is also clear which will have the lowest expected risks. But such domination of one model over the other is rare, especially if both were fitted to the data. Much more common is that the loss distributions will overlap substantially.

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "hide-input"]
# Qdists = []
# for a, c in zip("ABCD", colors.LP_candidates.values):
#     Qdist = hv.Distribution(Qrisk[a](LP_data.get_data()), kdims=["loss"], label=f"Model {a}").opts(facecolor=c, edgecolor=c)
#     Qcurve = hv.operation.stats.univariate_kde(Qdist).to.curve().opts(color=c)
#     Qdists.append(Qdist*Qcurve)
# fig_Qdist = hv.Overlay(Qdists)
# fig_Qdist.opts(
#     hv.opts.Distribution(alpha=0.5, xlim=(2.8, 9),
#                          xticks=[3, 5, 7, 9],
#                          xformatter=lambda x: f"{x:.0f}" if x == 3 or x == 9 else "",
#                          yformatter=lambda y: f"{y:.1f}" if np.isclose(y, [0, 1]).any() else ""),
#     hv.opts.Curve(linewidth=3, alpha=0.7),
#     hv.opts.Overlay(fig_inches=4, fontscale=1.3, aspect=3,
#                     show_legend=True, legend_opts={"labelspacing": 0.4}),
# )
# fig_Qdist.opts(
#     hooks=[viz.despine_hook, viz.xlabel_shift_hook(8), viz.ylabel_shift_hook(10)]
# )

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell", "active-ipynb"]
# hv.save(fig_bootstrap, config.paths.figures/"prinz_R-bootstrap-dists.svg")
# hv.save(fig_Qdist, config.paths.figures/"prinz_Qdist-overlap.svg")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Fully-synthetic resampling
#
# As a final comparison, we make a completely synthetic test, where the loss is evaluated on its own candidate model. This has two immediate drawbacks:
# - Still, we only consider aleatoric variability. _Especially so_, since now the computation in no way depends on the actual data.
# - The value of the loss on synthetic samples can be very different from the real ones, especially since we neglected epistemic errors.
#
# We do this for both the Prinz models (A, B, C, D) and black body radiation models (Planck, Rayleigh-Jeans).

# %%
n_synthetic_resamples = 400
Ls_bs = [400, 4000, 40_000, 400_000]  # Dataset sizes for bootstrapping illustration
#assert LP_data_big.L >= max(Ls_bs)

# %% [markdown]
# ### Prinz models
#
# - `a` ∈ {A,B,C,D} : The candidate model we consider.
# - `LP_model` ∈ {`LP 1`, `LP 2`, `LP 3`, `LP 4`, `LP 5`}: The model used to generate the dataset.
#     - In this experiment, this is always the model corresponding to `a`. (So effectively `LP 1` is never used.)
# - `L_synth`: Number of synthetic data samples in one dataset
# - `subsamble_idx`: Arbitrary integer; used to seed the RNG, so that we get different datasets.
# - `n_synthetic_resamples`: Number of times we recreate a new dataset.
#
# For each dataset, we:
# 1. Generate the dataset given `LP_model` and the ground truth $σ_\mathrm{obs}$ used everywhere in the paper. (Currently 2 mV.)
# 2. Fit the observation $σ_\mathrm{obs}$ by maximum likelihood.
#    - For the synthetic experiments this is somewhat redundant, since the MLE will be very close to the ground truth value used to generate the dataset.
#    - For experiments where there is model mismatch, this is essential to both to represent actual practice and to get consistent results.
# 3. Evaluate $Q(\mathcal{D}^{\mathrm{test}} | \mathcal{M}_a, σ_\mathrm{obs})$.

# %% [markdown] editable=true slideshow={"slide_type": ""}
#     LP_data_big = replace(LP_data_big, L=L_synth)
#     dataset = replace(LP_data_big, LP_model=LP_model)
#     #fitted_σ = fit_gaussian_σ(dataset, phys_models[a], "Gaussian")
#     
#     data = get_data(LP_model, 0)#.iloc[:L_synth]
#     print(LP_data_big.L)
#     print(len(data))

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{admonition} Caution – Hack
# :class: caution
# We are here enlarging `LP_data_big`, because I hard-coded it into the memoized functions, and I don’t want to invalidate the cache for the experiments above.
# This means the notebook won’t work as expected if executed out of order.
# :::

# %% editable=true slideshow={"slide_type": ""}
# Hack to enlarge the dataset size
LP_data_big = replace(LP_data_big, L=400_000)

# %%
# These two packages are used to track missing computations and print a summary message
from collections import Counter
from tabulate import tabulate

# %% [markdown] editable=true slideshow={"slide_type": ""}
#     # Temporary hack to invalidate the part of the cache related to synth datasets
#     def invalid(metadata):
#         return False
#     get_Qarr.cache_validation_callback = invalid
#     get_data.cache_validation_callback = invalid

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The loop below has two code paths:
# - If `precomputing` is `True`: Assume that we are pre-computing the datasets and `Qarr` functions.
#   - No point in generating the R samples since we won’t use them.
# - Otherwise: Assume that we want to generate the plots.
#   - To allow plotting results before a batch of computations are done, _we skip examples for which `Qarr` is not precomputed_.
#
# So as currently coded, this loop _must_ be executed in two different ways.
# - First with multiple processes, to generate all the datasets and `Q_arr` functions.
# - Then in a single process, to generate the plots.
#
# If needed, we could add flags to allow generating everything in one go (at the cost of potentially locking up the process for a long time).

# %% editable=true slideshow={"slide_type": ""}
models = "ABCD"
R_synthetic_samples = {L_synth: {a: [] for a in models}
                       for L_synth in Ls_bs}
skipped = Counter()
print_status = precomputing   # Don’t print status messages while preparing plots in the notebook
for L_synth, a in tqdm(itertools.product(Ls_bs, models), total=len(Ls_bs)*len(models),
                       desc="Pure synthetic bootstrap (Prinz model)"):
    #if print_status: print(f"Synthetic resamples with dataset size L={L_synth}")
    if precomputing:
        print(f"Caching the Qarr function for {n_synthetic_resamples // num_processes} resamples of model {a}. (Subset #{process_idx})")
    else:
        if print_status: print(f"Computing {n_synthetic_resamples} resamples of model {a}.")

    LP_model = f"LP {ord(a) - ord('A') + 2}"
    for subsample_idx in range(n_synthetic_resamples):
        # print(f"{a}, L={L_synth}, LP_model={LP_model}, seed={subsample_idx} – "
        #       f"{get_Qarr.check_call_in_cache(a, L=L_synth, LP_model=LP_model, seed=subsample_idx)}")
        # continue
        if subsample_idx % num_processes != process_idx:   # Poor man’s parallelization: execute script multiple times with different args
            continue
        if print_status: tqdm.write(f"Computing synthetic sample #{subsample_idx}")

        if precomputing:
            # We are pre-computing the Qarr function – no point to generate R samples
            get_Qarr(a, L=L_synth, LP_model=LP_model, seed=subsample_idx)
        else:
            # We are (probably) generating the plots
            if get_Qarr.check_call_in_cache(a, L=L_synth, LP_model=LP_model, seed=subsample_idx):
                Qarr = get_Qarr(a, L=L_synth, LP_model=LP_model, seed=subsample_idx)
                R_synthetic_samples[L_synth][a].append(Qarr.mean())
            else:
                # To prevent locking the process, we skip those examples that don’t have pre-computed Qarr
                skipped[(L_synth, a)] += 1                

if skipped.total() > 0:
    total_examples = len(Ls_bs) * len("ABCD") * n_synthetic_resamples
    print(f"\nA total of {skipped.total()} examples (out of {total_examples}) were skipped because they don’t have a precomputed Qarr:")
    print(tabulate([(L, model, v) for (L, model), v in skipped.items()], headers=["dataset size (L)", "model", "# skipped"]))

# %% editable=true slideshow={"slide_type": ""}
for L, sampledict in R_synthetic_samples.items():
    to_remove = [a for a, samplelist in sampledict.items()
                   if len(samplelist) == 0]
    for a in to_remove:
        del sampledict[a]

# %%
R_synthetic_samples_prinz = R_synthetic_samples

# %% [markdown]
# ### Rayleigh-Jeans models

# %%
import Ex_UV

# %% [markdown]
# :::{hint}
# :class: margin
# The RNG seed is determined from a dataset’s `purpose`.
# :::

# %%
models = ["Planck", "Rayleigh-Jeans"]
R_synthetic_samples = {L_synth: {a: [] for a in models}
                       for L_synth in Ls_bs}

for L_synth, a in tqdm(itertools.product(Ls_bs, models), total=len(Ls_bs)*len(models),
                       desc="Pure synthetic bootstrap (BB radiation)"):
    for subsample_idx in range(n_synthetic_resamples):
        dataset = replace(Ex_UV.observed_dataset,
                          L=L_synth, phys_model=a,
                          purpose=f"Pure synthetic bootstrap – {L_synth}, {a}, {subsample_idx}")
        Qarr = Ex_UV.Qrisk[a](dataset.get_data())
        R_synthetic_samples[L_synth][a].append(Qarr.mean())

# %%
R_synthetic_samples_blackbody = R_synthetic_samples


# %% [markdown]
# ### Plot results

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{margin}
# **Synthetic estimate of finite-size uncertainty**
# :::
#
# Accurately representing the distributions is delicate: we want to show that they become Dirac deltas, but the effect is so pronounced that they can’t all be plotted on the same scale.
# We _could_ plot them with a logarithmic y scale, but this is not what people expect. Moreover, while it is accurate, it falsely gives the impression that there is meaningful probability mass in tails.
#
# Our solution is to keep the same scale for all plots, but to allow the lower ones to stretch into the upper ones. (With some shading effects so the main plot on each level stands out.) This is easiest to achieve by reducing the `aspect` (so the plot outputs taller but with the same width), and then stacking the levels in Inkscape.

# %%
def plot_synthetic_stack(R_synthetic_samples, colors, xticks, xtext):
    figs = []
    for L_synth, samples in R_synthetic_samples.items():
        fig = viz.make_Rdist_fig(R_synthetic_samples[L_synth],
                                 colors = colors, xticks = xticks, yticks = []
                                )
        ymax = fig.range("Density")[1]
        _opts = fig.opts  # Adding Text replaces the Overlay, so we need to transfer opts over
        fig = fig * hv.Text(xtext, 0.9*ymax, f"L = {L_synth:<5}", halign="left", valign="top",
                            kdims=[dims.R, "R_density"], fontsize=10)
        figs.append(fig.opts(_opts.get()))
    hooks = figs[-1].opts.get().options["hooks"]
    for panel in figs[:-1]:
        panel.opts(hv.opts.Overlay(hooks=hooks+[viz.yaxis_off_hook, viz.xaxis_off_hook]))
    figs[-1].opts(hooks = hooks+[viz.yaxis_off_hook])
    fig_synthetic = hv.Layout(figs)
    fig_synthetic.opts(
        hv.opts.Overlay(#xlim=(4.15, 5),
                        fontscale=1.3, fig_inches=4,
                        aspect=5, show_legend=False, sublabel_format=""),
        hv.opts.Layout(vspace=0.0, shared_axes=False)
    )
    return fig_synthetic.cols(1)


# %%
fig_synthetic_Prinz = plot_synthetic_stack(
    R_synthetic_samples_prinz,
    colors = colors.LP_candidates,
    xticks = [3, 6],
    xtext  = 5
)
fig_synthetic_Prinz

# %%
fig_synthetic_Prinz.opts(
    hv.opts.Overlay(aspect=1, ylim=(0,18.5))
)

# %%
fig_synthetic_UV = plot_synthetic_stack(
    R_synthetic_samples_blackbody,
    colors=hv.Cycle([Ex_UV.colors.Planck, Ex_UV.colors.RJ]),
    xticks = [-8.7, -8.6, -8.5, -8.4],
    xtext  = -8.45
)
fig_synthetic_UV.opts(hv.opts.Overlay(aspect=0.3, ylim=(0, 240)))

# %%
hv.save(fig_synthetic_Prinz, config.paths.figures/"synth-bootstrap_Prinz_raw.svg")
hv.save(fig_synthetic_UV, config.paths.figures/"synth-bootstrap_UV_raw.svg")

# %% [markdown]
# :::{note} We decided not to include the black body figure.
# :::

# %% [markdown]
# ## Finalization in Inkscape
#
# - Combine all figures together.
# - Remove vertical white space between subpanels.
# - Align the $L$ labels
# - Add vertical guidelines (`#f2f2f2`)
# - Add vertical scale bars for each panel (0.4, 2 and 40)
# - Add subfigure labels
# - Replace all axis labels with text generated with _TexText_.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# ## Export variables

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
# glue("n_bootstrap_resamples", n_bootstrap_resamples)
# glue("n_synthetic_resamples", n_synthetic_resamples)

# %% editable=true slideshow={"slide_type": ""} tags=["remove-input"]
emd.utils.GitSHA()
