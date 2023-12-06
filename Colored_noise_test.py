# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Exploration colored external input
#
# The idea here is to find a good noise level for the external input $I_{\mathrm{ext}}$ which will cause some variation in spike timing, but not so much that neuron models can’t be differentiated.

# %% editable=true slideshow={"slide_type": ""}
import time
import numpy as np

# %% editable=true slideshow={"slide_type": ""}
from Ex_Prinz2004 import Model
from colored_noise.colored_noise import ColoredNoise
from viz import make_int_superscript

# %% editable=true slideshow={"slide_type": ""}
import holoviews as hv
hv.extension("bokeh")


# %% editable=true slideshow={"slide_type": ""}
model = Model(LP_model="LP 3")
#tarr = np.linspace(3000, 7000)
tarr = 3000. + np.arange(4000)*1.

# %% editable=true slideshow={"slide_type": ""}
integrate_times = {}

# %% editable=true slideshow={"slide_type": ""}
traces = []

# %% editable=true slideshow={"slide_type": ""}
t1 = time.perf_counter()
model(tarr)
t2 = time.perf_counter()
print(f"{t2-t1:.0f} s")

integrate_times[0] = t2-t1

# %% editable=true slideshow={"slide_type": ""}
traces.append(hv.Curve(zip(tarr, model(tarr)),
                       kdims=["t"], vdims=["V"], label=f"σ=0"))

# %% editable=true slideshow={"slide_type": ""}
for p in [-12, -8, -6, -4, -2]:
    σ = 2**p
    I_ext = ColoredNoise(t_min=2000., t_max=7500.,
                         scale=σ, 
                         corr_time=100.,
                         impulse_density=30,
                         rng  =np.random.Generator(333))
    t1 = time.perf_counter()
    model(tarr, I_ext=I_ext)
    t2 = time.perf_counter()
    print(f"{t2-t1:.0f} s")
    integrate_times[σ] = t2-t1
    traces.append(hv.Curve(zip(tarr, model(tarr, I_ext=I_ext)),
                           kdims=["t"], vdims=["V"],
                           label=f"σ=2{make_int_superscript(p)}"))

# %% editable=true slideshow={"slide_type": ""}
fig = hv.Overlay(traces).opts(
    hv.opts.Overlay(legend_position="right", width=600, backend="bokeh"),
    hv.opts.Curve(width=500, backend="bokeh")
)
hv.save(fig, "Colored noise exploration.html")
fig

# %% editable=true slideshow={"slide_type": ""}
hv.HoloMap({σ: curve for σ, curve in zip(integrate_times, traces)},
           kdims="σ")

# %% editable=true slideshow={"slide_type": ""}
times = hv.Curve([(σ,T) for σ,T in integrate_times.items()
                         if σ>0],
                 kdims=["σ"], vdims=[hv.Dimension("T", unit="s")])
times.opts(logx=True, title="Simulation time")              

# %% editable=true slideshow={"slide_type": ""}
