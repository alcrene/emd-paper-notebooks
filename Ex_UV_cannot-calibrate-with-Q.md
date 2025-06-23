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

# Rejection probability is not predicted by loss distribution – Planck vs Rayleigh-Jeans

{{ prolog }}

%{{ startpreamble }}
%{{ endpreamble }}

+++

> **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).

> **NOTE** This notebook is synced with a Python file using [Jupytext](https://jupytext.readthedocs.io/). **That file is required** to run this notebook, and it must be in the current working directory.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from typing import Literal

from   addict import Dict
from   scityping.pint import PintQuantity
import smttask
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
from   scipy import optimize

import emdcmp
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import holoviews as hv
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import utils
import viz
import viz.emdcmp

from Ex_UV import (
    Bunits, Dataset, Q, gaussian_noise, CandidateModel,
    L_med, data_T, data_λ_min, data_λ_max, data_noise_s,
    EpistemicDistBiasSweep,
    c_chosen
)
import task_bq
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
hv.extension("matplotlib")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
c_list = [2**-3, 2**-2, 2**-1, 2**0, 2**1]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
#N = 64
#N = 256
N = 4096
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
Ω = EpistemicDistBiasSweep()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
task = emdcmp.tasks.Calibrate(
    reason = "UV calibration – RJ vs Plank – bias sweep",
    c_list = c_list,
    #c_list = [c_chosen],
    experiments = Ω.generate(N),
    Ldata = 1024,
    Linf = 12288
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
task_BQ = task_bq.CalibrateBQ(
    reason = "UV calibration – RJ vs Plank – bias sweep",
    c_list = [-2**1, -2**0, -2**-1, -2**-2, -2**-3, -2**-4, 0, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1],
    experiments = Ω.generate(N),
    Ldata = 1024,
    Linf = 12288
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
task_BQ.run();
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
calib_results = task.unpack_results(task.run(cache=True))
calib_normal = viz.emdcmp.calibration_plot(calib_results, target_bin_size=32)

calib_results = task_BQ.unpack_results(task_BQ.run(cache=True))
task_bq.calib_point_dtype.names = ("Bemd", "Bconf")  # Hack to relabel fields in the way calibration_plot expects them
calib_BQ = viz.emdcmp.calibration_plot(calib_results, target_bin_size=32)
task_bq.calib_point_dtype.names = ("BQ", "Bepis")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
fig = calib_normal.overlayed_scatters + calib_BQ.overlayed_scatters.redim(c="c_Q", Bemd="BQ")
fig
```

```{code-cell} ipython3
fig.opts(
    hv.opts.Layout(fig_inches=7, fontscale=5),
    hv.opts.Curve(linewidth=1, fontscale=2),
    hv.opts.Scatter(s=20),
    hv.opts.Overlay(fontscale=2)
)
```

```{code-cell} ipython3
calib_normal.Bemd_hists.overlay() \
+ calib_BQ.Bemd_hists.select(c=[-2**1, -2**-1, -2**-3, 0, 2**-3, 2**-1, 2**1]).overlay()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
calib_BQ.Bemd_hists.select(c=[-2**1, -2**-1, -2**-3, 0, 2**-3, 2**-1, 2**1])
```

```{code-cell} ipython3

```
