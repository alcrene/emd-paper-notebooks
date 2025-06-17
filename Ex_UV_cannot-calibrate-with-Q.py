# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md:myst
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown]
# # Rejection probability is not predicted by loss distribution – Planck vs Rayleigh-Jeans
#
# {{ prolog }}
#
# %{{ startpreamble }}
# %{{ endpreamble }}

# %% [markdown]
# > **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).
#
# > **NOTE** This notebook is synced with a Python file using [Jupytext](https://jupytext.readthedocs.io/). **That file is required** to run this notebook, and it must be in the current working directory.

# %% editable=true slideshow={"slide_type": ""}
from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from typing import Literal

from   addict import Dict
from   scityping.pint import PintQuantity
import smttask

# %% editable=true slideshow={"slide_type": ""}
import numpy as np
from   scipy import optimize

import emdcmp

# %% editable=true slideshow={"slide_type": ""}
import holoviews as hv

# %% editable=true slideshow={"slide_type": ""}
import utils
import viz

from Ex_UV import (
    Bunits, Dataset, Q, gaussian_noise, CandidateModel,
    L_med, data_T, data_λ_min, data_λ_max, data_noise_s,
    c_chosen
)
import task_bq

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# hv.extension("matplotlib")

# %% editable=true slideshow={"slide_type": ""}
FitResult = namedtuple("FitResult", ["σ"])
def fit_gaussian_σ(data, T) -> Dict[str, FitResult]:
    """
    The candidate models depend on the temperature T and use a Gaussian
    observation model with std dev σ to compute their risk.
    T is treated as a known constant.
    σ is chosen by maximizing the likelihood.
    """
    
    fitted_σ = Dict()

    def f(log2_σ, candidate_model, _λ_B):
        σ = 2**log2_σ
        risk = Q(candidate_model, σ, T=T)(_λ_B).mean()
        priorσ = 2**(-log2_σ/128)  # Soft floor on σ so it cannot go too low
        return risk + priorσ
    
    res = optimize.minimize(f, np.log2(1e-4), ("Rayleigh-Jeans", data), tol=1e-5)
    σ = 2**res.x
    fitted_σ.RJ = FitResult(σ*Bunits,)
    
    res = optimize.minimize(f, np.log2(1e-4), ("Planck", data))
    σ = 2**res.x
    fitted_σ.Planck = FitResult(σ*Bunits,)

    return fitted_σ


# %% editable=true slideshow={"slide_type": ""}
@dataclass(frozen=True)
class FittedCandidateModels:
    """
    Candidate models need to be tuned to a dataset by fitting σ.
    This encapsulates that functionality.
    """
    dataset: Dataset

    @property
    @cache
    def fitted(self):  # Fitting is deffered until we actually need it.
        return fit_gaussian_σ(self.dataset.get_data(), T=data_T)

    @property
    def Planck(self):
        rng = utils.get_rng(*self.dataset.purpose, "candidate Planck")
        return utils.compose(
            gaussian_noise(0, self.fitted.Planck.σ, rng=rng),
            CandidateModel("Planck", T=data_T)
        )
    @property
    def RJ(self):
        rng = utils.get_rng(*self.dataset.purpose, "candidate RJ")
        return utils.compose(
            gaussian_noise(0, self.fitted.RJ.σ, rng=rng),
            CandidateModel("Rayleigh-Jeans", T=data_T)
        )

    @property
    def QPlanck(self):
        return Q("Planck", σ=self.fitted.Planck.σ, T=self.dataset.T)
    @property
    def QRJ(self):
        return Q("Rayleigh-Jeans", σ=self.fitted.RJ.σ, T=self.dataset.T)


# %% editable=true slideshow={"slide_type": ""} tags=["active-py"] raw_mimetype=""
@dataclass(frozen=True)  # frozen allows dataclass to be hashed
class EpistemicDistBiasSweep(emdcmp.tasks.EpistemicDist):
    N: int|Literal[np.inf] = np.inf

    #s_p_range   : tuple[int,int] = (13, 20)  # log₂ data_s ≈ 16.6
    B0_range    : PintQuantity   = (-1e-4, 1e-4) * Bunits
    
    __version__: int       = 1  # If the distribution is changed, update this number
                                # to make sure previous tasks are invalidated
    #def get_s(self, rng):
    #    p = rng.uniform(*self.s_p_range)
    #    return 2**p * Bunits**-1    # NB: Larger values => less noise
    def get_B0(self, rng):
        return rng.uniform(*self.B0_range.to(Bunits).m) * Bunits

    ## Experiment generator ##

    def __iter__(self):
        rng = utils.get_rng("uv", "calibration", "bias-only")
        n = 0
        while n < self.N:
            n += 1
            dataset = Dataset(
                ("uv", "calibration", "fit candidates", n),
                L    = L_med,          # L only used to fit model candidates. `CalibrateTask` will
                λmin = data_λ_min,
                λmax = data_λ_max,
                s    = data_noise_s,
                T    = data_T,
                B0   = self.get_B0(rng),
                phys_model = rng.choice(["Rayleigh-Jeans", "Planck"])
            )
            # Fit the candidate models to the data
            candidates = FittedCandidateModels(dataset)
            # Yield the data model, candidate models along with their loss functions
            yield emdcmp.tasks.Experiment(
                data_model=dataset,
                candidateA=candidates.Planck, candidateB=candidates.RJ,
                QA=candidates.QPlanck, QB=candidates.QRJ)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ```python
# @dataclass(frozen=True)  # frozen allows dataclass to be hashed
# class EpistemicDistBiasSweep(emdcmp.tasks.EpistemicDist):
#     N: int|Literal[np.inf] = np.inf
#
#     #s_p_range   : tuple[int,int] = (13, 20)  # log₂ data_s ≈ 16.6
#     B0_range    : PintQuantity   = (-1e-4, 1e-4) * Bunits
#     
#     __version__: int       = 1  # If the distribution is changed, update this number
#                                 # to make sure previous tasks are invalidated
#     #def get_s(self, rng):
#     #    p = rng.uniform(*self.s_p_range)
#     #    return 2**p * Bunits**-1    # NB: Larger values => less noise
#     def get_B0(self, rng):
#         return rng.uniform(*self.B0_range.to(Bunits).m) * Bunits
#
#     ## Experiment generator ##
#
#     def __iter__(self):
#         rng = utils.get_rng("uv", "calibration", "bias-only")
#         n = 0
#         while n < self.N:
#             n += 1
#             dataset = Dataset(
#                 ("uv", "calibration", "fit candidates", n),
#                 L    = L_med,          # L only used to fit model candidates. `CalibrateTask` will
#                 λmin = data_λ_min,
#                 λmax = data_λ_max,
#                 s    = data_noise_s,
#                 T    = data_T,
#                 B0   = self.get_B0(rng),
#                 phys_model = rng.choice(["Rayleigh-Jeans", "Planck"])
#             )
#             # Fit the candidate models to the data
#             candidates = FittedCandidateModels(dataset)
#             # Yield the data model, candidate models along with their loss functions
#             yield emdcmp.tasks.Experiment(
#                 data_model=dataset,
#                 candidateA=candidates.Planck, candidateB=candidates.RJ,
#                 QA=candidates.QPlanck, QB=candidates.QRJ)
# ```

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell", "active-ipynb"]
# import importlib
# mod = importlib.import_module("Ex_UV_cannot-calibrate-with-Q")
# EpistemicDistBiasSweep = mod.EpistemicDistBiasSweep

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# c_list = [2**-3, 2**-2, 2**-1, 2**0, 2**1]

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# #N = 64
# #N = 256
# N = 4096

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# Ω = EpistemicDistBiasSweep()

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# task = emdcmp.tasks.Calibrate(
#     reason = "UV calibration – RJ vs Plank – bias sweep",
#     c_list = c_list,
#     #c_list = [c_chosen],
#     experiments = Ω.generate(N),
#     Ldata = 1024,
#     Linf = 12288
# )

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# task_BQ = task_bq.CalibrateBQ(
#     reason = "UV calibration – RJ vs Plank – bias sweep",
#     c_list = [-2**1, -2**0, -2**-1, -2**-2, -2**-3, -2**-4, 0, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1],
#     experiments = Ω.generate(N),
#     Ldata = 1024,
#     Linf = 12288
# )

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# calib_results = task.unpack_results(task.run(cache=True))
# calib_normal = emdcmp.viz.calibration_plot(calib_results)
#
# calib_results = task_BQ.unpack_results(task_BQ.run(cache=True))
# task_bq.calib_point_dtype.names = ("Bemd", "Bconf")  # Hack to relabel fields in the way calibration_plot expects them
# calib_BQ = emdcmp.viz.calibration_plot(calib_results)
# task_bq.calib_point_dtype.names = ("BQ", "Bepis")

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# calib_BQ

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# fig = calib_normal + calib_BQ
# fig

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# calib_normal
