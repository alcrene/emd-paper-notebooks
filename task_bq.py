# ---
# math:
#     '\Bemd' : 'B^{\mathrm{EMD}}_{#1}'
#     '\Bconf': 'B^{\mathrm{epis}}_{#1}'
#     '\BQ' : 'B^{Q}_{#1}'
#     '\nN'   : '\mathcal{N}'
#     '\Unif' : '\operatorname{Unif}'
#     '\Mtrue': '\mathcal{M}_{\mathrm{true}}'
# ---

# # Task definition for testing using the $Q$ distribution for model comparison

# This implements a modified version of the `Calibrate` task packaged with `emdcmp`,
# where insted of using of EMD distribution, the loss distribution (i.e. the distribution of $Q$) is directly used to estimate the probability via the simple ratio
#
# $$
# \BQ{AB;c_Q} &:= P(Q_A < Q_B + ε)\,, \\
# ε &\sim \nN(0, c) \,.
# $$ 
#
# While simple, there is no reason to expect this rule to work, since $Q$ describes aleatoric uncertainty while we are trying to estimate replication uncertainty.
# And indeed this is what we find; see [](./Ex_UV_cannot-calibrate-with-Q.ipynb) and [](./Ex_Prinz2004_cannot-calibrate-with-Q.ipynb).

import logging
from   typing import List
from   functools import partial

import multiprocessing as mp
import time
import psutil
from   tqdm.auto import tqdm

import numpy as np

from scityping import Dataclass
from smttask import RecordedTask, TaskOutput
from emdcmp.tasks import compute_Bconf as compute_Bepis

from config import config
from utils import get_rng

logger = logging.getLogger(__name__)

calib_point_dtype = np.dtype([("BQ", float), ("Bepis", bool)])
CalibrateResult = dict[float, np.ndarray[calib_point_dtype]]

class CalibrateOutput(TaskOutput):
    """Compact format used to store task results to disk.
    Use `task.unpack_result` to convert to a `CalibrateResult` object.
    """
    BQ : List[float]
    Bepis: List[float]

def compute_BQ(ω, c, Ldata, LQ):
    """
    If c is array-like, return an array of values, one for each c.
    """

    logger.debug(f"Compute BQ - Generating {Ldata} data points."); t1 = time.perf_counter()
    data = ω.data_model(Ldata)                                   ; t2 = time.perf_counter()
    logger.debug(f"Compute BQ - Done generating {Ldata} data points. Took {t2-t1:.2f} s")

    # Draw a bunch of random ε to perform the convolution with Monte Carlo
    try:
        descA = ω.QA.candidate_model
        descB = ω.QB.candidate_model
    except AttributeError:
        descA = ω.QA.phys_model + "+" + ω.candidateA.obs_model
        descB = ω.QB.phys_model + "+" + ω.candidateB.obs_model
    rng = get_rng(ω.data_model.purpose, descA, descB, c, Ldata, LQ)
    c_shape = np.shape(c)
    c = np.reshape(c, (*c_shape, 1, 1))
    ε = rng.normal(0, c, size=(*c_shape, LQ, Ldata))

    # For each value of c, average over the ε draws and the data points.
    return np.mean(ω.QA(ω.candidateA(data)) < ω.QB(ω.candidateB(data)) + ε,
                   axis=(-2,-1))

def compute_BQ_and_Bepis(i_ω, c, Ldata, Linf, LQ):
    i, ω = i_ω
    BQ = compute_BQ(ω, c, Ldata, LQ)
    Bepis = compute_Bepis(ω.data_model, ω.QA, ω.QB, Linf)
    return i, BQ, Bepis


@RecordedTask
class CalibrateBQ:

    def __call__(
        self,
        c_list     : List[float],
        experiments: Dataclass,   # Iterable of Experiment elements
        Ldata      : int,
        Linf       : int,
        LQ         : int,
        ) -> CalibrateOutput:

        # Validation
        if any(c < 0 for c in c_list):
            raise ValueError("All `c` values must be positive")

        # Bind arguments to the compute* function, so it takes one argument
        c_hashable = tuple(c_list)
        compute_partial = partial(compute_BQ_and_Bepis,
                                  c=c_hashable, Ldata=Ldata, Linf=Linf, LQ=LQ)


        # Define dictionaries into which we accumulate results
        BQ_results = {}
        Bepis_results = {}

        # - Set the iterator over parameter combinations (we need two identical ones)
        # - Set up progress bar.
        # - Determine the number of multiprocessing cores we will use.
        try:
            N = len(experiments)
        except (TypeError, AttributeError):  # Typically TypeError, but AttributeError seems also plausible
            logger.info("Data model iterable has no length: it will not be possible to estimate the remaining computation time.")
            total = None
        else:
            total = N
        progbar = tqdm(desc="Calib. experiments", total=total)
        ncores = psutil.cpu_count(logical=False)
        ncores = min(ncores, total, config.mp.max_cores)

        # Use multiprocessing to run experiments in parallel.
        # i is used as an id for each different model/Qs set
        ω_gen = ((i, ω) for i, ω in enumerate(experiments))

        if ncores > 1:
            with mp.Pool(ncores, maxtasksperchild=config.mp.maxtasksperchild) as pool:
                # Chunk size calculated following mp.Pool's algorithm (See https://stackoverflow.com/questions/53751050/multiprocessing-understanding-logic-behind-chunksize/54813527#54813527)
                # (Naive approach would be total/ncores. This is most efficient if all taskels take the same time. Smaller chunks == more flexible job allocation, but more overhead)
                chunksize, extra = divmod(N, ncores*6)
                if extra:
                    chunksize += 1
                BQ_Bepis_it = pool.imap_unordered(compute_partial, ω_gen,
                                                  chunksize=chunksize)
                for (i, BQarr, Bepis) in BQ_Bepis_it:
                    progbar.update(1)        # Updating first more reliable w/ ssh
                    for c, BQ in zip(c_list, BQarr):
                        BQ_results[i, c] = BQ
                    Bepis_results[i] = Bepis
        # Variant without multiprocessing
        else:
            BQ_Bepis_it = (compute_partial(arg) for arg in ω_gen)
            for (i, BQarr, Bepis) in BQ_Bepis_it:
                progbar.update(1)
                for c, BQ in zip(c_list, BQarr):
                    BQ_results[i, c] = BQ
                Bepis_results[i] = Bepis

        # Cleanup
        progbar.close()

        # Return results in the compressed format
        return dict(BQ =[BQ_results [i,c] for i in range(len(experiments)) for c in c_list],
                    Bepis=[Bepis_results[i] for i in range(len(experiments))])

    def unpack_results(self, result: "CalibrateBQ.Outputs.result_type"
                      ) -> CalibrateResult:
        """
        Take the compressed result exported by the task, and recreate the
        dictionary structure of a `CalibrateResult`, where experiments are
        organized by their value of `c`.
        """
        assert len(result.BQ) == len(self.c_list) * len(result.Bepis), \
            "`result` argument seems not to have been created with this task."

        # Reconstruct the dictionary as it was at the end of task execution
        BQ_dict    = {}; BQ_it    = iter(result.BQ)
        Bepis_dict = {}; Bepis_it = iter(result.Bepis)
        for i in range(len(result.Bepis)):         # We don’t actually need the models
            Bepis_dict[i] = next(Bepis_it)         # => we just use integer ids
            for c in self.taskinputs.c_list:       # This avoids unnecessarily
                BQ_dict[i, c] = next(BQ_it)        # instantiating models.

        # Package results into record arrays – easier to sort and plot
        calib_curve_data = {c: [] for c in self.taskinputs.c_list}
        for i, c in BQ_dict:
            calib_curve_data[c].append(
                (BQ_dict[i, c], Bepis_dict[i]) )

        return {c: np.array(calib_curve_data[c], dtype=calib_point_dtype)
                for c in self.taskinputs.c_list}
