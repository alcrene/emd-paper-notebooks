import os
import logging
from typing import ClassVar, Optional
from pathlib import Path
from pydantic import validator
from holoviews import Palette
from valconfig import ValConfig, ensure_dir_exists
from valconfig.contrib.holoviews import FiguresConfig
from scityping import Config as ScitypingConfig
from emdcmp import Config as EmdConfig

## Workaround for function deserialization ##
# Needed because:
# - There is currently no fine-grained control of trust for serialized functions.
#   Only "trust_all_inputs"
# - The Calibrate task is packaged with emd_falsify. So during deserialization,
#   none of our own project code is initially executed.
#   This means there is no place for us to configure trust.
# So what we do is set that here, and use `smttask run --import config`.
import scityping
scityping.config.trust_all_inputs = True
## End workaround ##

import multiprocessing as mp
#try:
#    mp.set_start_method('spawn')  # A warning is now emitted for code combining "fork" with multiprocessing
#except RuntimeError:
#    pass    # Start method has already been set; move on


# Configure JAX to use double precision
try:
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    jax_config.update("jax_platform_name", "cpu")  # Prevents warning that GPU was not found - https://github.com/google/jax/issues/6805
    del jax_config
except ImportError:
    pass

# Put the results we actually used into a shareable/publishable folder
import smttask
smttask.config.track_folder = "data/tracked_tasks"

class Config(ValConfig):
    __default_config_path__: ClassVar = "defaults.cfg"
    __local_config_filename__ = "local.cfg"
    __create_template_config__ = False  # At least while we are developing, avoid having to keep local.cfg and defaults.cfg in sync

    class logging:
        level: str

        @validator("level")
        def setloglevel(cls, level):
            logging.getLogger("emd_falsify").setLevel(level)
            logging.getLogger("Ex_Prinz2004").setLevel(level)
            logging.getLogger("Ex_UV").setLevel(level)
            return level


    class paths:
        project   : Path
        # configdir  : Path="config"
        smtproject: Path
        data      : Path
        labnotes  : Path
        figures   : Path
        glue_shelf: Path

        _ensure_dir_exists = validator("figures", allow_reuse=True,
                                      )(ensure_dir_exists)


    class random:
        entropy: int

    class mp:
        max_cores: int
        maxtasksperchild: int|None

    emd: EmdConfig=EmdConfig()

    scityping: ScitypingConfig=ScitypingConfig()

    class figures(FiguresConfig):
        pass

config = Config()
