"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-07
Last Modified : 2024-08-07

Summary       : Class for holding data of the parameters for producing the maps of mean and median values

List of classes:
- MainRunParams
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List
from copy import deepcopy

from common_utils import log_utils


logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


class PlotParams:
    """
    Class containing all info for creating maps of mean and median values
    """

    lat_bin_size_degree: float = 1
    mlt_bin_size_hr: float = 1

    r1_avg_vort_vs_mlt_aacgm_lat_lim: List[float] = [72, 77]

    def __init__(self):

        # Setting values
        json_path = Path(__file__).parent.parent / "params" / "plot_params.json"

        with open(json_path, "r") as f:
            paramsdict = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

        # Automatically turns the data of the param json file into ths class' attributes
        for attr in dir(paramsdict):

            # Only go through attributes present in the json file
            if not attr.startswith("__") and not attr.endswith("__"):

                attr_2_set = getattr(paramsdict, attr)

                if attr == "lat_bin_size_degree" or attr == "mlt_bin_size_hr":
                    try:
                        attr_2_set = float(attr_2_set)
                    except ValueError:
                        raise ValueError("Bin sizes must be floats")
                else:
                    pass

                setattr(self, attr, attr_2_set)

        ##################
        # Ensuring class has valid values

        # Checking latitude bin size
        if not (0 < self.lat_bin_size_degree < 90):
            raise ValueError("Latitude bin size must be within 0 and 90 degrees")
        elif 90 % self.lat_bin_size_degree != 0:
            raise ValueError(
                f"Latitude bin size {self.lat_bin_size_degree} must be divisible than 90 degrees"
                f"Since we are plotting from 0 to 90 degrees"
            )
        else:
            pass

        # Checking MLT bin size
        if not (0 < self.mlt_bin_size_hr < 24):
            raise ValueError("MLT bin size must be within 0 and 24")
        elif 24 % self.mlt_bin_size_hr != 0:
            raise ValueError(
                f"MLT bin size {self.mlt_bin_size_hr} must be divisible than 24"
                f"Since we are plotting from 0h to 24h"
            )
        else:
            pass
