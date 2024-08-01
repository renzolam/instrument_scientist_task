"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-01
Last Modified : 2024-08-01

Summary       : Reads in json files and store them in a numpy array of custom objects

List of functions:
- gen_vort_obj_by_year
- json_to_array
"""


import logging
import json
from pathlib import Path
from typing import List

import numpy as np
import ray
from numpy.typing import NDArray

from common_utils import log_utils
from classes import main_runparams_cls
from classes.data_class import VortMeasurement

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


@ray.remote
def gen_vort_obj_by_year(
        file_abs_path: Path
) -> NDArray[VortMeasurement]:
    """
    Turns data in a json file into a numpy array of VortMeasurement objects

    Parameters
    ----------
    file_abs_path: Path
        Absolute path to json file containing data for a certain year

    Returns
    -------
    numpy array of VortMeasurement objects
    """

    with open(file_abs_path, 'r') as f:
        data_dict = json.load(f)

    vort_yearly = np.array(
        [VortMeasurement(vort_dict, iso_time_str)
         for iso_time_str, vort_list in data_dict.items()
         for vort_dict in vort_list]
    )

    return vort_yearly


def json_to_array() -> NDArray[VortMeasurement]:
    """
       Turns data in several json files into a single numpy array of VortMeasurement objects

       Parameters
       ----------

       Returns
       -------
       numpy array of VortMeasurement objects
       """

    json_files = list(main_runparams_cls.json_out_dir.glob('*vorticity.json'))  # list of all json files

    ray.init()
    all_vort = ray.get(
        [gen_vort_obj_by_year.remote(vort_file) for vort_file in json_files]
    )
    ray.shutdown()

    return np.concatenate(all_vort)
