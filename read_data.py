"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-01
Last Modified : 2024-08-15

Summary       : Reads in json files and store them in a numpy array of custom objects

List of functions:
- gen_vort_obj_by_year
- json_to_array
"""

import logging
import json
import sys
from time import time
from pathlib import Path
from typing import Union

import numpy as np
import ray
from numpy.typing import NDArray

from common_utils import log_utils
from params import common_params
from classes.data_class import VortMeasurement

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


@ray.remote
def gen_vort_obj_by_year(file_abs_path: Path) -> Union[NDArray[VortMeasurement], None]:
    """
    Turns data in a json file into a numpy array of VortMeasurement objects

    Parameters
    ----------
    file_abs_path: Path
        Absolute path to json file containing data for a certain year

    Returns
    -------
    Union[NDArray[VortMeasurement], None]
        numpy array of VortMeasurement objects, or None if the reading process failed
    """

    logger = logging.getLogger(__name__)
    log_utils.set_logger(logger)

    try:
        with open(file_abs_path, "r") as f:
            data_dict = json.load(f)

        vort_yearly = np.array(
            [
                VortMeasurement(vort_dict, iso_time_str)
                for iso_time_str, vort_list in data_dict.items()
                for vort_dict in vort_list
            ]
        )

        return vort_yearly

    except Exception as e:
        logger.exception(f"{e}\nExiting programme")
        sys.exit()


def json_to_array() -> NDArray:
    """
    Turns data in several json files into a single numpy array of VortMeasurement objects, in parallel

    Parameters
    ----------

    Returns
    -------
    NDArray
        numpy array of VortMeasurement objects, or None if the reading process failed
    """

    json_files = sorted(
        list(common_params.json_out_dir.glob("*vorticity.json"))
    )  # list of all json files

    # Reading in data in parallel
    t_read_start = time()
    ray.init()
    all_vort = ray.get(
        [gen_vort_obj_by_year.remote(vort_file) for vort_file in json_files]
    )
    ray.shutdown()

    logger.info(f"Reading in all json files took {time() - t_read_start:.2f} seconds")

    return np.concatenate(all_vort)
