import logging
import json
from pathlib import Path
from typing import List

import numpy as np
import ray
from numpy.typing import NDArray

from common_utils import log_utils
from classes import main_runparams_cls
from classes.data_class import vort_measurement

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


@ray.remote
def gen_vort_obj_by_year(
        file_abs_path: Path
) -> NDArray[vort_measurement]:

    with open(file_abs_path, 'r') as f:
        data_dict = json.load(f)

    vort_yearly = np.array(
        [vort_measurement(vort_dict, iso_time_str)
        for iso_time_str, vort_list in data_dict.items()
        for vort_dict in vort_list]
    )

    return vort_yearly


def json_to_list() -> NDArray[vort_measurement]:

    json_files = list(main_runparams_cls.json_out_dir.glob('*vorticity.json'))  # list of all json files

    ray.init()
    all_vort = ray.get(
        [gen_vort_obj_by_year.remote(vort_file) for vort_file in json_files]
    )
    ray.shutdown()

    return np.concatenate(all_vort)
