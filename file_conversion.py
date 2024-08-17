"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-01
Last Modified : 2024-08-01

Summary       : Helps store data in different file formats

List of functions:
- convert_1_txt_to_json
- all_data_to_json
"""

import logging
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from time import time
from copy import deepcopy

import numpy as np
import ray

from common_utils import log_utils
from params import common_params
from classes.main_runparams_cls import MainRunParams

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


@ray.remote
def convert_1_txt_to_json(abs_txt_path: Path) -> None:
    """
    For 1 given txt file containing vorticity data, converts it into json files separated by year

    Parameters
    ----------
    abs_txt_path: Path
        Absolute path to txt file with vorticity data. Assumes its name has not been modified after downloading it from
        the internet

    Returns
    -------

    """

    logger = logging.getLogger(__name__ + ".convert_1_txt_to_json.remote")
    log_utils.set_logger(logger)

    # Dictionary containing all the data
    data = {}
    record_time_list = []

    # For each dictionary which represents a measurement, its keys are:
    vorticity_keys = [
        "r1_b1",
        "r1_b2",
        "r2_b1",
        "r2_b2",
        "area_km2",
        "vorticity_mHz",
        "MLT",
    ]
    geo_coord_keys = [
        "geo_lat_c",
        "geo_long_c",
        "geo_lat_1",
        "geo_long_1",
        "geo_lat_2",
        "geo_long_2",
        "geo_lat_3",
        "geo_long_3",
        "geo_lat_4",
        "geo_long_4",
    ]
    aacgm_coord_keys = [
        "aacgm_lat_c",
        "aacgm_long_c",
        "aacgm_lat_1",
        "aacgm_long_1",
        "aacgm_lat_2",
        "aacgm_long_2",
        "aacgm_lat_3",
        "aacgm_long_3",
        "aacgm_lat_4",
        "aacgm_long_4",
    ]

    # Opening the data txt file
    with open(abs_txt_path, "r") as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):

            if line.startswith("#"):  # Ignoring comments
                pass
            else:

                try:

                    split_line = np.array(line.split(), dtype=float)

                    if split_line.size == 4:  # If it is a line indicating time
                        hours = float(split_line[-1])
                        date_data = split_line[:-1].astype(int)

                        record_time = datetime(
                            *date_data, tzinfo=timezone.utc
                        ) + timedelta(seconds=hours * 60 * 60)
                        record_time_list.append(record_time)

                        # Splitting data according to their years
                        if record_time.year not in data.keys():
                            data[record_time.year] = {}

                        # Add a list of dictionaries. Each dict contains data of 1 measurement made at that time
                        data[record_time.year][record_time.isoformat()] = []

                    elif (
                        split_line.size == 7
                    ):  # If it is a line containing data of the vorticity

                        split_line[
                            -2
                        ] *= (
                            -1e3
                        )  # Converting vorticity to mHz with correct sign convention

                        data[record_time.year][record_time_list[-1].isoformat()].append(
                            # Adds a dict to the latest timestamp
                            dict(zip(vorticity_keys, split_line))
                        )

                    elif (
                        split_line.size == 11
                    ):  # If it is a line containing data of the coords

                        if split_line[0] == 0:

                            # Add coords to the latest dict which contains a measurement
                            data[record_time.year][record_time_list[-1].isoformat()][
                                -1
                            ].update(dict(zip(geo_coord_keys, split_line[1:])))

                        elif split_line[0] == 1:
                            data[record_time.year][record_time_list[-1].isoformat()][
                                -1
                            ].update(dict(zip(aacgm_coord_keys, split_line[1:])))

                        else:
                            logger.error(
                                f"""
                                    This line contains an unexpected coord index:
                                    {line}
                                    """
                            )

                    elif (
                        split_line.size == 1
                    ):  # If it is a line indicating how many records there are for 1 timestamp
                        pass
                    else:
                        logger.error(
                            f"""
                                This line contains an unexpected number of columns ({split_line.size} columns):
                                {line}
                                """
                        )

                except Exception as e:
                    logger.error(
                        f"""It seems like a line with non-numeric data has been found, 
                        when converting the data txt file into json files:
                        {line}
                        {e}
                        """
                    )
                    pass

    ##################################
    # Saving data to separate json files
    for year in list(data.keys()):
        out_json_path = (
            common_params.json_out_dir
            / f'{abs_txt_path.name.split("_")[0]}_{year}_vorticity.json'
        )

        with open(out_json_path, "w") as f:
            json.dump(data[year], f, indent=4)
        logger.info(f"{out_json_path.name} saved to {str(out_json_path.parent)}")

    return None


def all_data_to_json(main_params: MainRunParams) -> None:
    """
    Converts all txt files in a directory into json files, separated by year

    Parameters
    ----------
    main_params: MainRunParams
        Used here to find all the txt files with the vorticity data.
    Returns
    -------

    """
    # Initialisation

    # Directory where the json files are going to be stored at
    json_out_dir = deepcopy(common_params.json_out_dir)
    if not json_out_dir.exists():
        json_out_dir.mkdir(parents=True)

    # Start of time
    t_start = time()

    ####################################
    ray.init()
    refs = [
        convert_1_txt_to_json.remote(abs_txt_path)
        for abs_txt_path in list(main_params.abs_data_txt_dir.glob("*.txt"))
    ]
    ray.get(refs)
    ray.shutdown()

    logger.info(
        f"Converting data file into json files took {(time() - t_start) / 60:.2f} minute(s)"
    )

    return None
