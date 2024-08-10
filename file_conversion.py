"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-01
Last Modified : 2024-08-01

Summary       : Helps store data in different file formats

List of functions:
- data_to_json
"""

import logging
from datetime import datetime, timedelta, timezone
import json
from time import time
from copy import deepcopy

import numpy as np

from common_utils import log_utils
from classes import main_runparams_cls

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def data_to_json(
        main_params: main_runparams_cls.MainRunParams,
):
    """

    Parameters
    ----------
    main_params: main_runparams_cls.MainRunParams
        Parameters for the run. Used here to know where the data txt file is

    Returns
    -------

    """
    # Initialisation

    # Directory where the json files are going to be stored at
    json_out_dir = deepcopy(main_runparams_cls.json_out_dir)
    if not json_out_dir.exists():
        json_out_dir.mkdir(parents=True)

    # Dictionary containing all the data
    data = {}
    record_time_list = []

    # For each dictionary which represents a measurement, its keys are:
    vorticity_keys = ['r1_b1', 'r1_b2', 'r2_b1', 'r2_b2', 'area', 'vorticity_mHz', 'MLT']
    geo_coord_keys = ['geo_lat_c', 'geo_long_c', 'geo_lat_1', 'geo_long_1', 'geo_lat_2', 'geo_long_2', 'geo_lat_3',
                      'geo_long_3', 'geo_lat_4', 'geo_long_4']
    aacgm_coord_keys = ['aacgm_lat_c', 'aacgm_long_c', 'aacgm_lat_1', 'aacgm_long_1', 'aacgm_lat_2', 'aacgm_long_2',
                        'aacgm_lat_3', 'aacgm_long_3', 'aacgm_lat_4', 'aacgm_long_4']

    # Start of time
    t_start = time()

    ####################################
    # Opening the data txt file
    with open(main_params.abs_data_file_path, 'r') as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):

            if line.startswith('#'):  # Ignoring comments
                pass
            else:

                try:

                    split_line = np.array(line.split(), dtype=float)

                    if split_line.size == 4:  # If it is a line indicating time
                        hours = float(split_line[-1])
                        date_data = split_line[:-1].astype(int)

                        record_time = (datetime(*date_data, tzinfo=timezone.utc)
                                       + timedelta(seconds=hours * 60 * 60))
                        record_time_list.append(record_time)

                        # Splitting data according to their years
                        if record_time.year not in data.keys():
                            data[record_time.year] = {}

                        # Add a list of dictionaries. Each dict contains data of 1 measurement made at that time
                        data[record_time.year][record_time.isoformat()] = []

                    elif split_line.size == 7:  # If it is a line containing data of the vorticity

                        # TODO: Perhaps add a switch here from runparams
                        split_line[-2] *= -1e3  # Converting vorticity to mHz with correct sign convention

                        data[record_time.year][record_time_list[-1].isoformat()].append(  # Adds a dict to the latest timestamp
                            dict(
                                zip(vorticity_keys, split_line)
                            )
                        )

                    elif split_line.size == 11:  # If it is a line containing data of the coords

                        if split_line[0] == 0:

                            # Add coords to the latest dict which contains a measurement
                            data[record_time.year][record_time_list[-1].isoformat()][-1].update(
                                dict(
                                    zip(geo_coord_keys, split_line[1:])
                                )
                            )

                        elif split_line[0] == 1:
                            data[record_time.year][record_time_list[-1].isoformat()][-1].update(
                                dict(
                                    zip(aacgm_coord_keys, split_line[1:])
                                )
                            )

                        else:
                            logger.error(
                                f"""
                                This line contains an unexpected coord index:
                                {line}
                                """)

                    elif split_line.size == 1:
                        pass
                    else:
                        logger.error(
                            f"""
                            This line contains an unexpected number of columns ({split_line.size} columns):
                            {line}
                            """)

                # TODO: Better sepearte the fail conversion exception
                except Exception as e:
                    logger.error(
                        f"""
                        Line with non-numeric data found, when converting the data txt file into json files:
                        {line}
                        {e}
                        """
                    )
                    pass

    ##################################
    # Saving data to separate json files
    for year in list(data.keys()):

        out_json_path = json_out_dir / f'{year}_vorticity.json'

        with open(out_json_path, 'w') as f:
            json.dump(
                data[year],
                f,
                indent=4
            )
        logger.info(f'{out_json_path.name} saved to {str(out_json_path.parent)}')

    logger.info(f'Converting data file into json files took {(time() - t_start) / 60:.2f} minute(s)')

    return None
