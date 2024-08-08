"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-07-31
Last Modified : 2024-08-01

Summary       : Main module for the analysis
"""

import logging
from time import time
from sys import getsizeof, exit

import numpy as np

from classes import main_runparams_cls
from classes.map_params_cls import MapParams
from common_utils import log_utils
import read_data
import file_conversion
from plotting import avg_median_map

if __name__ == "__main__":

    # Initialisation
    main_runparams = main_runparams_cls.MainRunParams()
    map_params = MapParams()

    logger = logging.getLogger(__name__)
    log_utils.set_logger(logger)

    t_start = time()
    ############################

    try:

        # Convert original txt data file into json files, if explicitly told to do so,
        # or there aren't any json files in the directory
        if (not main_runparams.use_existing_json_data
                or len(list(main_runparams_cls.json_out_dir.glob('*vorticity.json'))) == 0):

            file_conversion.data_to_json(main_runparams)
        else:
            pass

        # Store all data in 1 numpy array
        vort_array = read_data.json_to_array()

        # Ensure all data has been read in
        # if np.isnan(vort_array).any():
        #     logger.exception('There was a problem reading in the json files. See log file for details.')
        #     exit()
        # else:
        #     pass

        logger.info(f'Memory occupied by all vorticity data is {getsizeof(vort_array) / 1e6} MB')

        ###################################
        # Produce plot for map of mean and median values
        avg_median_map.plot(map_params, vort_array)

    except Exception as e:
        logger.exception(e)

    # End
    logger.info(f'The run took {(time() - t_start) / 60:.2f} minutes in total')
