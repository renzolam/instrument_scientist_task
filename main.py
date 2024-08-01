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
from sys import getsizeof

from classes import main_runparams_cls
from common_utils import log_utils
import read_data
import file_conversion

if __name__ == "__main__":

    # Initialisation
    main_runparams = main_runparams_cls.MainRunParams()

    logger = logging.getLogger(__name__)
    log_utils.set_logger(logger)

    t_start = time()
    ############################

    # Convert original txt data file into json files, if explicitly told to do so,
    # or there aren't any json files in the directory
    if (not main_runparams.use_existing_json_data
            or len(list(main_runparams_cls.json_out_dir.glob('*vorticity.json'))) == 0):

        try:
            file_conversion.data_to_json(main_runparams)
        except Exception as e:
            logger.exception(e)

    else:
        pass

    # Store all data in 1 numpy array
    vort_array = read_data.json_to_array()
    logger.info(f'Memory occupied by all vorticity data is {getsizeof(vort_array) / 1e6} MB')

    logger.info(f'The run took {(time() - t_start) / 60:.2f} minutes in total')
