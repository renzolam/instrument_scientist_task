"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-07-31
Last Modified : 2024-08-12

Summary       : Main module for the analysis
"""

import logging
from time import time
from sys import getsizeof

from classes import main_runparams_cls
from classes.map_params_cls import MapParams
from common_utils import log_utils
from runparams import common_params
import read_data
import file_conversion
from plotting import plot_avg_all, plot_avg_by_season, plot_sd_all, plot_sd_by_season
import season_analysis

if __name__ == "__main__":

    # Initialisation
    main_params = main_runparams_cls.MainRunParams()
    map_params = MapParams()

    logger = logging.getLogger(__name__)
    log_utils.set_logger(logger)

    t_start = time()
    ############################

    try:

        # Convert original txt data file into json files, if explicitly told to do so,
        # or there aren't any json files in the directory
        if (
            main_params.txt_files_to_json
            or len(list(common_params.json_out_dir.glob("*vorticity.json"))) == 0
        ):
            file_conversion.all_data_to_json(main_params)
        else:
            pass

        # Store all data in 1 numpy array
        all_vort = read_data.json_to_array()

        logger.info(
            f"Memory occupied by all vorticity data is {getsizeof(all_vort) / 1e6:.2f} MB"
        )

        vort_by_season = season_analysis.separate_by_seasons(all_vort)

        ###################################
        # Produces plots for mean, median and number of data points
        # plot_avg_all.plot_mean_median_counts(main_params, map_params, all_vort)
        #
        # plot_avg_by_season.plot_mean_median_counts(main_params, map_params, vort_by_season)

        # Produces plots that show distribution of data
        # plot_sd_all.plot_sd_max_min(main_params, map_params, all_vort)
        plot_sd_by_season.plot(main_params, map_params, vort_by_season)

    except Exception as e:
        logger.exception(e)

    # End
    logger.info(f"The run took {(time() - t_start) / 60:.2f} minutes in total")
