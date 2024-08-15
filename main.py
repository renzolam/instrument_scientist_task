"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-07-31
Last Modified : 2024-08-15

Summary       : Main module for the analysis
"""

import logging
from time import time
from sys import getsizeof

import ray

from classes import main_runparams_cls
from classes.plot_params_cls import PlotParams
from common_utils import log_utils
from params import common_params
import read_data
import file_conversion
from plotting import avg_all, avg_by_season, sd_all, sd_by_season, r1_avg_vs_mlt
import season_analysis

if __name__ == "__main__":

    # Initialisation
    main_params = main_runparams_cls.MainRunParams()
    plot_params = PlotParams()

    logger = logging.getLogger(__name__)
    log_utils.set_logger(logger)

    t_start = time()
    ############################

    try:

        # Convert original txt data file into json files, if explicitly told to do so,
        # or there aren't any json files in the directory
        # This allows data to be read in parallel later on
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

        vort_by_season = season_analysis.separate_by_seasons(all_vort)  # Data separated by seasons

        ###################################
        # Plots in parallel
        ray.init()
        refs = [
            # Produces plots for mean, median and number of data points
            avg_all.plot_mean_median_counts.remote(
                plot_params, all_vort
            ),
            avg_by_season.plot_mean_median_counts.remote(
                plot_params, vort_by_season
            ),
            # Produces plots that show distribution of data
            sd_all.plot_sd_max_min.remote(
                plot_params, all_vort
            ),
            sd_by_season.plot.remote(
                plot_params, vort_by_season
            ),
            # Analyses the R1 vorticity
            r1_avg_vs_mlt.plot.remote(
                plot_params, vort_by_season
            ),
        ]
        ray.get(refs)
        ray.shutdown()

    except Exception as e:
        logger.exception(e)

    # End
    logger.info(f"The run took {(time() - t_start) / 60:.2f} minutes in total")
