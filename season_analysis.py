"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 
Last Modified : 

Summary       : 
"""

import logging
from copy import copy
from typing import List, Tuple
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray
from skyfield import almanac
from skyfield.api import N, S, E, W, load, wgs84

from common_utils import log_utils
from classes.data_class import VortMeasurement

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def separate_by_seasons(
        vort_array: NDArray[VortMeasurement]
) -> Tuple[NDArray[VortMeasurement], NDArray[VortMeasurement], NDArray[VortMeasurement], NDArray[VortMeasurement]]:

    # Initialisation
    ts = load.timescale()
    eph = load('de421.bsp')

    all_timestamps = np.array([
        datetime.timestamp(
            vort_data.utc_time)
        for vort_data in vort_array
    ])
    all_years = np.unique([vort_data.utc_time.year for vort_data in vort_array])

    winter_data, spring_data, summer_data, autumn_data = [[] for _ in range(4)]
    ############################
    for year in all_years:
        first_day = ts.utc(year, 1, 1)
        lasy_day = ts.utc(year, 12, 31)

        t, _ = almanac.find_discrete(first_day, lasy_day, almanac.seasons(eph))

        season_edges = [
            datetime.timestamp(
                datetime.fromisoformat(ti.utc_iso()[:-1]).replace(tzinfo=timezone.utc)
            )
            for ti in t]
        season_edges.insert(
            0,
            datetime.timestamp(
                datetime(year, 1, 1, 0, 0, tzinfo=timezone.utc)
            )
        )
        season_edges.append(
            datetime.timestamp(
                datetime(year + 1, 1, 1, 0, 0, tzinfo=timezone.utc)
            )
        )

        which_season = np.digitize(all_timestamps, season_edges, right=False)

        for i, season_data in enumerate((winter_data, spring_data, summer_data, autumn_data)):
            season_data.append(vort_array[which_season == i+1])
        winter_data.append(vort_array[which_season == 5])

    winter_data = np.concatenate(winter_data)
    spring_data = np.concatenate(spring_data)
    summer_data = np.concatenate(summer_data)
    autumn_data = np.concatenate(autumn_data)

    return winter_data, spring_data, summer_data, autumn_data
