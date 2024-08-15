"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-12
Last Modified : 2024-08-12

Summary       : Separates array of vorticity data by season

List of functions:
- separate_by_seasons
"""

import logging
from typing import Dict
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray
from skyfield import almanac
from skyfield.api import load

from common_utils import log_utils
from classes.data_class import VortMeasurement

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def separate_by_seasons(
    vort_array: NDArray[VortMeasurement],
) -> Dict[str, NDArray[VortMeasurement]]:
    """
    Separates array of vorticity data by season

    Parameters
    ----------
    vort_array: NDArray[VortMeasurement]

    Returns
    -------
    Dict[str, NDArray[VortMeasurement]]
        Array of vorticity data separated by season
    """

    # Initialisation
    ts = load.timescale()  # Create timescale object
    eph = load("de421.bsp")  # Loads JPL ephemeris (covers 1900 - 2050)

    all_timestamps = np.array(
        [datetime.timestamp(vort_data.utc_time) for vort_data in vort_array]
    )
    all_years = np.unique([vort_data.utc_time.year for vort_data in vort_array])

    winter_data, spring_data, summer_data, autumn_data = [[] for _ in range(4)]
    ############################
    for year in all_years:
        first_day = ts.utc(year, 1, 1)
        lasy_day = ts.utc(year, 12, 31)

        season_start_t, _ = almanac.find_discrete(
            first_day, lasy_day, almanac.seasons(eph)
        )  # Find when did the seasons change

        season_edges = [
            datetime.timestamp(
                datetime.fromisoformat(ti.utc_iso()[:-1]).replace(tzinfo=timezone.utc)
            )
            for ti in season_start_t
        ]
        season_edges.insert(
            0, datetime.timestamp(datetime(year, 1, 1, 0, 0, tzinfo=timezone.utc))
        )
        season_edges.append(
            datetime.timestamp(datetime(year + 1, 1, 1, 0, 0, tzinfo=timezone.utc))
        )

        which_season = np.digitize(all_timestamps, season_edges, right=False)

        for i, season_data in enumerate(
            (winter_data, spring_data, summer_data, autumn_data)
        ):
            season_data.append(vort_array[which_season == i + 1])
        winter_data.append(vort_array[which_season == 5])

    vort_by_season = dict(
        spring=np.concatenate(spring_data),
        summer=np.concatenate(summer_data),
        autumn=np.concatenate(autumn_data),
        winter=np.concatenate(winter_data),
    )

    return vort_by_season
