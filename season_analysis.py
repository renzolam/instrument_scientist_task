"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 
Last Modified : 

Summary       : 
"""

import logging
from typing import List
from datetime import datetime, timezone

import numpy as np
from skyfield import almanac
from skyfield.api import N, S, E, W, load, wgs84

from common_utils import log_utils
from classes.data_class import VortMeasurement

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def separate_by_seasons(vort_array: List[VortMeasurement]):

    # Initialisation
    ts = load.timescale()
    eph = load('de421.bsp')

    all_times = np.array([vort_data.utc_time for vort_data in vort_array])
    all_years = np.unique([vort_data.utc_time.year for vort_data in vort_array])

    for year in all_years:
        first_day = ts.utc(year, 1, 1)
        lasy_day = ts.utc(year, 12, 31)

        t, _ = almanac.find_discrete(first_day, lasy_day, almanac.seasons(eph))
        season_edges = [
            datetime.fromisoformat(ti.utc_iso()[:-1]).replace(tzinfo=timezone.utc)
            for ti in t]

    return
