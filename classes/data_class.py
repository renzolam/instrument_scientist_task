"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-01
Last Modified : 2024-08-10

Summary       : Class for holding a vorticity datapoint at a specified time and location

List of classes:
- VortMeasurement
"""

import logging
from datetime import datetime, timezone

import numpy as np

from common_utils import log_utils

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


class VortMeasurement:
    """
    Class for holding a vorticity datapoint at a specified time and location
    """

    r1_b1: float
    r1_b2: float
    r2_b1: float
    r2_b2: float
    area: float
    vorticity_mHz: float  # in the 'correct' sign convention, as recommended by the original txt data file
    MLT: float
    geo_lat_c: float
    geo_long_c: float
    geo_lat_1: float
    geo_long_1: float
    geo_lat_2: float
    geo_long_2: float
    geo_lat_3: float
    geo_long_3: float
    geo_lat_4: float
    geo_long_4: float
    aacgm_lat_c: float
    aacgm_long_c: float
    aacgm_lat_1: float
    aacgm_long_1: float
    aacgm_lat_2: float
    aacgm_long_2: float
    aacgm_lat_3: float
    aacgm_long_3: float
    aacgm_lat_4: float
    aacgm_long_4: float

    utc_time: datetime

    def __init__(self, datapoint: dict, iso_time_str: str):

        for key, value in datapoint.items():

            # Filters out invalid values
            if key == "vorticity_mHz" and (not (-1e3 < value < 1e3)):
                value = np.nan
            elif key == "MLT" and (not (0 <= value <= 24)):
                value = np.nan
            elif "_long_" in key and (not (-180 <= value <= 180)):
                value = np.nan
            elif "_lat_" in key and (not (-90 <= value <= 90)):
                value = np.nan
            else:
                pass

            setattr(self, key, value)

        self.utc_time = datetime.fromisoformat(iso_time_str).replace(
            tzinfo=timezone.utc
        )
