import logging
from datetime import datetime, timedelta, timezone

from common_utils import log_utils
from classes import main_runparams_cls

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


class vort_measurement:

    r1_b1: float
    r1_b2: float
    r2_b1: float
    r2_b2: float
    area: float
    vorticity_mHz: float
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

    timestamp: datetime

    def __init__(self, datapoint: dict, iso_time_str: str):

        for key, value in datapoint.items():
            setattr(self, key, value)

        self.timestamp = datetime.fromisoformat(iso_time_str).replace(tzinfo=timezone.utc)
