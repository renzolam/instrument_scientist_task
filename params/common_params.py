"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-12
Last Modified : 2024-08-12

Summary       : Contains parameters that are to be shared across scripts
"""

from datetime import datetime, timezone

from common_utils import log_utils
from classes.main_runparams_cls import MainRunParams

# Set up the name of log file
log_dir = MainRunParams().output_dir / "logs"
if not log_dir.exists():
    log_dir.mkdir(parents=True)
log_abs_path = log_dir / f"SWIS_task_{datetime.now(timezone.utc):%Y%m%d_%H%M}_UT.log"

# Sets up dir where data in json forms are found
json_out_dir = MainRunParams().output_dir / "vorticity_json"
