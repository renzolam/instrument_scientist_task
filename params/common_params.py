"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-12
Last Modified : 2024-08-18

Summary       : Contains parameters that are to be shared across scripts
"""

from datetime import datetime, timezone

from classes.main_runparams_cls import MainRunParams

# Set up the name of log file
log_dir = MainRunParams().output_dir / "logs"
if not log_dir.exists():
    log_dir.mkdir(parents=True)
log_abs_path = log_dir / f"SWIS_task.log"

# Sets up dir where data in json forms are found
json_out_dir = MainRunParams().output_dir / "vorticity_json"
if not json_out_dir.exists():
    json_out_dir.mkdir(parents=True)

# The dir where plots are stored
plot_dir = MainRunParams().output_dir / "plots"
if not plot_dir.exists():
    plot_dir.mkdir(parents=True)
