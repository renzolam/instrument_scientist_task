"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-07-31
Last Modified : 2024-07-31

Summary       : Class for holding data of the parameters of the run

List of classes:
- MainRunParams
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List
from copy import deepcopy


class MainRunParams:
    """
    Class containing all info for the run
    """

    abs_data_file_path: Path  # The absolute path of the downloaded data file
    output_dir: Path  # Where all the generated files go to

    def __init__(self):

        json_path = Path(__file__).parent.parent / "runparams" / "main_runparams.json"

        with open(json_path, "r") as f:
            paramsdict = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

        # Automatically turns the data of the param json file into ths class' attributes
        for attr in dir(paramsdict):

            # Only go through attributes present in the json file
            if not attr.startswith("__") and not attr.endswith("__"):

                attr_2_set = getattr(paramsdict, attr)

                # Turns dirs and paths into 'Path' Python objects
                if attr.endswith("dir") or attr.endswith("path"):

                    attr_2_set = Path(attr_2_set)

                    # Create directory if it does not exist
                    if not attr_2_set.exists() and attr.endswith("dir"):
                        attr_2_set.mkdir(parents=True)
                    else:
                        pass

                # Stores all other params in the same format as stored in json file
                else:
                    pass

                setattr(self, attr, attr_2_set)


# Set up the name of log file
log_dir = MainRunParams().output_dir / "logs"
if not log_dir.exists():
    log_dir.mkdir(parents=True)
log_abs_path = log_dir / f"SWIS_task_{datetime.now(timezone.utc):%Y%m%d_%H%M}_UT.log"

# Sets up dir where data in json forms are found
json_out_dir = MainRunParams().output_dir / 'vorticity_json'
