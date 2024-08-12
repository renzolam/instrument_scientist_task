"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-07-31
Last Modified : 2024-08-12

Summary       : Class for holding data of the parameters of the run

List of classes:
- MainRunParams
"""

import json
from pathlib import Path
from types import SimpleNamespace


class MainRunParams:
    """
    Class containing all info for the run
    """

    abs_data_txt_dir: Path  # The absolute path of the downloaded data file
    output_dir: Path  # Where all the generated files go to

    # Whether to use the json files which have been converted from the original txt file
    # If false, will re-run the function where the txt file will be converted into a series of json files
    txt_files_to_json: bool

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
                if attr.endswith("dir"):

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
