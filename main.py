"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-07-31
Last Modified : 2024-07-31

Summary       : Main module for the analysis
"""

import logging

from classes import main_runparams_cls
from common_utils import log_utils
import file_conversion


if __name__ == "__main__":

    # Initialisation
    main_runparams = main_runparams_cls.MainRunParams()

    logger = logging.getLogger(__name__)
    log_utils.set_logger(logger)

    file_conversion.data_to_json(main_runparams)
