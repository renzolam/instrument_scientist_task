"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-07-31
Last Modified : 2024-07-31

Summary       : Main module for the analysis

List of functions:
- set_loggers
"""

import logging

from classes import main_runparams_cls


def set_loggers(main_params: main_runparams_cls.MainRunParams) -> None:
    """

    Sets up logging for the project, so that all logs are both in a file, and streamed to the console

    Parameters
    ----------
    main_params: main_runparams_cls.MainRunParams
        Parameters for the run. Used here to get the dir the log file should be in

    Returns
    -------

    """

    # Setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(main_params.log_dir / "SWIS_task.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return None


if __name__ == "__main__":

    main_runparams = main_runparams_cls.MainRunParams()

    set_loggers(main_runparams)
