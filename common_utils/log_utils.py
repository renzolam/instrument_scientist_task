"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-01
Last Modified : 2024-08-15

Summary       : Sets up loggers
"""

import logging

from params import common_params


def set_logger(logger_2_set: logging.Logger) -> None:
    """
    Sets up logger objects so that logs will go to both the console and a file

    Parameters
    ----------
    logger_2_set: logging.Logger
        logger object to be set

    Returns
    -------

    """

    logger_2_set.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "\n%(asctime)s - %(name)s line %(lineno)s - %(levelname)s: \n\n\t%(message)s"
    )

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(common_params.log_abs_path, mode="a+")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger_2_set.addHandler(file_handler)
    logger_2_set.addHandler(console_handler)

    return None
