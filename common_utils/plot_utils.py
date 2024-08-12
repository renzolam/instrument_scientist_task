"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-12
Last Modified : 2024-08-12

Summary       : Provide functions that are necessary during plotting
"""

import logging
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from common_utils import log_utils

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def create_bin_edges(
        lims: Tuple[float, float],
        bin_size: float
) -> NDArray:
    edges = np.arange(lims[0], lims[1] + bin_size, bin_size)

    return edges


def mlt_to_phi(mlt: Union[NDArray, float]) -> Union[NDArray, float]:
    """
    Converts MLT values to phi values, defined as
    angles in RADIANS from the magnetic midnight (facing away from the sun)

    Parameters
    ----------
    mlt: Union[NDArray, float]
        MLT values

    Returns
    -------

    """

    phi = (mlt / 24) * (2 * np.pi)

    return phi


def lat_to_theta(lat: Union[NDArray, float]) -> Union[NDArray, float]:
    """
        Converts latitude values to theta values, defined as
        angles in DEGREES from the magnetic pole

        Parameters
        ----------
        lat: Union[NDArray, float]
            latitude values

        Returns
        -------

        """

    theta = 90 - lat

    return theta


def theta_to_lat(theta: Union[NDArray, float]) -> Union[NDArray, float]:
    """
        Converts theta values (see below) to latitude values

        Parameters
        ----------
        theta: Union[NDArray, float]
            angles in DEGREES from the magnetic pole

        Returns
        -------

        """

    lat = 90 - theta

    return lat
