"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-12
Last Modified : 2024-08-15

Summary       : Provide functions that are necessary during plotting

List of functions:
- create_bin_edges
- mlt_to_phi
- lat_to_theta
- theta_to_lat
- _common_formatting
"""

import logging
from typing import Tuple, Union, List
import warnings

from matplotlib import axes
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from numpy.typing import NDArray

from common_utils import log_utils

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def create_bin_edges(
    lims: Union[Tuple[float, float], List[float]], bin_size: float
) -> NDArray:
    """
    Create an array of the values of the edges of the bins

    Parameters
    ----------
    lims: Tuple[float, float]
        Upper and lower bounds for a given dimension (e.g. for Phi in spherical coordinates, it is 0 to 2 pi)
    bin_size: float
        Size of the bins

    Returns
    -------
    NDArray
        Array of the values of the edges of the bins

    """

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
    Union[NDArray, float]
        Phi value(s)

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
    Union[NDArray, float]
        Theta values

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
    Union[NDArray, float]
        Latitude value(s)

    """

    lat = 90 - theta

    return lat


def _common_formatting(ax: axes, fontsize: float, max_theta: float) -> None:
    """
    Formatting which applies to all sub-plots that are in polar coordinates

    Parameters
    ----------
    ax: axes
        axes object to be formatted
    fontsize: float
        Size of fonts (in general)
    max_theta: float
        Max value of theta

    Returns
    -------
    """

    # Rotate plot to match conventional MLT representation
    ax.set_theta_zero_location("S")

    # Set lim in radial direction
    max_theta_for_plot = max_theta - (max_theta % 5) + 5
    ax.set_rlim(0, max_theta_for_plot)

    # Ticks

    # Ticks in MLT
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        ax.set_xticklabels(
            ["00\nMLT", "", "06\nMLT", "", "12\nMLT", "", "18\nMLT", ""],
            fontsize=fontsize,
        )

    # Ticks in latitude
    r_ticks = np.arange(0, max_theta_for_plot + 5, 5)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([int(lat) for lat in theta_to_lat(r_ticks)], fontsize=fontsize)

    # Set grid lines
    ax.grid(
        visible=True, which="both", axis="both", linestyle=":", alpha=1, color="black"
    )

    return None


def divergent_cmap():
    """
    Provide a divergent colormap that is readable

    Returns
    -------

    """

    # Colours at the left end, centre, and right end
    colour_min = '#fdbc7e'
    colour_centre = '#000000'
    colour_max = '#baffff'

    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap',
        [colour_min, colour_centre, colour_max],
    )

    return cmap
