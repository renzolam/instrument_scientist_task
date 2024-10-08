"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-12
Last Modified : 2024-08-18

Summary       : Plots the standard distribution (s.d.), min and max absolute values at different MLT and latitudes for
all data

List of functions:
- _ax_formatting
- _fig_formatting
- _plot_subplot
- plot_mean_median_counts
"""

import logging
from copy import deepcopy
from typing import Dict

import matplotlib.pyplot as plt

import ray
from matplotlib import colors, figure, axes
from matplotlib.collections import QuadMesh
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d

from common_utils import log_utils, plot_utils
from classes.plot_params_cls import PlotParams
from classes.data_class import VortMeasurement
from params import common_params

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)

stat_type_full_name = {
    "std": "standard deviation",
    "min": "min magnitude",
    "max": "max magnitude",
}


def _ax_formatting(
    fig: figure,
    ax: axes,
    plot_to_format: QuadMesh,
    stat_type: str,
    coord: str,
    fontsize: float,
) -> None:
    """
    Formatting which applies to sub-plots specifically

    Parameters
    ----------
    fig: figure
        figure object to be formatted
    ax: axes
        axes object to be formatted
    plot_to_format: QuadMesh
        'Image' object to be associated with the colorbar
    stat_type: str
        Is it mean, median or counts?
    coord: str
        Coordinate system used for latitudes
    fontsize: float
        Size of fonts (in general)

    Returns
    -------

    """

    # Initialisation

    assert stat_type in ("std", "max", "min")

    label_dict = {
        "std": "Standard Deviation (mHz)",
        "max": "Max Vorticity Magnitudes (mHz)",
        "min": "Min Vorticity Magnitudes (mHz)",
    }

    ticks_dict = {
        "std": np.arange(-80, 80, 1),
        "max": np.arange(0, 90, 20),
        "min": np.arange(0, 0.1, 0.01),
    }

    ####################
    # Does the formatting

    # For the colorbar
    cbar = fig.colorbar(
        plot_to_format,
        ax=ax,
        orientation="horizontal",
        location="top",
        aspect=15,
        ticks=ticks_dict[stat_type],
    )

    # Set minor ticks
    cbar.ax.minorticks_on()

    # Format the colorbar
    cbar.ax.tick_params(
        labelsize=fontsize, length=fontsize / 1.25, width=fontsize / 6, which="major"
    )
    cbar.ax.tick_params(
        labelsize=fontsize, length=fontsize / 2.5, width=fontsize / 10, which="minor"
    )

    cbar.ax.set_title(label_dict[stat_type], fontsize=fontsize, pad=fontsize)

    # Label for radial axis
    label_position = ax.get_rlabel_position()
    ax.text(
        np.radians(label_position + 10),
        ax.get_rmax() * 1.1,
        f"{coord.upper()} Latitude",
        ha="left",
        va="top",
        fontsize=fontsize,
    )

    return None


def _fig_formatting(
    fig: figure, all_vort: NDArray[VortMeasurement], fontsize: float
) -> None:
    """
    Does formatting that is not specific to any subplot

    Parameters
    ----------
    fig: figure
        figure object to be formatted
    all_vort: NDArray[VortMeasurement]
        Array of all vorticity data points
    fontsize: float
        Size of fonts (in general)

    Returns
    -------

    """

    all_t = np.array([vort.utc_time for vort in all_vort])
    min_year = np.nanmin(all_t).year
    max_year = np.nanmax(all_t).year

    fig.suptitle(
        f"""
        Standard Deviation (s.d.), 
        Min and Max Absolute Values 
        of Vorticity Measurements
        in the Northern Hemisphere
        During {min_year} - {max_year}
        """,
        fontsize=fontsize,
        horizontalalignment="center",
        verticalalignment="center",
        position=(0.45, 0.9),
    )

    return None


def _plot_subplot(
    phi_edges: NDArray,
    theta_edges: NDArray,
    fig: figure,
    ax: axes,
    stat_data: Dict[str, NDArray],
    stat_type: str,
    max_theta: float,
    coord: str,
    fontsize: float,
) -> None:
    """
    Plots a subplot

    Parameters
    ----------
    phi_edges: NDArray
        Boundary values for the bins in phi
    theta_edges: NDArray
        Boundary values for the bins in theta
    fig: figure
        figure object to be formatted
    ax: axes
        axes object to be formatted
    stat_data: Dict[str, NDArray]
        Dictionary of statistics data (mean, median, counts)
    stat_type: str
        Which statistic should be plotted
    max_theta: float
        Max value of theta
    coord: str
        Coordinate system used for latitudes
    fontsize: float
        Size of fonts (in general)

    Returns
    -------

    """

    # Initialisation
    assert stat_type in ("std", "min", "max")

    colour_map_dict = {"std": "jet", "max": "jet", "min": "jet"}

    # Normalise data for the colorbar if needed
    stat_min = np.nanmin(stat_data[stat_type])
    stat_max = np.nanmax(stat_data[stat_type])
    norm = colors.Normalize(vmin=stat_min, vmax=stat_max)

    logger.info(
        f"""When plotting for the entire dataset,
    the highest value for {stat_type_full_name[stat_type]} is {stat_max:.2f} mHz,
    the lowest value for {stat_type_full_name[stat_type]} is {stat_min:.2f} mHz"""
    )

    #########################
    # Actual plotting
    plot = ax.pcolormesh(
        *np.meshgrid(phi_edges, theta_edges),
        stat_data[stat_type].T,
        cmap=colour_map_dict[stat_type],
        norm=norm,
    )
    plot_utils._common_formatting(ax, fontsize, max_theta)
    _ax_formatting(fig, ax, plot, stat_type, coord, fontsize)

    return None


def plot_sd_max_min(
    plot_params: PlotParams,
    vort_array: NDArray,
    coord: str = "aacgm",
    fontsize=40,
):
    """
    Plot the standard distribution (s.d.), min and max absolute values at different MLT and latitudes for
    all data

    In order to make the data plottable on a polar projection,
    - MLT is converted to phi (radians), which is the 'azimuthal angle', aka the angle
    of rotation of the radial line around the polar axis
    - latitude (degrees) is converted to theta (degrees), which is the 'polar angle', aka
    the angle between the radial line and the polar axis
    - (The radial line is the straight line connecting the origin and the data point)
    - (Theta and phi are defined according to the convention used mainly by physicists)

    Parameters
    ----------
    plot_params: PlotParams
        Used here for knowing the bin sizes to use for the plot, and for the count cutoff. Bins with fewer data points
        than this cutoff will not be plotted
    vort_array: List[VortMeasurement]
        List of VortMeasurement objects, each of which contain data for a measurement made
    coord: str
        Coordinate system to be used for the latitude. Only accepts AACGM or GEO
    fontsize: float
        Size of most words which appear on the plot

    Returns
    -------

    """

    logger = logging.getLogger(__name__ + ".plot_sd_max_min")
    log_utils.set_logger(logger)

    if coord not in ("aacgm", "geo"):
        raise ValueError('Coord must be either "aacgm" or "geo"')
    else:
        pass

    # Filtering out data that covers areas larger than the cutoff size
    area_data = np.array(
        [vort_measurement.area_km2 for vort_measurement in vort_array]
    )

    vort_array = vort_array[area_data <= plot_params.area_km2_cuttoff]

    # Extracts data
    phi_coords = plot_utils.mlt_to_phi(
        np.array([vort_data.MLT for vort_data in vort_array])
    )
    theta_coords = plot_utils.lat_to_theta(
        np.array([getattr(vort_data, f"{coord}_lat_c") for vort_data in vort_array])
    )
    vort_data = np.array(
        [vort_measurement.vorticity_mHz for vort_measurement in vort_array]
    )
    abs_vort_data = np.abs(vort_data)

    ####################
    # Creates bin edges

    # Bin sizes
    d_phi_rad = plot_utils.mlt_to_phi(plot_params.mlt_bin_size_hr)
    d_theta_deg = deepcopy(plot_params.lat_bin_size_degree)

    # All edges of the bins for PHI
    phi_edges = plot_utils.create_bin_edges((0, 2 * np.pi), d_phi_rad)

    # All edges of the bins for THETA
    min_lat = np.min(
        np.array([getattr(vort_data, f"{coord}_lat_c") for vort_data in vort_array])
    )
    min_lat_edge = (
        min_lat
        - (min_lat % plot_params.lat_bin_size_degree)
        + plot_params.lat_bin_size_degree
    )
    max_theta = 90 - min_lat_edge

    theta_edges = plot_utils.create_bin_edges((0, max_theta), d_theta_deg)
    ####################
    # Does the calculations

    stat_data = dict(
        (
            stat_type,
            binned_statistic_2d(
                phi_coords,
                theta_coords,
                vort_data,
                statistic=stat_type,
                bins=(phi_edges, theta_edges),
            ).statistic,
        )
        for stat_type in ("std", "count")
    )

    for stat_type in ("min", "max"):
        stat_data[stat_type] = binned_statistic_2d(
            phi_coords,
            theta_coords,
            abs_vort_data,
            statistic=stat_type,
            bins=(phi_edges, theta_edges),
        ).statistic

    assert not np.isnan(
        stat_data["count"]
    ).any()  # Assert there aren't any invalid values in the counts

    # Do not plot bins with fewer counts than a threshold (100 by default)
    for stat_type in ("std", "max", "min"):
        stat_data[stat_type][stat_data["count"] < plot_params.count_cutoff] = np.nan

    # Do not plot bins that have 0 counts
    stat_data["count"][stat_data["count"] == 0] = np.nan

    ####################
    # Setting up the plotting routine
    fig, axs = plt.subplots(1, 3, figsize=(36, 38), subplot_kw={"projection": "polar"})

    # Plots the data
    for column_idx, stat_type in enumerate(("std", "max", "min")):
        _plot_subplot(
            phi_edges,
            theta_edges,
            fig,
            axs[column_idx],
            stat_data,
            stat_type,
            max_theta,
            coord,
            fontsize,
        )

    # Does more formatting
    _fig_formatting(fig, vort_array, fontsize)
    fig.tight_layout()

    plt.savefig(
        common_params.plot_dir / "sd_abs_max_min_(all_data).png", bbox_inches="tight"
    )

    return None
