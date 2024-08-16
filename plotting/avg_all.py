"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-10
Last Modified : 2024-08-16

Summary       : Plots the mean, median and number of data points for all data

List of functions:
- _ax_formatting
- _fig_formatting
- _plot_subplot
- plot_mean_median_counts
"""

import logging
from copy import deepcopy, copy
from typing import Dict

import ray
import matplotlib.pyplot as plt
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


def _ax_formatting(
    fig: figure,
    ax: axes,
    plot_to_format: QuadMesh,
    plot_type: str,
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
    plot_type: str
        Is it mean, median or counts?
    coord: str
        Coordinate system used for latitudes
    fontsize: float
        Size of fonts (in general)

    Returns
    -------

    """

    # Initialisation

    assert plot_type in ("mean", "median", "count")

    label_dict = {
        "mean": "Mean Vorticity (mHz)",
        "median": "Median Vorticity (mHz)",
        "count": "Number of Data Points",
    }

    major_ticks_dict = {
        "mean": np.arange(-4, 4, 1, dtype=float),
        "median": np.arange(-3.5, 3.5, 0.5, dtype=float),
        "count": np.power(10, range(0, 10)),
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
        ticks=major_ticks_dict[plot_type],
    )

    # Set minor ticks
    cbar.ax.minorticks_on()

    # Format the colorbar
    cbar.ax.tick_params(
        labelsize=fontsize,
        length=fontsize / 1.25,
        width=fontsize / 6,
        which="major"
    )
    cbar.ax.tick_params(
        labelsize=fontsize,
        length=fontsize / 2.5,
        width=fontsize / 10,
        which="minor"
    )

    cbar.ax.set_title(label_dict[plot_type], fontsize=fontsize, pad=fontsize)

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
        Mean, Median, and Number of Data Points 
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
    ax_to_plot: axes,
    stat_data: Dict[str, NDArray],
    stat_type: str,
    max_theta: float,
    coord: str,
    fontsize: float
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
    ax_to_plot: axes
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
    assert stat_type in ("mean", "median", "count")

    ax = copy(ax_to_plot)

    colour_map_dict = {"mean": "RdBu", "median": "RdBu", "count": "jet"}

    # Normalise data for the colorbar if needed
    if stat_type in ("mean", "median"):
        data_min = np.nanmin(stat_data[stat_type])
        data_max = np.nanmax(stat_data[stat_type])

        # Make colorbar symmetrical about 0
        abs_biggest = np.max([np.abs(data_min), np.abs(data_max)])

        norm = colors.Normalize(vmin=-abs_biggest, vmax=abs_biggest)
    elif stat_type == "count":
        plot_cbar_min = np.power(10, np.floor(np.log10(np.nanmin(stat_data["count"]))))
        plot_cbar_max = np.power(10, np.ceil(np.log10(np.nanmax(stat_data["count"]))))

        norm = colors.LogNorm(vmin=plot_cbar_min, vmax=plot_cbar_max)
    else:
        raise ValueError(f"Plot_type {stat_type} not recognized")
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


@ray.remote
def plot_mean_median_counts(
    plot_params: PlotParams,
    vort_array: NDArray,
    coord: str = "aacgm",
    count_cutoff: int = 100,
    fontsize=40,
):
    """
    Plot the mean, median and number of data points for all data

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
        Used here for knowing the bin sizes to use for the plot
    vort_array: List[VortMeasurement]
        List of VortMeasurement objects, each of which contain data for a measurement made
    coord: str
        Coordinate system to be used for the latitude. Only accepts AACGM or GEO
    count_cutoff: int
        Bins with fewer data points than this cutoff will not be plotted
    fontsize: float
        Size of most words which appear on the plot

    Returns
    -------

    """

    logger = logging.getLogger(__name__ + ".plot_mean_median_counts")
    log_utils.set_logger(logger)

    if coord not in ("aacgm", "geo"):
        raise ValueError('Coord must be either "aacgm" or "geo"')
    else:
        pass

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
            stat,
            binned_statistic_2d(
                phi_coords,
                theta_coords,
                vort_data,
                statistic=stat,
                bins=(phi_edges, theta_edges),
            ).statistic,
        )
        for stat in ("mean", "median", "count")
    )

    assert not np.isnan(
        stat_data["count"]
    ).any()  # Assert there aren't any invalid values in the counts

    # Do not plot bins with fewer counts than a threshold (100 by default)
    stat_data["mean"][stat_data["count"] < count_cutoff] = np.nan
    stat_data["median"][stat_data["count"] < count_cutoff] = np.nan

    # Do not plot bins that have 0 counts
    stat_data["count"][stat_data["count"] == 0] = np.nan

    ####################
    # Setting up the plotting routine
    fig, axs = plt.subplots(1, 3, figsize=(36, 21), subplot_kw={"projection": "polar"})

    # Plots the data
    for column_idx, stat_type in enumerate(("mean", "median", "count")):
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

    # Saving the file
    plt.savefig(
        common_params.plot_dir / "avg_median_counts_(all_data).png", bbox_inches="tight"
    )

    return None
