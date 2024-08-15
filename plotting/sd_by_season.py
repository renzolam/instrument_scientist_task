"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-14
Last Modified : 2024-08-15

Summary       : Plots the standard distribution (s.d.), min and max absolute values at different MLT and latitudes
according to their seasons

List of functions:
- _ax_formatting
- _fig_formatting
- _find_min_max_for_colorbar
- __plot_subplot
- _plot_1_season
- plot_mean_median_counts
"""

import logging
from typing import Dict
from copy import deepcopy

import matplotlib.pyplot as plt
import ray
from matplotlib import colormaps, colors, axes, figure
from matplotlib.collections import QuadMesh
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d

from common_utils import log_utils, plot_utils
from classes.plot_params_cls import PlotParams
from classes.main_runparams_cls import MainRunParams
from classes.data_class import VortMeasurement
from params import common_params

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def _ax_formatting(
    fig: figure,
    axs: NDArray[NDArray[axes]],
    plot_to_format: QuadMesh,
    row_idx: int,
    column_idx: int,
    fontsize: float,
    coord: str = "aacgm",
) -> None:
    """
    Formatting which applies to sub-plots

    Parameters
    ----------
    fig: figure
        fig object to be formatted
    axs: NDArray[NDArray[axes]]
        List of axes objects from matplotlib
    plot_to_format: QuadMesh
        The 'image' related to the colorbar
    row_idx: int
        Which row is it
    column_idx: int
        Which column is it
    fontsize: float
        Size of fonts (in general)
    coord: str = 'aacgm'
        Which coordinate system is used for latitudes


    Returns
    -------

    """

    label_dict = {
        0: "Standard Deviations (mHz)",
        1: "Max Vorticity Magnitudes (mHz)",
        2: "Min Vorticity Magnitudes (mHz)",
    }

    ticks_dict = {
        0: np.arange(1, 15, 1),
        1: np.arange(0, 90, 10),
        2: np.arange(0, 0.3, 0.02),
    }

    season_names = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}

    ax: axes = axs[row_idx][column_idx]

    ####################
    # Does the formatting

    if row_idx == 1:

        # Plots colorbar only for the 1st row
        cax = axs[0][column_idx]
        cax.set_visible(False)
        cbar = fig.colorbar(
            plot_to_format,
            ax=cax,
            orientation="horizontal",
            location="top",
            fraction=1,
            aspect=25,
            ticks=ticks_dict[column_idx],
        )
        cbar.ax.tick_params(labelsize=fontsize, length=fontsize / 2, width=fontsize / 6)
        cbar.ax.set_title(
            label_dict[column_idx], fontsize=fontsize, fontweight="bold", pad=fontsize
        )

        # Only label the radial axis for the 1st row to avoid unnecessary duplications
        label_position = ax.get_rlabel_position()
        ax.text(
            np.radians(label_position + 10),
            ax.get_rmax() * 1.1,
            f"{coord.upper()} Latitude",
            ha="left",
            va="top",
            fontsize=fontsize,
        )
    else:
        pass

    # Add subtitles to indicate season
    if column_idx == 1:
        ax.set_title(
            season_names[row_idx],
            fontsize=fontsize * 1.5,
            pad=fontsize * 3,
            fontweight="bold",
        )
    else:
        pass

    return None


def _fig_formatting(
    fig: figure, all_vort: NDArray[VortMeasurement], fontsize: float
) -> None:
    """
    Does formatting that is not specific to any subplot

    Parameters
    ----------
    fig: figure
        fig object to be formatted
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
        (Separated by Seasons)
        During the Period {min_year} - {max_year}
        """,
        fontsize=fontsize * 1.5,
        horizontalalignment="center",
        verticalalignment="center",
        position=(0.45, 0.97),
    )

    return None


def _find_min_max_for_colorbar(
    phi_edges: NDArray,
    theta_edges: NDArray,
    vort_by_season: Dict[str, NDArray[VortMeasurement]],
    coord: str,
    count_cutoff: float,
    vort_cbar_step: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Determines the minimum and maximum limits for the colorbars

    Parameters
    ----------
    phi_edges: NDArray
        Boundary values for the bins in phi
    theta_edges: NDArray
        Boundary values for the bins in theta
    vort_by_season: Dict[str, NDArray[VortMeasurement]]
        Dictionary of Vorticity Measurements, by season
    coord: str
        Coordinate system for latitudes
    count_cutoff: float
        Count threshold for bins. If below this threshold, vorticity will not be plotted for this bin
    vort_cbar_step: float
        Round up/ down the max/ min value of vorticity to the nearest vort_cbar_step value
        e.g. vort_cbar_step = 0.5, then 4.3 will be rounded up to 4.5, and -0.3 to -0.5

    Returns
    -------
    Dictionary containing the minimum and maximum limits for the colorbars

    """

    seasonal_min_max_dict = dict(
        (stat_type, {"min": [], "max": []}) for stat_type in ("std", "max", "min")
    )

    for vort_1_season in [
        vort_by_season[season] for season in ("spring", "summer", "autumn", "winter")
    ]:
        phi_coords = plot_utils.mlt_to_phi(
            np.array([vort_data.MLT for vort_data in vort_1_season])
        )
        theta_coords = plot_utils.lat_to_theta(
            np.array(
                [getattr(vort_data, f"{coord}_lat_c") for vort_data in vort_1_season]
            )
        )
        vort_season = np.array(
            [vort_measurement.vorticity_mHz for vort_measurement in vort_1_season]
        )
        abs_vort_season = np.abs(
            [vort_measurement.vorticity_mHz for vort_measurement in vort_1_season]
        )
        ###################
        # Calculating the stats for each season
        seasonal_data = dict(
            (
                stat_type,
                binned_statistic_2d(
                    phi_coords,
                    theta_coords,
                    vort_season,
                    statistic=stat_type,
                    bins=(phi_edges, theta_edges),
                ).statistic,
            )
            for stat_type in ("std", "count")
        )
        for stat_type in ("max", "min"):
            seasonal_data[stat_type] = binned_statistic_2d(
                phi_coords,
                theta_coords,
                abs_vort_season,
                statistic=stat_type,
                bins=(phi_edges, theta_edges),
            ).statistic

        # Filtering out unwanted data
        for stat_type in ("std", "max", "min"):
            seasonal_data[stat_type][seasonal_data["count"] < count_cutoff] = np.nan
        seasonal_data["count"][seasonal_data["count"] == 0] = np.nan

        for stat_type in ("std", "max", "min"):
            seasonal_min_max_dict[stat_type]["min"].append(
                np.nanmin(seasonal_data[stat_type])
            )
            seasonal_min_max_dict[stat_type]["max"].append(
                np.nanmax(seasonal_data[stat_type])
            )

    # The min and max values for mean, median or max across all seasons
    overall_min_max = dict(
        (
            stat_type,
            {
                "min": np.nanmin(seasonal_min_max_dict[stat_type]["min"]),
                "max": np.nanmax(seasonal_min_max_dict[stat_type]["max"]),
            },
        )
        for stat_type in ("std", "max", "min")
    )

    # Determine the limits for the colorbar based on the min and max values present across all seasons
    # cbar_min_max_output = {
    #     "std": {
    #         "min": np.power(10, np.floor(np.log10(overall_min_max["count"]["min"]))),
    #         "max": np.power(10, np.ceil(np.log10(overall_min_max["count"]["max"]))),
    #     },
    #     "max": {
    #         "min": overall_min_max["mean"]["min"]
    #         - (overall_min_max["mean"]["min"] % vort_cbar_step),
    #         "max": overall_min_max["mean"]["max"]
    #         - (overall_min_max["mean"]["max"] % vort_cbar_step)
    #         + vort_cbar_step,
    #     },
    #     "min": {
    #         "min": overall_min_max["median"]["min"]
    #         - (overall_min_max["median"]["min"] % vort_cbar_step),
    #         "max": overall_min_max["median"]["max"]
    #         - (overall_min_max["median"]["max"] % vort_cbar_step)
    #         + vort_cbar_step,
    #     },
    # }
    #
    # # Make the colorbar symmetric about 0 for means and medians. Assumes max val > 0
    # for stat_type in ("mean", "median"):
    #     min_for_stat = cbar_min_max_output[stat_type]["min"]
    #     max_for_stat = cbar_min_max_output[stat_type]["max"]
    #
    #     biggest_abs_val = np.max([np.abs(min_for_stat), np.abs(max_for_stat)])
    #
    #     cbar_min_max_output[stat_type]["min"] = -biggest_abs_val
    #     cbar_min_max_output[stat_type]["max"] = biggest_abs_val

    return overall_min_max


def __plot_subplot(
    fig: figure,
    axs: NDArray[NDArray[axes]],
    phi_edges: NDArray,
    theta_edges: NDArray,
    stat_data_for_season: NDArray,
    cbar_min_max: Dict[str, Dict[str, float]],
    column_idx: int,
    plot_type: str,
    row_idx: int,
    max_theta: float,
    fontsize: float,
) -> None:
    """
    Plots a subplot

    Parameters
    ----------
    fig: figure,
        fig object from matplotlib
    axs: NDArray[NDArray[axes]]
        all axes objects created
    phi_edges: NDArray
        Boundary values of the bins of phi
    theta_edges: NDArray
        Boundary values of the bins of theta
    stat_data_for_season: NDArray
        The data to be plotted (e.g. mean for spring)
    cbar_min_max: Dict[str, Dict[str, float]]
        This sets the minimum and maximum values of the colorbar
    column_idx: int
        Which column we are in
    plot_type: str
        Is it mean, median, or number of data points?
    row_idx: int
        Which row we are in
    max_theta: float
        Max value of theta
    fontsize: float
        Fontsize to be used (in general)

    Returns
    -------

    """

    # Initialisation
    colour_map_dict = {0: "jet", 1: "jet", 2: "jet"}

    # Normalise data for the colorbar if needed
    norm = colors.Normalize(
        vmin=cbar_min_max[plot_type]["min"], vmax=cbar_min_max[plot_type]["max"]
    )
    #########################
    # Actual plotting
    plot = axs[row_idx][column_idx].pcolormesh(
        *np.meshgrid(phi_edges, theta_edges),
        stat_data_for_season.T,
        cmap=colormaps[colour_map_dict[column_idx]],
        norm=norm,
    )
    plot_utils._common_formatting(axs[row_idx][column_idx], fontsize, max_theta)
    _ax_formatting(fig, axs, plot, row_idx, column_idx, fontsize)

    return None


def _plot_1_season(
    fig: figure,
    axs: NDArray[NDArray[axes]],
    phi_edges: NDArray,
    theta_edges: NDArray,
    data_1_season: NDArray[VortMeasurement],
    count_cutoff: int,
    cbar_min_max: Dict[str, Dict[str, float]],
    row_idx: int,
    max_theta: float,
    fontsize: float,
    coord: str,
) -> None:

    # Extracts data
    phi_coords = plot_utils.mlt_to_phi(
        np.array(
            [vort_data_season.MLT for vort_data_season in data_1_season], dtype=float
        )
    )
    theta_coords = plot_utils.lat_to_theta(
        np.array(
            [
                getattr(vort_data_season, f"{coord}_lat_c")
                for vort_data_season in data_1_season
            ],
            dtype=float,
        )
    )
    season_vort = np.array(
        [vort_data_season.vorticity_mHz for vort_data_season in data_1_season],
        dtype=float,
    )
    abs_season_vort = np.abs(
        [vort_data_season.vorticity_mHz for vort_data_season in data_1_season],
        dtype=float,
    )
    ####################
    # Does the calculations
    stat_data_season = dict(
        (
            stat,
            binned_statistic_2d(
                phi_coords,
                theta_coords,
                season_vort,
                statistic=stat,
                bins=(phi_edges, theta_edges),
            ).statistic,
        )
        for stat in ("std", "count")
    )
    for stat_type in ("max", "min"):
        stat_data_season[stat_type] = binned_statistic_2d(
            phi_coords,
            theta_coords,
            abs_season_vort,
            statistic=stat_type,
            bins=(phi_edges, theta_edges),
        ).statistic

    assert not np.isnan(
        stat_data_season["count"]
    ).any()  # Assert there aren't any invalid values in the counts

    # Do not plot bins with fewer counts than a threshold
    for stat_type in ("std", "max", "min"):
        stat_data_season[stat_type][stat_data_season["count"] < count_cutoff] = np.nan

    # Do not plot bins that have 0 counts
    stat_data_season["count"][stat_data_season["count"] == 0] = np.nan

    ####################
    # Plots the data
    for column_idx, stat_type in enumerate(["std", "max", "min"]):
        __plot_subplot(
            fig,
            axs,
            phi_edges,
            theta_edges,
            stat_data_season[stat_type],
            cbar_min_max,
            column_idx,
            stat_type,
            row_idx,
            max_theta,
            fontsize,
        )

    return None


@ray.remote
def plot(
    plot_params: PlotParams,
    vort_by_season: Dict[str, NDArray[VortMeasurement]],
    coord: str = "aacgm",
    count_cutoff: int = 120,
    fontsize=40,
):
    """
    Plots the mean, median, and number of data points for vorticity data separated by their seasons

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
    vort_by_season: Dict[str, NDArray[VortMeasurement]]
        Dictionary containing array of VortMeasurement objects, separated by seasons
    coord: str
        Coordinate system to be used for the latitude. Only accepts AACGM or GEO
    count_cutoff: int
        Bins with fewer data points than this cutoff will not be plotted
    fontsize: float
        Size of most words which appear on the plot

    Returns
    -------

    """

    # Initialisation

    logger = logging.getLogger(__name__ + ".plot")
    log_utils.set_logger(logger)

    # Checking
    if coord not in ("aacgm", "geo"):
        raise ValueError('Coord must be either "aacgm" or "geo"')
    else:
        pass

    # Creates bin edges

    # Bin sizes
    d_phi_rad = plot_utils.mlt_to_phi(plot_params.mlt_bin_size_hr)
    d_theta_deg = deepcopy(plot_params.lat_bin_size_degree)

    # All edges of the bins for PHI
    phi_edges = plot_utils.create_bin_edges((0, 2 * np.pi), d_phi_rad)

    # All edges of the bins for THETA
    all_vort = np.concatenate(
        [vort_by_season[season] for season in ("spring", "summer", "autumn", "winter")]
    )

    min_lat = np.min(
        np.array([getattr(vort_data, f"{coord}_lat_c") for vort_data in all_vort])
    )
    min_lat_edge = (
        min_lat
        - (min_lat % plot_params.lat_bin_size_degree)
        + plot_params.lat_bin_size_degree
    )
    max_theta = 90 - min_lat_edge

    theta_edges = plot_utils.create_bin_edges((0, max_theta), d_theta_deg)

    # Sets up the values needed for the common color-bars
    cbar_min_max = _find_min_max_for_colorbar(
        phi_edges,
        theta_edges,
        vort_by_season,
        coord,
        count_cutoff,
    )
    ####################
    # Setting up the plotting routine
    fig, axs = plt.subplots(
        5,
        3,
        figsize=(36, 60),
        subplot_kw={"projection": "polar"},
        gridspec_kw={"height_ratios": [0.05, 1, 1, 1, 1]},
    )

    for row_idx, vort_1_season in enumerate(
        [vort_by_season[season] for season in ("spring", "summer", "autumn", "winter")]
    ):
        row_idx += 1  # 1st row is for colorbar
        _plot_1_season(
            fig,
            axs,
            phi_edges,
            theta_edges,
            vort_1_season,
            count_cutoff,
            cbar_min_max,
            row_idx,
            max_theta,
            fontsize,
            coord,
        )

    # Does more formatting
    _fig_formatting(fig, all_vort, fontsize)
    fig.tight_layout()

    # Saving the file
    plt.savefig(
        common_params.plot_dir / "sd_abs_max_min_(by_season).png", bbox_inches="tight"
    )

    return None
