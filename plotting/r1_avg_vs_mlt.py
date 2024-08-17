"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-14
Last Modified : 2024-08-17

Summary       : Analyses the R1 current by plotting the average vorticity (across a specified range of AACGM latitudes)
across all MLT
"""

import logging
from typing import Dict
from copy import deepcopy

import ray
from matplotlib import pyplot as plt
from matplotlib import figure, axes
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d

from common_utils import log_utils, plot_utils
from classes.plot_params_cls import PlotParams
from classes.data_class import VortMeasurement
from params import common_params

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def get_avg_vs_mlt(
    season_vorts: NDArray[VortMeasurement],
    plot_params: PlotParams,
) -> NDArray:
    """
    For a given season, this calculates the average vorticity for each MLT

    Parameters
    ----------
    season_vorts: NDArray[VortMeasurement]
        Array of VortMeasurement objects
    plot_params: PlotParams
        Used here to decide which data falls within the latitude range set in plot_params, and for the count cutoff.
        Bins with fewer data points than this cutoff will not be plotted

    Returns
    -------
    NDArray
        Average vorticity for each MLT
    """

    # Initialisation

    # Only retain data within latitude range
    low_lat_lim, high_lat_lim = plot_params.r1_vort_aacgm_lat_lim

    season_vorts = np.array(
        [
            vort_data
            for vort_data in season_vorts
            if (low_lat_lim <= vort_data.aacgm_lat_c <= high_lat_lim)
        ]
    )

    # The data
    mlt_coords = np.array([vort_data.MLT for vort_data in season_vorts], dtype=float)
    lat_coords = np.array(
        [vort_data.aacgm_lat_c for vort_data in season_vorts], dtype=float
    )
    vort_vals = np.array(
        [vort_data.vorticity_mHz for vort_data in season_vorts], dtype=float
    )

    # Bin sizes
    d_mlt = deepcopy(plot_params.mlt_bin_size_hr)
    d_lat = deepcopy(plot_params.lat_bin_size_degree)

    # All edges of the bins for
    mlt_edges = plot_utils.create_bin_edges((0, 24), d_mlt)
    lat_edges = plot_utils.create_bin_edges((65, 90), d_lat)

    #########################
    # The calculations
    two_d_stats = dict(
        (
            stat_type,
            binned_statistic_2d(
                mlt_coords,
                lat_coords,
                vort_vals,
                statistic=stat_type,
                bins=(mlt_edges, lat_edges),
            ).statistic,
        )
        for stat_type in ("sum", "count")
    )

    # Throw away the bins with 0 counts
    for stat_type in ("sum", "count"):
        two_d_stats[stat_type][
            two_d_stats["count"] <= plot_params.count_cutoff
        ] = np.nan

    vort_sum = np.nansum(
        two_d_stats["sum"], axis=1
    )  # Sum of all vorticity measurement for a given MLT
    count_sum = np.nansum(
        two_d_stats["count"], axis=1
    )  # Total number of data points for a given MLT

    avg_vort = vort_sum / count_sum

    return avg_vort


def formatting(
    vort_by_season: Dict[str, NDArray[VortMeasurement]],
    fig: figure,
    ax: axes,
    plot_params: PlotParams,
    fontsize: float,
    linewidth: float,
) -> None:
    """
    Does all the formatting needed

    Parameters
    ----------
    vort_by_season: Dict[str, NDArray[VortMeasurement]]
        Used
    fig: figure
        fig object to be formatted
    ax: axes
        axes object to be formatted
    plot_params: PlotParams
        Used here to decide which data falls within the latitude range set in plot_params
    fontsize: float
        Size of the fonts (in general)
    linewidth: float
        Width of the lines (in general)

    Returns
    -------

    """

    # Find the range of years covered
    all_vort = np.concatenate(list(vort_by_season.values()))
    all_t = np.array([vort.utc_time for vort in all_vort])
    min_year = np.nanmin(all_t).year
    max_year = np.nanmax(all_t).year

    # Find the range of latitudes
    lat_min, lat_max = plot_params.r1_vort_aacgm_lat_lim

    # Set title
    fig.suptitle(
        f"""
        Average Vorticity vs MLT
        Within {lat_min}\N{DEGREE SIGN} - {lat_max}\N{DEGREE SIGN} AACGM Latitude
        (R1 Vorticity)
        in the Northern Hemisphere
        During {min_year} - {max_year}
        """,
        fontsize=fontsize,
        horizontalalignment="center",
        verticalalignment="center",
        x=0.5,
        y=0.9,
    )

    # Set x and y axes labels

    ax.set_xlabel(
        "Magnetic Local Time (MLT)",
        fontsize=fontsize,
        labelpad=fontsize * 1.5,
        horizontalalignment="center",
        verticalalignment="center",
    )

    ax.set_ylabel(
        "Average Vorticity (mHz)",
        fontsize=fontsize,
        labelpad=fontsize * 1.5,
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Ticks
    tick_length = {"major": linewidth * 8, "minor": linewidth * 3}
    show_num = {"major": True, "minor": False}

    # Set the ticks
    ax.set_xticks(np.arange(0, 25, 6), minor=False)
    ax.set_xticks(np.arange(0, 25, 1), minor=True)

    ax.set_yticks(np.arange(-3.5, 3.5, 0.5), minor=False)

    ax.set_yticks(np.arange(-3.5, 3.5, 0.1), minor=True)

    for tick_type in ("major", "minor"):
        ax.xaxis.set_tick_params(
            labelsize=fontsize,
            length=tick_length[tick_type],
            width=linewidth / 2,
            which=tick_type,
            direction="in",
            bottom=True,
            top=True,
            labelbottom=show_num[tick_type],
            labeltop=False,
            pad=fontsize,
        )

        ax.yaxis.set_tick_params(
            labelsize=fontsize,
            length=tick_length[tick_type],
            width=linewidth / 2,
            which=tick_type,
            direction="in",
            left=True,
            right=True,
            labelleft=show_num[tick_type],
            labelright=False,
            pad=fontsize,
        )

    # Set limit of x and y axes
    ax.set_xlim(0, 24)
    ax.set_ylim(-1.5, 1.5)

    # Set linewidth of box of each plot
    for box_line in ["top", "right", "bottom", "left"]:
        ax.spines[box_line].set_linewidth(linewidth)

    # Legend
    fig.legend(fontsize=fontsize, bbox_to_anchor=(1.2, 0.55))

    return None


@ray.remote
def plot(
    plot_params: PlotParams,
    vort_by_season: Dict[str, NDArray[VortMeasurement]],
    fontsize=40,
    linewidth=5,
) -> None:
    """
    Plot the R1 vorticities by plotting the average vorticity (across a specified range of AACGM latitudes) vs MLT

    Parameters
    ----------
    plot_params: PlotParams
        Used here to decide which data falls within the latitude range set in plot_params
    vort_by_season: Dict[str, NDArray[VortMeasurement]]
        All vorticity data separated by seasons
    fontsize: float
        Size of the fonts (in general)
    linewidth: float
        Width of the lines (in general)

    Returns
    -------

    """

    logger = logging.getLogger(__name__ + ".plot")
    log_utils.set_logger(logger)

    season_colours = {
        "spring": "#72B01D",
        "summer": "red",
        "autumn": "#F7B32B",
        "winter": "black",
    }

    bin_centres_mlt = np.arange(0.5, 24, 1)

    fig, ax = plt.subplots(figsize=(30, 17))

    for season, season_vorts in vort_by_season.items():

        avg_per_mlt = get_avg_vs_mlt(season_vorts, plot_params)

        ax.plot(
            bin_centres_mlt,
            avg_per_mlt,
            linewidth=linewidth,
            color=season_colours[season],
            marker="s",
            markersize=linewidth * 5,
            markeredgewidth=linewidth,
            fillstyle="none",
            label=season.capitalize(),
        )

    # Plot horizontal line at vorticity = 0 for reference
    ax.axhline(y=0, linewidth=linewidth / 2, linestyle="--", color="black")

    formatting(vort_by_season, fig, ax, plot_params, fontsize, linewidth)

    fig.tight_layout()

    # Saving the file
    plt.savefig(common_params.plot_dir / "r1_avg_vorticity.png", bbox_inches="tight")

    return None
