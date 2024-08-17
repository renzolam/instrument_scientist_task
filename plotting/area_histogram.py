"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-17
Last Modified : 2024-08-17

Summary       : Plots the distribution of the area of the loops used for measuring the vorticity values
across all MLT
"""

import logging

import ray
from matplotlib import pyplot as plt
from matplotlib import figure, axes
import numpy as np
from numpy.typing import NDArray

from common_utils import log_utils
from classes.plot_params_cls import PlotParams
from classes.data_class import VortMeasurement
from params import common_params

logger = logging.getLogger(__name__)
log_utils.set_logger(logger)


def formatting(
    all_vort: NDArray[VortMeasurement],
    fig: figure,
    ax: axes,
    fontsize: float
) -> None:
    """
    Does all the formatting needed

    Parameters
    ----------
    all_vort: NDArray[VortMeasurement]
        Used
    fig: figure
        fig object to be formatted
    ax: axes
        axes object to be formatted
    fontsize: float
        Size of the fonts (in general)

    Returns
    -------

    """

    all_t = np.array([vort.utc_time for vort in all_vort])
    min_year = np.nanmin(all_t).year
    max_year = np.nanmax(all_t).year

    # Set title
    fig.suptitle(
        f"""
        Histogram for Area of Loops Used
        to Measure Vorticity
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
        "Area (km$^2$)",
        fontsize=fontsize,
        labelpad=fontsize * 1.5,
        horizontalalignment="center",
        verticalalignment="center",
    )

    ax.set_ylabel(
        "Counts",
        fontsize=fontsize,
        labelpad=fontsize * 1.5,
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Ticks
    tick_length = {"major": 40, "minor": 15}
    show_num = {"major": True, "minor": True}

    for tick_type in ("major", "minor"):
        ax.xaxis.set_tick_params(
            labelsize=fontsize,
            length=tick_length[tick_type],
            width=5,
            which=tick_type,
            direction="out",
            bottom=True,
            top=False,
            labelbottom=show_num[tick_type],
            labeltop=False,
            pad=fontsize,
        )

        ax.yaxis.set_tick_params(
            labelsize=fontsize,
            length=tick_length[tick_type],
            width=5,
            which=tick_type,
            direction="in",
            left=True,
            right=True,
            labelleft=show_num[tick_type],
            labelright=False,
            pad=fontsize,
        )

    # Set scale of axes
    ax.set_yscale("log")

    # Set linewidth of box of each plot
    for box_line in ["top", "right", "bottom", "left"]:
        ax.spines[box_line].set_linewidth(5)

    return None


@ray.remote
def plot(
    plot_params: PlotParams,
    all_vort: NDArray[VortMeasurement],
    fontsize=40,
) -> None:
    """
    Plot the distribution of area as a histogram

    Parameters
    ----------
    plot_params: PlotParams
        Used here to decide which data falls within the latitude range set in plot_params
    all_vort: NDArray[VortMeasurement]
        Array of VortMeasurement objects, each of which contain data for a measurement made
    fontsize: float
        Size of the fonts (in general)

    Returns
    -------

    """

    # Initialisation

    logger = logging.getLogger(__name__ + ".plot")
    log_utils.set_logger(logger)

    area_km2_data = np.array([vort.area_km2 for vort in all_vort])

    ######################
    # Plotting

    fig, ax = plt.subplots(figsize=(30, 17))

    ax.hist(
        area_km2_data,
        bins=plot_params.histogram_no_of_bins,
        rwidth=0.9
    )

    formatting(all_vort, fig, ax, fontsize)

    fig.tight_layout()

    # Saving the file
    plt.savefig(common_params.plot_dir / "area_histogram.png", bbox_inches="tight")

    return None
