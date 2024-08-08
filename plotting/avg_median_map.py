import logging
from pathlib import Path
from typing import Tuple, Union
from copy import deepcopy, copy

import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib.collections import QuadMesh
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d

from common_utils import log_utils
from classes.map_params_cls import MapParams

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


def plot(
        map_params: MapParams,
        vort_array: NDArray,
        coord: str = 'aacgm',
        count_cutoff: int = 100
):
    """


    In order to make the data plottable on a polar projection,
    - MLT is converted to phi (radians), which is the 'azimuthal angle', aka the angle
    of rotation of the radial line around the polar axis
    - latitude (degrees) is converted to theta (degrees), which is the 'polar angle', aka
    the angle between the radial line and the polar axis
    - (The radial line is the straight line connecting the origin and the data point)
    - (Theta and phi are defined according to the convention used mainly by physicists)

    Parameters
    ----------
    map_params
    vort_array
    coord

    Returns
    -------

    """

    def _common_formatting(ax_to_set: axes) -> None:
        """
        Formatting which applies to all sub-plots
        """

        ax = copy(ax_to_set)

        ## Rotate plot to match conventional MLT representation
        ax.set_theta_zero_location('S')

        ## Set lim in radial direction
        max_theta_for_plot = max_theta - (max_theta % 5) + 5
        ax.set_rlim(0, max_theta_for_plot)

        ## Ticks

        # Ticks in MLT
        ax.set_xticklabels(['00\nMLT', '', '06\nMLT', '', '12\nMLT', '', '18\nMLT', ''])

        # Ticks in latitude
        r_ticks = np.arange(0, max_theta_for_plot + 5, 5)
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([
            int(lat) for lat in theta_to_lat(r_ticks)
        ])

        ## Set grid lines
        ax.grid(
            visible=True,
            which='both',
            axis='both',
            linestyle=':',
            alpha=1,
            color='black'
        )

        return None

    def _vort_formatting(
            ax_to_set: axes,
            plot: QuadMesh,
            plot_type: str
    ) -> None:
        """
        Formatting which applies to sub-plots which shows vorticities

        Returns
        -------

        """
        ax = copy(ax_to_set)

        fig.colorbar(
            plot,
            ax=ax,
            location='top',
            orientation='horizontal',
            fraction=0.15,
            pad=0.15,
            ticks=np.arange(-3.0, 3.5, 0.5),
            label='Vorticity (mHz)'
        )

        return None

    def _plot_vort(
            ax_to_plot: axes,
            data: NDArray,
            plot_type: str,
            vort_cbar_step: float = 0.5
    ) -> None:

        assert plot_type in ('mean', 'median')

        ax = copy(ax_to_plot)

        all_mean_and_median = np.concatenate((means.flatten(), medians.flatten()))
        data_min = np.nanmin(all_mean_and_median)
        data_max = np.nanmax(all_mean_and_median)

        plot_cbar_min = data_min - (data_min % vort_cbar_step)
        plot_cbar_max = data_max - (data_max % vort_cbar_step) + vort_cbar_step

        vort_plot = ax.pcolormesh(
            *np.meshgrid(phi_edges, theta_edges),
            data.T,
            cmap='plasma',
            vmin=plot_cbar_min,
            vmax=plot_cbar_max
        )
        _common_formatting(ax)
        _vort_formatting(ax, vort_plot, plot_type)

        return None

    if coord not in ('aacgm', 'geo'):
        raise ValueError('Coord must be either "aacgm" or "geo"')
    else:
        pass

    # Extracts data
    phi_coords = mlt_to_phi(
        np.array([vort_data.MLT for vort_data in vort_array])
    )
    theta_coords = lat_to_theta(
        np.array([getattr(vort_data, f'{coord}_lat_c') for vort_data in vort_array])
    )
    vort_data = np.array(
        [vort_measurement.vorticity_mHz for vort_measurement in vort_array]
    )

    ####################
    # Creates bin edges

    # Bin sizes
    d_phi_rad = mlt_to_phi(map_params.mlt_bin_size_hr)
    d_theta_deg = deepcopy(map_params.lat_bin_size_degree)

    # All edges of the bins for PHI
    phi_edges = create_bin_edges((0, 2 * np.pi), d_phi_rad)

    # All edges of the bins for THETA
    min_lat = np.min(
        np.array([getattr(vort_data, f'{coord}_lat_c') for vort_data in vort_array])
    )
    min_lat_edge = (min_lat
                    - (min_lat % map_params.lat_bin_size_degree)
                    + map_params.lat_bin_size_degree)
    max_theta = 90 - min_lat_edge

    theta_edges = create_bin_edges((0, max_theta), d_theta_deg)
    ####################
    # Does the calculations

    means, medians, counts = [
        binned_statistic_2d(
            phi_coords,
            theta_coords,
            vort_data,
            statistic=stat,
            bins=(phi_edges, theta_edges)
        ).statistic
        for stat in ('mean', 'median', 'count')
    ]

    # Do not plot bins with fewer counts than a threshold (100 by default)
    means[counts < count_cutoff] = np.nan
    medians[counts < count_cutoff] = np.nan

    # Do not plot bins that have 0 counts
    counts[counts == 0] = np.nan
    ####################
    # Setting up the plotting routine

    fig, axs = plt.subplots(1, 3, figsize=(36, 12), subplot_kw={'projection': 'polar'})
    mean_ax, median_ax, count_ax = axs

    # Plots the data
    _plot_vort(mean_ax, means, plot_type='mean')
    _plot_vort(median_ax, medians, plot_type='median')

    plt.show()

    return None
