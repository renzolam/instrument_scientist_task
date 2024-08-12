"""
Author        : Pak Yin (Renzo) Lam
                British Antarctic Survey
                paklam@bas.ac.uk

Date Created  : 2024-08-10
Last Modified : 2024-08-10

Summary       : Plots the plots required

List of functions:
- create_bin_edges
- mlt_to_phi
- lat_to_theta
- theta_to_lat
- plot_mean_median_counts
"""

import logging
from typing import Tuple, Union, List, Dict
from copy import deepcopy, copy

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import axes
from matplotlib.collections import QuadMesh
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d

from common_utils import log_utils
from classes.map_params_cls import MapParams
from classes.main_runparams_cls import MainRunParams
from classes.data_class import VortMeasurement

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


def plot_mean_median_counts(
        main_params: MainRunParams,
        map_params: MapParams,
        spring_data: NDArray,
        summer_data: NDArray,
        autumn_data: NDArray,
        winter_data: NDArray,
        coord: str = 'aacgm',
        count_cutoff: int = 100,
        fontsize=40
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

    main_params: MainRunParams
        Used here to get location of where the plot should be saved to
    map_params: MapParams
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

    def _common_formatting(ax_to_set: axes) -> None:
        """
        Formatting which applies to all sub-plots
        """

        ax = copy(ax_to_set)

        # Rotate plot to match conventional MLT representation
        ax.set_theta_zero_location('S')

        # Set lim in radial direction
        max_theta_for_plot = max_theta - (max_theta % 5) + 5
        ax.set_rlim(0, max_theta_for_plot)

        # Ticks

        # Ticks in MLT
        ax.set_xticklabels(
            ['00\nMLT', '', '06\nMLT', '', '12\nMLT', '', '18\nMLT', ''],
            fontsize=fontsize
        )

        # Ticks in latitude
        r_ticks = np.arange(0, max_theta_for_plot + 5, 5)
        ax.set_yticks(r_ticks)
        ax.set_yticklabels(
            [int(lat) for lat in theta_to_lat(r_ticks)],
            fontsize=fontsize
        )

        # Set grid lines
        ax.grid(
            visible=True,
            which='both',
            axis='both',
            linestyle=':',
            alpha=1,
            color='black'
        )

        return None

    def _ax_formatting(
            ax_to_set: axes,
            plot_to_format: QuadMesh,
            plot_type: str
    ) -> None:
        """
        Formatting which applies to sub-plots which shows vorticities

        Returns
        -------

        """

        # Initialisation
        ax = copy(ax_to_set)

        assert plot_type in ('mean', 'median', 'count')

        label_dict = {
            'mean': 'Mean Vorticity (mHz)',
            'median': 'Median Vorticity (mHz)',
            'count': 'Number of Data Points'
        }

        ticks_dict = {
            'mean': np.arange(-3.0, 3.5, 0.5),
            'median': np.arange(-3.0, 3.5, 0.5),
            'count': np.power(10, range(0, 10))
        }

        ####################
        # Does the formatting

        # For the colorbar
        cbar = fig.colorbar(
            plot_to_format,
            ax=ax,
            orientation='horizontal',
            location='top',
            aspect=15,
            ticks=ticks_dict[plot_type]
        )
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_title(label_dict[plot_type], fontsize=fontsize, pad=fontsize)

        # Label for radial axis
        label_position = ax.get_rlabel_position()
        ax.text(
            np.radians(label_position + 10),
            ax.get_rmax() * 1.1,
            f'{coord.upper()} Latitude',
            ha='left',
            va='top',
            fontsize=fontsize
        )

        return None

    def _fig_formatting() -> None:
        """
        Does formatting that is not specific to any subplot
        Returns
        -------

        """

        fig.suptitle(
            """
            Mean, Median, and Number of Data Points 
            for Vorticity Measurements
            of the Northern Hemisphere
            Made between 2000 and 2005
            """,
            fontsize=fontsize,
            horizontalalignment='center',
            verticalalignment='center',
            position=(0.45, 1.0)
        )

        return None

    def _plot_1_season(
            row_axs: NDArray
    ) -> None:

        def __plot_subplot(
                ax_to_plot: axes,
                data: NDArray,
                plot_type: str,
                vort_cbar_step: float = 0.5
        ) -> None:

            # Initialisation
            assert plot_type in ('mean', 'median', 'count')

            ax = copy(ax_to_plot)

            colour_map_dict = {
                'mean': 'jet',
                'median': 'jet',
                'count': 'jet'
            }

            # Normalise data for the colorbar if needed
            if plot_type in ('mean', 'median'):
                norm = colors.Normalize(vmin=cbar_min_max[plot_type]['min'], vmax=cbar_min_max[plot_type]['max'])
            elif plot_type == 'count':
                norm = colors.LogNorm(vmin=cbar_min_max[plot_type]['min'], vmax=cbar_min_max[plot_type]['max'])
            else:
                raise ValueError(f'Plot_type {plot_type} not recognized')
            #########################
            # Actual plotting
            plot = ax.pcolormesh(
                *np.meshgrid(phi_edges, theta_edges),
                data.T,
                cmap=colour_map_dict[plot_type],
                norm=norm
            )
            _common_formatting(ax)
            _ax_formatting(ax, plot, plot_type)

            return None

        # Extracts data
        phi_coords = mlt_to_phi(
            np.array([vort_data_season.MLT for vort_data_season in season_data], dtype=float)
        )
        theta_coords = lat_to_theta(
            np.array([getattr(vort_data_season, f'{coord}_lat_c') for vort_data_season in season_data], dtype=float)
        )
        season_vort = np.array([vort_data_season.vorticity_mHz for vort_data_season in season_data], dtype=float)
        ####################
        # Does the calculations
        means, medians, counts = [
            binned_statistic_2d(
                phi_coords,
                theta_coords,
                season_vort,
                statistic=stat,
                bins=(phi_edges, theta_edges)
            ).statistic
            for stat in ('mean', 'median', 'count')
        ]

        assert not np.isnan(counts).any()  # Assert there aren't any invalid values in the counts

        # Do not plot bins with fewer counts than a threshold (100 by default)
        means[counts < count_cutoff] = np.nan
        medians[counts < count_cutoff] = np.nan

        # Do not plot bins that have 0 counts
        counts[counts == 0] = np.nan

        ####################
        # Plots the data
        __plot_subplot(row_axs[0], means, plot_type='mean')
        __plot_subplot(row_axs[1], medians, plot_type='median')
        __plot_subplot(row_axs[2], counts, plot_type='count')

        return None

    if coord not in ('aacgm', 'geo'):
        raise ValueError('Coord must be either "aacgm" or "geo"')
    else:
        pass

    def _find_min_max_for_colorbar(
            vort_cbar_step: float = 0.5
    ) -> Dict[str, Dict[str, float]]:

        seasonal_min_max_dict = dict(
            (
                stat_type,
                {'min': [], 'max': []}
            ) for stat_type in ('mean', 'median', 'count')
        )

        for data_by_season in (spring_data, summer_data, autumn_data, winter_data):
            phi_coords = mlt_to_phi(
                np.array([vort_data.MLT for vort_data in data_by_season])
            )
            theta_coords = lat_to_theta(
                np.array([getattr(vort_data, f'{coord}_lat_c') for vort_data in data_by_season])
            )
            vort_season = np.array(
                [vort_measurement.vorticity_mHz for vort_measurement in data_by_season]
            )

            seasonal_data = dict(
                (stat_type,
                 binned_statistic_2d(
                     phi_coords,
                     theta_coords,
                     vort_season,
                     statistic=stat_type,
                     bins=(phi_edges, theta_edges)
                 ).statistic
                 )
                for stat_type in ('mean', 'median', 'count'))

            # Filtering out unwanted data
            seasonal_data['count'][seasonal_data['count'] == 0] = np.nan
            seasonal_data['mean'][seasonal_data['count'] < count_cutoff] = np.nan
            seasonal_data['median'][seasonal_data['count'] < count_cutoff] = np.nan

            for stat_type in ('mean', 'median', 'count'):
                seasonal_min_max_dict[stat_type]['min'].append(np.nanmin(seasonal_data[stat_type]))
                seasonal_min_max_dict[stat_type]['max'].append(np.nanmax(seasonal_data[stat_type]))

        # The min and max values for mean, median or max across all seasons
        overall_min_max = dict(
            (
                stat_type,
                {
                    'min': np.nanmin(seasonal_min_max_dict[stat_type]['min']),
                    'max': np.nanmax(seasonal_min_max_dict[stat_type]['max'])
                }
            )
            for stat_type in ('mean', 'median', 'count')
        )

        # Determine the limits for the colorbar based on the min and max values present across all seasons
        cbar_min_max_output = {
            'count':
                {
                    'min':
                        np.power(
                            10,
                            np.floor(
                                np.log10(
                                    overall_min_max['count']['min']
                                )
                            )
                        ),
                    'max':
                        np.power(
                            10,
                            np.ceil(
                                np.log10(
                                    overall_min_max['count']['max']
                                )
                            )
                        )
                },
            'mean':
                {
                    'min': overall_min_max['mean']['min'] - (overall_min_max['mean']['min'] % vort_cbar_step),
                    'max': overall_min_max['mean']['max'] - (overall_min_max['mean']['max'] % vort_cbar_step) + vort_cbar_step
                },
            'median':
                {
                    'min': overall_min_max['median']['min'] - (overall_min_max['median']['min'] % vort_cbar_step),
                    'max': overall_min_max['median']['max'] - (
                                overall_min_max['median']['max'] % vort_cbar_step) + vort_cbar_step
                }
        }

        # Make the colorbar symmetric about 0 for means and medians. Assumes max val > 0
        for stat_type in ('mean', 'median'):
            min_for_stat = cbar_min_max_output[stat_type]['min']
            max_for_stat = cbar_min_max_output[stat_type]['max']

            biggest_abs_val = np.max([np.abs(min_for_stat), np.abs(max_for_stat)])

            cbar_min_max_output[stat_type]['min'] = - biggest_abs_val
            cbar_min_max_output[stat_type]['max'] = biggest_abs_val

        return cbar_min_max_output

    ####################
    # Creates bin edges

    # Bin sizes
    d_phi_rad = mlt_to_phi(map_params.mlt_bin_size_hr)
    d_theta_deg = deepcopy(map_params.lat_bin_size_degree)

    # All edges of the bins for PHI
    phi_edges = create_bin_edges((0, 2 * np.pi), d_phi_rad)

    # All edges of the bins for THETA
    vort_array = np.concatenate([winter_data, spring_data, summer_data, autumn_data])

    min_lat = np.min(
        np.array([getattr(vort_data, f'{coord}_lat_c') for vort_data in vort_array])
    )
    min_lat_edge = (min_lat
                    - (min_lat % map_params.lat_bin_size_degree)
                    + map_params.lat_bin_size_degree)
    max_theta = 90 - min_lat_edge

    theta_edges = create_bin_edges((0, max_theta), d_theta_deg)

    ####################
    # Sets up the values needed for the common color-bars
    cbar_min_max = _find_min_max_for_colorbar()
    ####################
    # Setting up the plotting routine

    fig, axs = plt.subplots(4, 3, figsize=(36, 70),
                            subplot_kw={'projection': 'polar'})

    for season_idx, season_data in enumerate((spring_data, summer_data, autumn_data, winter_data)):
        _plot_1_season(axs[season_idx])

    # Does more formatting
    _fig_formatting()
    fig.tight_layout()

    # Saving the file
    plot_dir = main_params.output_dir / 'plots'
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    plt.savefig(
        plot_dir / 'avg_median_counts_(by_season).png',
        bbox_inches="tight"
    )

    return None
