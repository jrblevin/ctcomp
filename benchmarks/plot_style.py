"""
Common plotting style configuration for black and white printing.
"""

import matplotlib.pyplot as plt
import numpy as np


# Grayscale color palette suitable for black and white printing
COLORS = {
    'primary': '0.2',      # Dark gray (almost black)
    'secondary': '0.5',    # Medium gray
    'tertiary': '0.7',     # Light gray
    'quaternary': '0.85',  # Very light gray
    'black': '0.0',        # Pure black
    'white': '1.0',        # Pure white
}

# Extended grayscale palette for multiple series
GRAYSCALE_PALETTE = ['0.2', '0.4', '0.6', '0.8', '0.3', '0.5', '0.7', '0.9']

# Hatching patterns for additional distinction in B&W
HATCHING_PATTERNS = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...']

# Marker styles for line plots
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# Line styles for distinction
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


def setup_plot_style():
    """Set up matplotlib for academic paper styling."""
    plt.style.use('default')
    plt.rcParams.update({
        # Font settings
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,

        # Color settings for B&W compatibility
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',

        # Line and marker settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 1.2,

        # Grid settings
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'grid.color': '0.5',

        # Figure settings
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',

        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
    })


def get_color_scheme(n_series):
    """
    Get a color scheme for n series that works in grayscale.

    Parameters
    ----------
    n_series : int
        Number of data series to plot

    Returns
    -------
    list
        List of grayscale color values
    """
    if n_series <= len(GRAYSCALE_PALETTE):
        return GRAYSCALE_PALETTE[:n_series]
    else:
        # Generate evenly spaced grayscale values
        return [str(i) for i in np.linspace(0.2, 0.8, n_series)]


def get_bar_styles(n_series):
    """
    Get bar styles including colors and hatching patterns.

    Parameters
    ----------
    n_series : int
        Number of data series

    Returns
    -------
    tuple
        (colors, hatches) where each is a list of length n_series
    """
    colors = get_color_scheme(n_series)
    hatches = HATCHING_PATTERNS[:n_series] if n_series <= len(HATCHING_PATTERNS) else HATCHING_PATTERNS * (n_series // len(HATCHING_PATTERNS) + 1)
    return colors[:n_series], hatches[:n_series]


def get_line_styles(n_series):
    """
    Get line styles including colors, markers, and line styles.

    Parameters
    ----------
    n_series : int
        Number of data series

    Returns
    -------
    tuple
        (colors, markers, linestyles) where each is a list of length n_series
    """
    colors = get_color_scheme(n_series)
    markers = MARKERS[:n_series] if n_series <= len(MARKERS) else MARKERS * (n_series // len(MARKERS) + 1)
    linestyles = LINESTYLES[:n_series] if n_series <= len(LINESTYLES) else LINESTYLES * (n_series // len(LINESTYLES) + 1)
    return colors[:n_series], markers[:n_series], linestyles[:n_series]


def add_value_labels(ax, bars, format_str='{:.1f}', offset_ratio=0.02):
    """
    Add value labels on top of bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add labels to
    bars : matplotlib.container.BarContainer
        The bars to label
    format_str : str
        Format string for the labels
    offset_ratio : float
        Offset as ratio of y-axis range
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = y_range * offset_ratio

    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only label positive values
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    format_str.format(height),
                    ha='center', va='bottom', fontsize=9, color='black')


def save_figure(fig, base_path, formats=['png', 'pdf'], dpi=300):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    base_path : str or Path
        Base path without extension
    formats : list
        List of file formats to save
    dpi : int
        Resolution for raster formats
    """
    for fmt in formats:
        fig.savefig(f"{base_path}.{fmt}", dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
