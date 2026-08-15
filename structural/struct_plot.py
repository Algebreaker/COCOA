import nibabel as nib
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from nilearn.plotting import plot_surf_stat_map
from nilearn import surface
import os



# variables

hemis = ['left', 'right']
views = ['lateral', 'medial', 'dorsal', 'ventral']

# Thickness omitted because it was empty
types = ['curv', 'area', 'vol']

# Measure-specific p-value thresholds
p_ths = {
    'curv': 6.2529e-04,
    'area': 1.0658e-04,
    'vol':  1.0909e-04
}

file_dir = '/data/pt_02825/MPCDF/freesurfer/lme/Ver_based_res/lin/'
surf_f   = '/data/pt_02825/MPCDF/freesurfer/fsaverage/surf/'


# Load background curvature
fsaverage_meshes = []

for hemi in hemis:

    if hemi == 'left':
        curv_file = surf_f + 'lh.curv'
    else:
        curv_file = surf_f + 'rh.curv'

    fsaverage_meshes.append(
        surface.load_surf_data(curv_file)
    )



# Surface meshes

vert_meshs = []

for hemi in hemis:

    if hemi == 'left':
        vert_meshs.append(surf_f + 'lh.inflated')
    else:
        vert_meshs.append(surf_f + 'rh.inflated')



# Load statistical maps and convert signed p-values to Z-values

def load_z_maps(measure):

    mgh_lr = []
    sign = []

    for hemi in hemis:

        if hemi == 'left':
            filename = file_dir + f'{measure}_p_lh.mgh'
        else:
            filename = file_dir + f'{measure}_p_rh.mgh'

        mgh = nib.load(filename)

        data = mgh.get_fdata().squeeze()

        mgh_lr.append(data)
        sign.append(data < 0)


    # Convert signed p-values to signed Z-values

    z_map_lr = []

    for i in range(len(mgh_lr)):

        abs_p = np.abs(mgh_lr[i])

        z = norm.isf(abs_p / 2)

        z[np.isnan(z)] = 0
        z[np.isinf(z)] = 0

        # Restore sign
        z[sign[i]] *= -1

        z_map_lr.append(z)

    return z_map_lr


# Load all three measures

z_maps = {}

for measure in types:

    print(f'Loading {measure}...')

    z_maps[measure] = load_z_maps(measure)



# Calculate measure-specific Z thresholds

z_ths = {}

for measure in types:

    z_ths[measure] = norm.isf(
        p_ths[measure] / 2
    )

    print(
        f'{measure}: '
        f'p = {p_ths[measure]:.4e}, '
        f'Z threshold = ±{z_ths[measure]:.3f}'
    )


# Determine colour scale for each measure

vmax = {}

for measure in types:

    all_values = np.concatenate([
        np.abs(z_maps[measure][0]),
        np.abs(z_maps[measure][1])
    ])

    all_values = all_values[np.isfinite(all_values)]

    vmax[measure] = np.max(all_values)

    # Ensure the threshold is included
    vmax[measure] = max(
        vmax[measure],
        z_ths[measure]
    )

    print(
        f'{measure}: '
        f'colour scale = ±{vmax[measure]:.3f}'
    )


def make_threshold_cmap(threshold, vmax, n=256):
    """
    Create a custom blue-grey-red colour map.

    Negative significant values:
        dark blue -> light blue

    Non-significant values:
        grey

    Positive significant values:
        yellow -> orange -> dark red

    The grey region corresponds to -threshold to +threshold.
    """


    # Number of colour levels in each section

    n_neg = n // 3
    n_grey = n // 3
    n_pos = n - n_neg - n_grey


    # Negative values:
    # dark blue -> light blue


    blue_cmap = plt.cm.Blues_r

    blue_colours = blue_cmap(
        np.linspace(0.0, 0.75, n_neg)
    )


    grey_colours = np.tile(
        np.array([0.72, 0.72, 0.72, 1.0]),
        (n_grey, 1)
    )


    red_cmap = plt.cm.YlOrRd

    red_colours = red_cmap(
        np.linspace(0.15, 1.0, n_pos)
    )


    colours = np.vstack([
        blue_colours,
        grey_colours,
        red_colours
    ])

    cmap = ListedColormap(colours)

    return cmap



# Figure

fig = plt.figure(
    figsize=(30, 13)
)


outer_grid = fig.add_gridspec(
    nrows=1,
    ncols=3,
    wspace=0.08
)


# Plot each measure

for measure_idx, measure in enumerate(types):

    threshold = z_ths[measure]



    cmap = make_threshold_cmap(
        threshold,
        vmax[measure]
    )


    panel_grid = outer_grid[0, measure_idx].subgridspec(
        nrows=5,
        ncols=2,
        height_ratios=[1, 1, 1, 1, 0.12],
        hspace=0.01,
        wspace=0.01
    )



    panel_x = [0.015, 0.345, 0.675][measure_idx]

    fig.text(
        panel_x,
        0.995,
        f'{chr(65 + measure_idx)})',
        fontsize=20,
        fontweight='bold',
        va='top'
    )



    for row, view in enumerate(views):

        for col, hemi in enumerate(hemis):

            ax = fig.add_subplot(
                panel_grid[row, col],
                projection='3d'
            )



            if hemi == 'left':

                z_map = z_maps[measure][0]
                inflt = fsaverage_meshes[0]
                mesh = vert_meshs[0]

            else:

                z_map = z_maps[measure][1]
                inflt = fsaverage_meshes[1]
                mesh = vert_meshs[1]



            plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=z_map,
                hemi=hemi,
                view=view,

                # Significance threshold
                threshold=threshold,

                bg_map=inflt,
                bg_on_data=False,
                darkness=None,

                # Custom colour map
                cmap=cmap,

                # We add our own colourbar below
                colorbar=False,

                title='',

                figure=fig,
                axes=ax
            )

            ax.set_title('')



    cbar_ax = fig.add_subplot(
        panel_grid[4, :]
    )

    norm_obj = Normalize(
        vmin=-vmax[measure],
        vmax=vmax[measure]
    )

    sm = ScalarMappable(
        norm=norm_obj,
        cmap=cmap
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation='horizontal'
    )



    ticks = np.linspace(
        -vmax[measure],
        vmax[measure],
        5
    )

    cbar.set_ticks(ticks)



    cbar.ax.axvline(
        -threshold,
        color='black',
        linestyle='--',
        linewidth=1.2
    )

    cbar.ax.axvline(
        threshold,
        color='black',
        linestyle='--',
        linewidth=1.2
    )



    cbar.set_label(
        f'Z  (threshold = ±{threshold:.2f})',
        fontsize=11
    )

    cbar.ax.tick_params(
        labelsize=9
    )



outfile = os.path.join(
    file_dir,
    'FDR_corr_z_curv_area_vol_landscape.png'
)

print(f'\nSaving figure to:\n{outfile}')

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.05
)

plt.close(fig)

print('\nDone.')
