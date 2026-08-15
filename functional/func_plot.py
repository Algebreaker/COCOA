#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

# ================= USER SETTINGS =================
input_dir = '/data/pt_02825/MPCDF/univariate7mm/'
output_dir = '/data/pt_02825/MPCDF/univariate7mm/plots'
min_cluster_size = 14

# key statistical threshold (IMPORTANT)
z_threshold = 2.3
# ================================================

os.makedirs(output_dir, exist_ok=True)

def save_plot(display, filename):
    path = os.path.join(output_dir, filename)
    display.savefig(path, dpi=300)
    display.close()
    print(f"Saved: {path}")


all_files = os.listdir(input_dir)

stat_files = [
    f for f in all_files
    if f.endswith('.nii.gz')
    and '-clusters' not in f
    and 'brain_mask' not in f
]

cluster_files = [f for f in all_files if f.endswith('-clusters.nii.gz')]

print(f"Found {len(stat_files)} stat files")

for stat_file in stat_files:

    base_name = stat_file.replace('.nii.gz', '')

    cluster_candidates = [c for c in cluster_files if c.startswith(base_name)]

    if not cluster_candidates:
        print(f"No cluster map for {stat_file}")
        continue

    cluster_file = cluster_candidates[0]

    print(f"\nProcessing {stat_file}")

    try:

        stat_img = nib.load(os.path.join(input_dir, stat_file))
        cluster_img = nib.load(os.path.join(input_dir, cluster_file))

        stat_data = stat_img.get_fdata()
        cluster_data = cluster_img.get_fdata()


        filtered_cluster_data = np.zeros_like(cluster_data)

        for label in np.unique(cluster_data):
            if label == 0:
                continue

            mask = cluster_data == label

            if np.sum(mask) >= min_cluster_size:
                filtered_cluster_data[mask] = 1

        if np.count_nonzero(filtered_cluster_data) == 0:
            print("No clusters survive size threshold")
            continue


        masked_stat_data = np.where(
            np.abs(stat_data) >= z_threshold,
            stat_data,
            0
        )

        masked_stat_data = masked_stat_data * filtered_cluster_data

        if np.count_nonzero(masked_stat_data) == 0:
            print("No suprathreshold voxels after masking")
            continue

        masked_img = nib.Nifti1Image(masked_stat_data, stat_img.affine)


        pos_data = np.where(masked_stat_data > 0, masked_stat_data, 0)
        neg_data = np.where(masked_stat_data < 0, masked_stat_data, 0)

        pos_img = nib.Nifti1Image(pos_data, stat_img.affine)

        match = re.search(r'desc-(.*?)(?:_z\d+)?$', stat_file)
        contrast = match.group(1) if match else base_name

        # PLOTS

        # positive effect
        if np.any(pos_data):
            vmax_val = np.max(pos_data)
            disp = plotting.plot_stat_map(
                pos_img,
                cmap='YlOrRd',           # Yellow at 2.3 -> Red at max
                threshold=z_threshold,
                vmin=z_threshold,        # Sets lower map boundary at 2.3
                vmax=vmax_val,
                symmetric_cbar=False,
                display_mode='ortho',
                title=None               # Removes title from top-left corner
            )
            # Adjust colorbar axis bounds to drop sub-threshold values
            if hasattr(disp, '_cbar') and disp._cbar is not None:
                disp._cbar.mappable.set_clim(vmin=z_threshold, vmax=vmax_val)
                disp._cbar.set_label('Z-score', fontsize=10)

            save_plot(disp, f'{contrast}_pos_ortho.png')

        # negative effects
        if np.any(neg_data):
            abs_neg_data = np.abs(neg_data)
            abs_neg_img = nib.Nifti1Image(abs_neg_data, stat_img.affine)
            vmax_val = np.max(abs_neg_data)

            disp = plotting.plot_stat_map(
                abs_neg_img,
                cmap='YlOrRd',           # Yellow at 2.3 -> Red at max
                threshold=z_threshold,
                vmin=z_threshold,        # Sets lower map boundary at 2.3
                vmax=vmax_val,
                symmetric_cbar=False,
                display_mode='ortho',
                title=None               # Removes title from top-left corner
            )
            if hasattr(disp, '_cbar') and disp._cbar is not None:
                disp._cbar.mappable.set_clim(vmin=z_threshold, vmax=vmax_val)
                disp._cbar.set_label('|Z-score|', fontsize=10)

            save_plot(disp, f'{contrast}_neg_ortho.png')

        # glass brain all
        disp = plotting.plot_glass_brain(
            masked_img,
            colorbar=True,
            threshold=z_threshold,
            symmetric_cbar=False,
            title=None                   # Removes title from top-left corner
        )
        if hasattr(disp, '_cbar') and disp._cbar is not None:
            disp._cbar.set_label('Z-score', fontsize=10)

        save_plot(disp, f'{contrast}_glassbrain.png')

        # ---- MOSAIC VIEW ----
        vmax_val = np.max(masked_stat_data)
        disp = plotting.plot_stat_map(
            masked_img,
            display_mode='mosaic',
            threshold=z_threshold,
            vmin=z_threshold,            # Sets lower map boundary at 2.3
            vmax=vmax_val,
            symmetric_cbar=False,
            cmap='YlOrRd',               # Yellow at 2.3 -> Red at max
            title=None                   # Removes title from top-left corner
        )
        if hasattr(disp, '_cbar') and disp._cbar is not None:
            disp._cbar.mappable.set_clim(vmin=z_threshold, vmax=vmax_val)
            disp._cbar.set_label('Z-score', fontsize=10)

        save_plot(disp, f'{contrast}_mosaic.png')

    except Exception as e:
        print(f"Error: {stat_file} → {e}")

print("\nDone.")
