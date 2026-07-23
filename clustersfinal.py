#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
from nilearn.image import load_img
from scipy import ndimage
from nibabel.affines import apply_affine


# ==========================================================
# INPUT FILES
# ==========================================================

# Already cluster-corrected map
cluster_map_path = (
    "/data/pt_02825/MPCDF/univariate7mm/"
    "task-Literacy_space-MNI152NLin6Asym_"
    "desc-congruentMinusIncongruent_z1-clusters.nii.gz"
)

# Original Z map (for peak statistics)
z_map_path = (
    "/data/pt_02825/MPCDF/univariate7mm/"
    "task-Literacy_space-MNI152NLin6Asym_"
    "desc-congruentMinusIncongruent_z1.nii.gz"
)


# ==========================================================
# LOAD IMAGES
# ==========================================================

cluster_img = load_img(cluster_map_path)
cluster_data = cluster_img.get_fdata()

z_img = load_img(z_map_path)
z_data = z_img.get_fdata()


affine = cluster_img.affine


# ==========================================================
# FUNCTION TO EXTRACT CLUSTERS
# ==========================================================

def extract_clusters(cluster_data, z_data, sign_name):

    print("\n" + "=" * 70)
    print(f"{sign_name.upper()} CLUSTERS")
    print("=" * 70)


    # Reconstruct cluster labels
    binary_mask = cluster_data != 0

    labeled_array, n_clusters = ndimage.label(
        binary_mask
    )


    if n_clusters == 0:
        print("No clusters found.")
        return


    results = []


    for label in range(1, n_clusters + 1):

        cluster_mask = labeled_array == label


        n_voxels = np.sum(cluster_mask)

        voxel_volume = np.prod(
            cluster_img.header.get_zooms()
        )

        cluster_volume = n_voxels * voxel_volume


        # Peak Z
        cluster_z = np.where(
            cluster_mask,
            z_data,
            0
        )


        if sign_name == "negative":

            peak_index = np.unravel_index(
                np.argmin(cluster_z),
                cluster_z.shape
            )

            peak_z = cluster_z[peak_index]


        else:

            peak_index = np.unravel_index(
                np.argmax(cluster_z),
                cluster_z.shape
            )

            peak_z = cluster_z[peak_index]


        peak_mni = apply_affine(
            affine,
            peak_index
        )


        results.append(
            {
                "Cluster": label,
                "Voxels": int(n_voxels),
                "Volume_mm3": int(cluster_volume),
                "Peak_Z": float(peak_z),
                "MNI_x": round(peak_mni[0], 2),
                "MNI_y": round(peak_mni[1], 2),
                "MNI_z": round(peak_mni[2], 2),
            }
        )


    results = sorted(
        results,
        key=lambda x: x["Voxels"],
        reverse=True
    )


    for i, r in enumerate(results, 1):

        print(
            f"\nCluster {i}"
            f"\n  Size: {r['Voxels']} voxels "
            f"({r['Volume_mm3']} mm³)"
            f"\n  Peak Z: {r['Peak_Z']:.3f}"
            f"\n  MNI: "
            f"({r['MNI_x']}, {r['MNI_y']}, {r['MNI_z']})"
        )

    return results



# ==========================================================
# SPLIT POSITIVE AND NEGATIVE CLUSTERS
# ==========================================================

positive_clusters = cluster_data > 0
negative_clusters = cluster_data < 0


# Extract
positive_results = extract_clusters(
    positive_clusters,
    z_data,
    "positive"
)


negative_results = extract_clusters(
    negative_clusters,
    z_data,
    "negative"
)


print("\nDone.")
