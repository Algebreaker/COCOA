#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 00:36:41 2025

@author: finnemann
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import re
import matplotlib.pyplot as plt

# -------------------------------
# User configuration
# -------------------------------
TASK = "Literacy"
SPACE = "MNI152NLin6Asym"
UNIVARIATE_DIR = Path("/data/pt_02825/MPCDF/univariate7mm")

MAKE_PLOTS = True
PLOT_DIR = UNIVARIATE_DIR / "psc_plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------------
# Contrast mapping for 9mm naming
# -------------------------------
contrast_map = {
    "congruentMinusAudioMinusVisual": "congruentMinusAudioMinusVisual",
    "congruentMinusAudio": "congruentMinusAudio",
    "congruentMinusIncongruent": "congruentMinusIncongruent",
    "congruentMinusVisual": "congruentMinusVisual",
    "congruent": "congruent",
    "incongruent": "incongruent",
    "unimodalAudios": "unimodalAudios",
    "unimodalImages": "unimodalImages"
}

# -------------------------------
# Load metadata
# -------------------------------
meta_path = UNIVARIATE_DIR / f"metadata_task-Literacy.tsv"
meta_df = pd.read_csv(meta_path, sep="\t", dtype={"session": str})

if "good_ixs" in meta_df.columns:
    meta_df = meta_df[meta_df["good_ixs"] == "Y"]

meta_df["session"] = meta_df["session"].apply(
    lambda x: f"{int(x):02d}"
)

print(
    f"Loaded metadata for "
    f"{meta_df.shape[0]} (subject, session) pairs"
)

# -------------------------------
# Find cluster maps
# -------------------------------
cluster_files = sorted(
    list(
        UNIVARIATE_DIR.glob(
            f"task-{TASK}_space-{SPACE}_desc-*_z1-clusters.nii*"
        )
    )
)

if not cluster_files:
    raise FileNotFoundError(
        f"No cluster maps found in {UNIVARIATE_DIR}"
    )

print(f"Found {len(cluster_files)} cluster maps")

# -------------------------------
# PSC extraction
# -------------------------------
psc_records = []

for cluster_file in cluster_files:

    # Extract cluster contrast name
    fname = cluster_file.name

    m = re.search(
        r"desc-(.+?)_z1-clusters",
        fname
    )

    if not m:
        continue

    contrast_cluster = m.group(1)

    if contrast_cluster not in contrast_map:
        print(
            f" ⚠️ No mapping for cluster contrast "
            f"{contrast_cluster}, skipping"
        )
        continue

    contrast_effect = contrast_map[contrast_cluster]

    # Load cluster map
    cluster_img = nib.load(cluster_file)
    cluster_data = cluster_img.get_fdata()

    cluster_ids = np.unique(
        cluster_data[cluster_data != 0]
    )

    print(
        f"\nProcessing contrast: "
        f"{contrast_cluster} → {contrast_effect}, "
        f"{len(cluster_ids)} clusters found"
    )

    # -------------------------------
    # Loop over subjects/sessions
    # -------------------------------
    for _, row in meta_df.iterrows():

        sub = row["subject"]
        ses = row["session"]
        time = row.get("time", np.nan)

        effect_dir = (
            UNIVARIATE_DIR
            / "derivatives"
            / f"sub-{sub}"
        )

        all_effect_files = list(
            effect_dir.glob(
                f"sub-{sub}_ses-{ses}_task-{TASK}"
                f"_FirstLevel_contrast-{contrast_effect}"
                f"_stat-*_statmap.nii.gz"
            )
        )

        if not all_effect_files:
            print(
                f"  ⚠️ Missing effect size map for "
                f"subject {sub}, session {ses}, "
                f"contrast {contrast_effect}"
            )
            continue

        # Prefer stat-effect map if it exists
        effect_path = next(
            (
                ef
                for ef in all_effect_files
                if "_stat-effect_" in ef.name
            ),
            all_effect_files[0]
        )

        if (
            len(all_effect_files) > 1
            and "_stat-effect_" not in effect_path.name
        ):
            print(
                f"  ⚠️ Multiple matches for "
                f"{contrast_effect} in {sub}, {ses}, "
                f"using first one: {effect_path.name}"
            )

        # -------------------------------
        # Load effect map
        # -------------------------------
        img = nib.load(effect_path)
        img_data = img.get_fdata()

        # -------------------------------
        # Compute PSC per cluster
        # -------------------------------
        for clust_id in cluster_ids:

            mask = cluster_data == clust_id

            mean_psc = np.mean(
                img_data[mask]
            )

            psc_records.append(
                {
                    "subject": sub,
                    "session": ses,
                    "time": time,
                    "cluster_contrast": contrast_cluster,
                    "contrast_effect": contrast_effect,
                    "cluster": int(clust_id),
                    "PSC": mean_psc,
                }
            )

# -------------------------------
# Save PSC table
# -------------------------------
psc_df = pd.DataFrame(psc_records)

out_csv = (
    UNIVARIATE_DIR
    / "psc_all_contrasts.tsv"
)

psc_df.to_csv(
    out_csv,
    sep="\t",
    index=False
)

print(
    f"\n✅ Saved PSC data: "
    f"{out_csv} ({psc_df.shape[0]} rows)"
)

# -------------------------------
# Optional plotting
# -------------------------------
if not psc_df.empty and MAKE_PLOTS:

    psc_df["session_num"] = (
        psc_df["session"].astype(int)
    )

    # -----------------------------------------
    # Publication-friendly plot style
    # -----------------------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    # -----------------------------------------
    # One figure per contrast
    # -----------------------------------------
    for (
        contrast_cluster,
        contrast_effect
    ), sub_df in psc_df.groupby(
        ["cluster_contrast", "contrast_effect"]
    ):

        # Get all clusters for this contrast
        clusters = sorted(
            sub_df["cluster"].unique()
        )

        n_clusters = len(clusters)

        # -----------------------------------------
        # Determine subplot layout
        # -----------------------------------------
        if n_clusters == 1:
            ncols = 1
        elif n_clusters == 2:
            ncols = 2
        elif n_clusters == 3:
            ncols = 3
        elif n_clusters == 4:
            ncols = 2
        else:
            ncols = 3

        nrows = int(
            np.ceil(n_clusters / ncols)
        )

        # -----------------------------------------
        # Figure size
        # -----------------------------------------
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                5.5 * ncols,
                3.8 * nrows
            ),
            squeeze=False
        )

        axes = axes.flatten()

        # -----------------------------------------
        # Plot each cluster
        # -----------------------------------------
        for i, cluster_id in enumerate(clusters):

            ax = axes[i]

            df = sub_df[
                sub_df["cluster"] == cluster_id
            ].copy()

            # -------------------------------------
            # Plot individual subject trajectories
            # -------------------------------------
            for subject, subject_df in df.groupby("subject"):

                subject_df = subject_df.sort_values(
                    "session_num"
                )

                ax.plot(
                    subject_df["session_num"],
                    subject_df["PSC"],
                    color="gray",
                    alpha=0.35,
                    linewidth=1
                )

            # -------------------------------------
            # Calculate mean ± SEM across subjects
            # -------------------------------------
            agg_df = (
                df.groupby("session_num")["PSC"]
                .agg(["mean", "sem"])
                .reset_index()
            )

            # -------------------------------------
            # Plot group mean ± SEM
            # -------------------------------------
            ax.errorbar(
                agg_df["session_num"],
                agg_df["mean"],
                yerr=agg_df["sem"],
                fmt="-o",
                color="black",
                linewidth=2.5,
                markersize=5,
                capsize=4,
                capthick=1.5
            )

            # -------------------------------------
            # X-axis
            # -------------------------------------
            session_ticks = sorted(
                agg_df["session_num"].unique()
            )

            ax.set_xticks(session_ticks)
            ax.set_xticklabels(
                [str(x) for x in session_ticks]
            )

            ax.set_xlabel("Session")

            if session_ticks:
                ax.set_xlim(
                    min(session_ticks),
                    max(session_ticks)
                )

            # -------------------------------------
            # Y-axis
            # -------------------------------------
            ax.set_ylabel(
                "Percent Signal Change"
            )

            # -------------------------------------
            # Grid
            # -------------------------------------
            ax.grid(
                alpha=0.25
            )

            # -------------------------------------
            # Panel label
            # -------------------------------------
            ax.text(
                -0.18,
                1.05,
                chr(65 + i),
                transform=ax.transAxes,
                fontsize=15,
                fontweight="bold",
                va="top"
            )

            # -------------------------------------
            # Optional cluster title
            # -------------------------------------
            ax.set_title(
                f"Cluster {cluster_id}",
                fontsize=11
            )

        # -----------------------------------------
        # Remove unused subplot axes
        # -----------------------------------------
        for j in range(
            n_clusters,
            len(axes)
        ):
            fig.delaxes(axes[j])

        # -----------------------------------------
        # Final layout
        # -----------------------------------------
        plt.tight_layout()

        # -----------------------------------------
        # Output filenames
        # -----------------------------------------
        base_name = (
            f"psc_{contrast_effect}_all_clusters"
        )

        # -----------------------------------------
        # Save TIFF
        # -----------------------------------------
        fig.savefig(
            PLOT_DIR / f"{base_name}.tiff",
            dpi=600,
            bbox_inches="tight"
        )

        # -----------------------------------------
        # Save PDF
        # -----------------------------------------
        fig.savefig(
            PLOT_DIR / f"{base_name}.pdf",
            bbox_inches="tight"
        )

        # -----------------------------------------
        # Save SVG
        # -----------------------------------------
        fig.savefig(
            PLOT_DIR / f"{base_name}.svg",
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"  📈 Saved: "
            f"{base_name}.tiff / .pdf / .svg"
        )

    print(
        f"\n📈 PSC plots saved in: {PLOT_DIR}"
    )
