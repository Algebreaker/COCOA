import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ---------- Read data ----------
df = pd.read_csv("ddm_fit_results_simple.csv")

# Extract subject and session
df["subject"] = df["subjID"].str.extract(r'^(.*?)_sess')
df["session"] = df["subjID"].str.extract(r'sess(\d+)')

# Drop missing values
df = df.dropna(subset=["subject", "session"])

# Ensure correct data types
df["subject"] = df["subject"].astype(str)
df["session"] = df["session"].astype(int)

# ---------- Metrics ----------
metrics = ["drift_v", "bound_a", "nondec_t"]

# Publication-friendly labels
ylabels = {
    "drift_v": "Drift rate (v)",
    "bound_a": "Boundary separation (a)",
    "nondec_t": "Non-decision time (s)"
}

panel_labels = ["A", "B", "C"]

# ---------- Plot style ----------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ---------- Create figure ----------
fig, axes = plt.subplots(
    1,
    3,
    figsize=(11, 3.8),
    sharex=True
)

for ax, metric, panel in zip(axes, metrics, panel_labels):

    print(f"\n{'='*40}")
    print(f"Mixed-effects regression for {metric}")
    print(f"{'='*40}")

    # Mixed-effects model
    model = smf.mixedlm(
        f"{metric} ~ session",
        data=df,
        groups=df["subject"]
    )

    result = model.fit()

    print(result.summary())

    # Plot individual subject trajectories
    for subj, subdf in df.groupby("subject"):
        ax.plot(
            subdf["session"],
            subdf[metric],
            color="gray",
            alpha=0.35,
            linewidth=1
        )

    # Mixed model prediction line
    x_pred = pd.DataFrame({
        "session": [1, 2, 3, 4]
    })

    y_pred = result.predict(x_pred)

    ax.plot(
        x_pred["session"],
        y_pred,
        color="black",
        linewidth=2.5
    )

    # Axis labels
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabels[metric])

    # Force session ticks to be integers only
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlim(1, 4)

    # Panel label
    ax.text(
        -0.18,
        1.05,
        panel,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top"
    )

    ax.grid(alpha=0.25)


# ---------- Final formatting ----------
plt.tight_layout()

# Save publication-quality versions
plt.savefig(
    "DDM_session_effects.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "DDM_session_effects.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "DDM_session_effects.svg",
    bbox_inches="tight"
)

plt.show()
