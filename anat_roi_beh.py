import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import mixedlm

BEHAVIOUR_FILE = "ddm_fit_results_simple.csv"

# choose anatomy file:
#STRUCTURAL_FILE = "curv_cluster_stats.csv"
STRUCTURAL_FILE = "vol_cluster_stats.csv"
#STRUCTURAL_FILE = "area_cluster_stats.csv"
# choose measure:
# "ThickAvg"
# "SurfArea"
# "GrayVol"
# "MeanCurv"

STRUCTURAL_MEASURE = "GrayVol"

BEHAVIOUR_MEASURE = "drift_v"

# LOAD DATA

behaviour = pd.read_csv(BEHAVIOUR_FILE)

structure = pd.read_csv(
    STRUCTURAL_FILE
)

# Extract subject and session from behavioural IDs
behaviour["Subject"] = (
    "sub-" +
    behaviour["subjID"]
    .str.lower()
    .str.extract(r"(.*)_sess")[0]
)

behaviour["Session"] = (
    "ses-" +
    behaviour["subjID"]
    .str.extract(r"sess([0-9]+)")[0]
    .str.zfill(2)
)

# STRUCTURE LONG FORMAT

structure = structure[
    structure["Session"].isin(
        ["ses-01","ses-02","ses-03","ses-04"]
    )
]


# convert session to number
structure["session_num"] = (
    structure["Session"]
    .str.extract("([0-9]+)")
    .astype(int)
)

#merge
print("\nBehaviour IDs:")
print(behaviour[["Subject","Session"]].drop_duplicates().head(10))

print("\nAnatomy IDs:")
print(structure[["Subject","Session"]].drop_duplicates().head(10))
merged = pd.merge(
    behaviour,
    structure,
    on=["Subject","Session"]
)


print(
    "Merged rows:",
    len(merged)
)

print(
    "Participants:",
    merged.Subject.nunique()
)

rois = merged["Cluster"].unique()

print("\nROIs:")
print(rois)

#LMMs
results = []


for roi in rois:

    roi_data = merged[
        merged["Cluster"] == roi
    ].copy()


    roi_data = roi_data.dropna(
        subset=[
            BEHAVIOUR_MEASURE,
            STRUCTURAL_MEASURE
        ]
    )


    if len(roi_data) < 10:
        continue


    model = mixedlm(
        f"{BEHAVIOUR_MEASURE} ~ {STRUCTURAL_MEASURE} + session_num",
        roi_data,
        groups=roi_data["Subject"]
    )


    fit = model.fit(
        reml=False
    )


    results.append({

        "ROI": roi,

        "beta_anatomy":
            fit.params[STRUCTURAL_MEASURE],

        "p_anatomy":
            fit.pvalues[STRUCTURAL_MEASURE],

        "beta_session":
            fit.params["session_num"],

        "p_session":
            fit.pvalues["session_num"],

        "N":
            len(roi_data)

    })


results_df = pd.DataFrame(results)

from statsmodels.stats.multitest import multipletests

results_df["p_fdr"] = multipletests(
    results_df["p_anatomy"],
    method="fdr_bh"
)[1]

print(
    results_df.sort_values("p_fdr")
)


print("\nMixed model results:")
print(
    results_df.sort_values(
        "p_anatomy"
    )
)


# plot1
corr_results=[]


for session in sorted(
    merged.session_num.unique()
):

    session_data = merged[
        merged.session_num == session
    ]

    for roi in rois:

        tmp=session_data[
            session_data.Cluster==roi
        ]

        if len(tmp)>2:

            r=np.corrcoef(
                tmp[STRUCTURAL_MEASURE],
                tmp[BEHAVIOUR_MEASURE]
            )[0,1]


            corr_results.append({

                "ROI":roi,
                "Session":session,
                "r":r

            })


corr_df=pd.DataFrame(
    corr_results
)


plt.figure(figsize=(12,6))

sns.lineplot(
    data=corr_df,
    x="Session",
    y="r",
    hue="ROI",
    marker="o"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.ylabel(
    "Correlation (r)"
)

plt.title(
    f"{STRUCTURAL_MEASURE} vs {BEHAVIOUR_MEASURE} across sessions"
)

plt.tight_layout()

plt.show()

# plot2
top_roi = (
    results_df
    .sort_values("p_anatomy")
    .iloc[0]["ROI"]
)


plot_data = merged[
    merged.Cluster == top_roi
]


plt.figure(figsize=(8,5))


sns.lineplot(
    data=plot_data,
    x="session_num",
    y=STRUCTURAL_MEASURE,
    hue="Subject",
    legend=False
)


plt.title(
    f"Individual trajectories: {top_roi}"
)

plt.xlabel(
    "Session"
)

plt.ylabel(
    STRUCTURAL_MEASURE
)

plt.tight_layout()

plt.show()

#plot3
plt.figure(figsize=(8,5))


sns.lineplot(
    data=merged,
    x="session_num",
    y=BEHAVIOUR_MEASURE,
    hue="Subject",
    legend=False
)


plt.title(
    "Individual behavioural trajectories"
)

plt.xlabel(
    "Session"
)

plt.ylabel(
    BEHAVIOUR_MEASURE
)

plt.tight_layout()

plt.show()
