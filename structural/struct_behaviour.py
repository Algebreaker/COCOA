import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

# user options for measures
BEHAVIOUR_FILE = "ddm_fit_results_simple.csv"
STRUCTURAL_FILE = "vol_cluster_stats.csv" #select curv, area or vol
STRUCTURAL_MEASURE = "GrayVol" # "SurfArea" "GrayVol" or "MeanCurv"
BEHAVIOUR_MEASURE = "accuracy" #select drift_v or accuracy

MIN_SESSION, MAX_SESSION = 1, 4
FDR_ALPHA = 0.05

LME_OUTPUT = f"LME_within_person_{STRUCTURAL_MEASURE}_{BEHAVIOUR_MEASURE}.csv"
CHANGE_OUTPUT = f"baseline_anatomy_predicts_maximum_behaviour_change_{STRUCTURAL_MEASURE}_{BEHAVIOUR_MEASURE}.csv"

# functions
def apply_fdr(df, p_col):
    """Applies Benjamini-Hochberg FDR correction to a dataframe column."""
    df["p_FDR"] = np.nan
    df["significant_FDR"] = False
    valid = df[p_col].notna()
    if valid.any():
        reject, p_fdr = multipletests(df.loc[valid, p_col], alpha=FDR_ALPHA, method="fdr_bh")[:2]
        df.loc[valid, "p_FDR"] = p_fdr
        df.loc[valid, "significant_FDR"] = reject
    return df

def fit_lme_roi(roi, roi_data):
    """Fits LME with random intercept + random session slope."""
    formula = f"{BEHAVIOUR_MEASURE} ~ anatomy_within + anatomy_between_c + session_c"
    model = mixedlm(formula, roi_data, groups=roi_data["Subject"], re_formula="~session_c")
    
    fit, optimizer = None, None
    for opt in ["lbfgs", "powell"]:
        try:
            fit = model.fit(method=opt, reml=False, maxiter=5000, disp=False)
            optimizer = opt
            print(f"  Random-slope model converged using {opt}.")
            break
        except Exception as e:
            print(f"  {opt} failed: {e}")
            
    if fit is None:
        raise RuntimeError("All optimizers failed.")
        
    ci_low, ci_high = fit.conf_int().loc["anatomy_within"]
    return {
        "ROI": roi, "model": "random_slope", "optimizer": optimizer, "status": "converged",
        "beta_anatomy_within": fit.params["anatomy_within"], "CI_low": ci_low, "CI_high": ci_high,
        "p_anatomy_within": fit.pvalues["anatomy_within"],
        "beta_anatomy_between": fit.params["anatomy_between_c"], "p_anatomy_between": fit.pvalues["anatomy_between_c"],
        "beta_session": fit.params["session_c"], "p_session": fit.pvalues["session_c"],
        "n_subjects": roi_data["Subject"].nunique(), "n_observations": len(roi_data)
    }

# load data
print("\n" + "="*45 + "\nLOADING & MERGING DATA\n" + "="*45)
behaviour = pd.read_csv(BEHAVIOUR_FILE)
structure = pd.read_csv(STRUCTURAL_FILE)

# standardise IDs
behaviour["Subject"] = "sub-" + behaviour["subjID"].str.lower().str.extract(r"(.*)_sess")[0]
behaviour["Session"] = "ses-" + behaviour["subjID"].str.extract(r"sess([0-9]+)")[0].str.zfill(2)

structure["session_num"] = structure["Session"].str.extract(r"([0-9]+)").astype(float)
structure = structure[structure["session_num"].between(MIN_SESSION, MAX_SESSION)].copy()
structure["session_num"] = structure["session_num"].astype(int)

# merge & filter
merged = pd.merge(behaviour, structure, on=["Subject", "Session"], how="inner")
merged = merged.dropna(subset=[BEHAVIOUR_MEASURE, STRUCTURAL_MEASURE, "Subject", "session_num"]).copy()
merged["session_c"] = merged["session_num"] - merged["session_num"].mean()

print(f"Merged observations: {len(merged)} | Unique Subjects: {merged['Subject'].nunique()}")
print("Participants per session:\n", merged.groupby("session_num")["Subject"].nunique())

# ANALYSIS 1: WITHIN-PERSON LONGITUDINAL LME
print("\n" + "="*45 + "\nANALYSIS 1: WITHIN-PERSON LME\n" + "="*45)
rois = sorted(merged["Cluster"].dropna().unique())
lme_results = []

for roi in rois:
    print(f"\n----------------------------------------------\nROI: {roi}\n----------------------------------------------")
    roi_data = merged[merged["Cluster"] == roi].dropna(subset=[BEHAVIOUR_MEASURE, STRUCTURAL_MEASURE, "session_c"]).copy()
    
    if roi_data["Subject"].nunique() < 5:
        print("Skipping: too few participants")
        continue

    # ROI-specific within/between decomposition
    roi_data["anatomy_between"] = roi_data.groupby("Subject")[STRUCTURAL_MEASURE].transform("mean")
    roi_data["anatomy_within"] = roi_data[STRUCTURAL_MEASURE] - roi_data["anatomy_between"]
    roi_data["anatomy_between_c"] = roi_data["anatomy_between"] - roi_data["anatomy_between"].mean()

    if roi_data["anatomy_within"].std() == 0:
        print("Skipping: no within-person anatomical variation")
        continue

    try:
        res = fit_lme_roi(roi, roi_data)
        lme_results.append(res)
        print(f"Within-person beta: {res['beta_anatomy_within']:.4f}, p: {res['p_anatomy_within']:.4f}")
    except Exception as e:
        print(f"LME FAILED: {e}")
        lme_results.append({
            "ROI": roi, "model": "random_slope", "optimizer": "failed", "status": "failed",
            "beta_anatomy_within": np.nan, "CI_low": np.nan, "CI_high": np.nan, "p_anatomy_within": np.nan,
            "beta_anatomy_between": np.nan, "p_anatomy_between": np.nan, "beta_session": np.nan, "p_session": np.nan,
            "n_subjects": roi_data["Subject"].nunique(), "n_observations": len(roi_data)
        })

lme_results_df = apply_fdr(pd.DataFrame(lme_results), "p_anatomy_within")

print("\n" + "="*45 + "\nLME RESULTS\n" + "="*45)
print(lme_results_df.sort_values("p_FDR").to_string(index=False) if not lme_results_df.empty else "No LME results.")
lme_results_df.to_csv(LME_OUTPUT, index=False)

# ANALYSIS 2: BASELINE ANATOMY -> BEHAVIOURAL CHANGE
print("\n" + "="*45 + "\nANALYSIS 2: BASELINE -> CHANGE OLS\n" + "="*45)

baseline_df = merged[merged["session_num"] == MIN_SESSION][["Subject", "Cluster", BEHAVIOUR_MEASURE, STRUCTURAL_MEASURE]].rename(
    columns={BEHAVIOUR_MEASURE: "baseline_behaviour", STRUCTURAL_MEASURE: "baseline_anatomy"}
)

# Keep max available follow-up session per subject
followup_all = merged[merged["session_num"] > MIN_SESSION]
idx_max = followup_all.groupby(["Subject", "Cluster"])["session_num"].idxmax()
followup_df = followup_all.loc[idx_max, ["Subject", "Cluster", "session_num", BEHAVIOUR_MEASURE]].rename(
    columns={BEHAVIOUR_MEASURE: "followup_behaviour", "session_num": "endpoint_session"}
)

change_df = pd.merge(baseline_df, followup_df, on=["Subject", "Cluster"], how="inner")
change_df["delta_behaviour"] = change_df["followup_behaviour"] - change_df["baseline_behaviour"]
change_df["followup_duration"] = change_df["endpoint_session"] - MIN_SESSION

change_results = []

for roi in rois:
    roi_data = change_df[change_df["Cluster"] == roi].dropna(
        subset=["delta_behaviour", "baseline_anatomy", "baseline_behaviour", "followup_duration"]
    ).copy()
    
    n_sub = roi_data["Subject"].nunique()
    if n_sub < 5 or roi_data["baseline_anatomy"].std() == 0:
        continue

    # Standardise baseline measures
    roi_data["baseline_anatomy_z"] = (roi_data["baseline_anatomy"] - roi_data["baseline_anatomy"].mean()) / roi_data["baseline_anatomy"].std()
    b_std = roi_data["baseline_behaviour"].std()
    roi_data["baseline_behaviour_z"] = (roi_data["baseline_behaviour"] - roi_data["baseline_behaviour"].mean()) / b_std if b_std > 0 else 0

    X = sm.add_constant(roi_data[["baseline_anatomy_z", "baseline_behaviour_z", "followup_duration"]])
    y = roi_data["delta_behaviour"]

    try:
        ols = sm.OLS(y, X).fit()
        ci_low, ci_high = ols.conf_int().loc["baseline_anatomy_z"]
        change_results.append({
            "ROI": roi,
            "beta_baseline_anatomy": ols.params["baseline_anatomy_z"], "CI_low": ci_low, "CI_high": ci_high,
            "p_anatomy": ols.pvalues["baseline_anatomy_z"],
            "beta_baseline_behaviour": ols.params["baseline_behaviour_z"], "p_baseline_behaviour": ols.pvalues["baseline_behaviour_z"],
            "beta_followup_duration": ols.params["followup_duration"], "p_followup_duration": ols.pvalues["followup_duration"],
            "R_squared": ols.rsquared, "n_subjects": n_sub
        })
    except Exception as e:
        print(f"OLS FAILED for {roi}: {e}")

change_results_df = apply_fdr(pd.DataFrame(change_results), "p_anatomy")

print("\n" + "="*45 + "\nCHANGE RESULTS\n" + "="*45)
print(change_results_df.sort_values("p_FDR").to_string(index=False) if not change_results_df.empty else "No change results.")
change_results_df.to_csv(CHANGE_OUTPUT, index=False)

#plots
valid_lme = lme_results_df[lme_results_df["p_anatomy_within"].notna()]
if not valid_lme.empty:
    top_lme = valid_lme.sort_values("p_FDR").iloc[0]
    top_roi = top_lme["ROI"]
    
    p_data = merged[merged["Cluster"] == top_roi].copy()
    p_data["anatomy_within"] = p_data[STRUCTURAL_MEASURE] - p_data.groupby("Subject")[STRUCTURAL_MEASURE].transform("mean")

    # Plot 1: Within-Person Effect
    plt.figure(figsize=(8, 6))
    sns.regplot(data=p_data, x="anatomy_within", y=BEHAVIOUR_MEASURE, scatter_kws={"s": 50}, line_kws={"color": "black"})
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.xlabel(f"Within-person deviation in {STRUCTURAL_MEASURE}")
    plt.ylabel(BEHAVIOUR_MEASURE)
    plt.title(f"{top_roi}\nWithin-person β = {top_lme['beta_anatomy_within']:.3f}, FDR p = {top_lme['p_FDR']:.4f}")
    plt.tight_layout()
    plt.show()

    # Plot 3: Anatomical Trajectories
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=p_data, x="session_num", y=STRUCTURAL_MEASURE, hue="Subject", marker="o", legend=False)
    plt.xlabel("Session")
    plt.ylabel(STRUCTURAL_MEASURE)
    plt.title(f"Individual anatomical trajectories\n{top_roi}")
    plt.tight_layout()
    plt.show()

valid_change = change_results_df[change_results_df["p_anatomy"].notna()]
if not valid_change.empty:
    top_change = valid_change.sort_values("p_FDR").iloc[0]
    top_c_roi = top_change["ROI"]

    p_change = change_df[change_df["Cluster"] == top_c_roi].copy()
    p_change["baseline_anatomy_z"] = (p_change["baseline_anatomy"] - p_change["baseline_anatomy"].mean()) / p_change["baseline_anatomy"].std()

    # Plot 2: Baseline Anatomy vs Change
    plt.figure(figsize=(8, 6))
    sns.regplot(data=p_change, x="baseline_anatomy_z", y="delta_behaviour", scatter_kws={"s": 60}, line_kws={"color": "black"})
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.axvline(0, linestyle="--", linewidth=0.8)
    plt.xlabel(f"Baseline {STRUCTURAL_MEASURE} (within-ROI z-score)")
    plt.ylabel(f"Change in {BEHAVIOUR_MEASURE}")
    plt.title(f"{top_c_roi}\nβ = {top_change['beta_baseline_anatomy']:.3f}, FDR p = {top_change['p_FDR']:.4f}")
    plt.tight_layout()
    plt.show()

# Plot 4: Behavioural Trajectories
plt.figure(figsize=(8, 5))
sns.lineplot(data=merged, x="session_num", y=BEHAVIOUR_MEASURE, hue="Subject", marker="o", legend=False)
plt.xlabel("Session")
plt.ylabel(BEHAVIOUR_MEASURE)
plt.title("Individual behavioural trajectories")
plt.tight_layout()
plt.show()

# Plot 2: Adjusted baseline anatomy vs behavioural change
# -------------------------------------------------------
# This visualises the partial relationship represented by:
# delta_behaviour ~ baseline_anatomy_z + baseline_behaviour_z + followup_duration

p_change = change_df[change_df["Cluster"] == top_c_roi].copy()

# Standardise predictors in the same way as the OLS analysis
p_change["baseline_anatomy_z"] = (
    p_change["baseline_anatomy"] - p_change["baseline_anatomy"].mean()
) / p_change["baseline_anatomy"].std()

p_change["baseline_behaviour_z"] = (
    p_change["baseline_behaviour"] - p_change["baseline_behaviour"].mean()
) / p_change["baseline_behaviour"].std()

# Residualise anatomy against the covariates
X_cov = sm.add_constant(
    p_change[["baseline_behaviour_z", "followup_duration"]]
)
anatomy_model = sm.OLS(
    p_change["baseline_anatomy_z"], X_cov
).fit()

# Residualise behavioural change against the same covariates
change_model = sm.OLS(
    p_change["delta_behaviour"], X_cov
).fit()

p_change["anatomy_resid"] = anatomy_model.resid
p_change["change_resid"] = change_model.resid

# Plot partial relationship
plt.figure(figsize=(8, 6))

sns.regplot(
    data=p_change,
    x="anatomy_resid",
    y="change_resid",
    scatter_kws={"s": 60},
    line_kws={"color": "black"}
)

plt.axhline(0, linestyle="--", linewidth=0.8)
plt.axvline(0, linestyle="--", linewidth=0.8)

plt.xlabel(
    f"Baseline {STRUCTURAL_MEASURE}\n"
    "adjusted for baseline behaviour and follow-up duration"
)
plt.ylabel(
    f"Change in {BEHAVIOUR_MEASURE}\n"
    "adjusted for baseline behaviour and follow-up duration"
)

plt.title(
    f"{top_c_roi}\n"
    f"Adjusted β = {top_change['beta_baseline_anatomy']:.3f}, "
    f"FDR p = {top_change['p_FDR']:.4f}"
)

plt.tight_layout()
plt.show()

print("\n" + "="*45 + "\nANALYSIS COMPLETE\n" + "="*45)