from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# user options
CLUSTER_MASK = Path(
    "/data/pt_02825/MPCDF/univariate7mm/"
    "task-Literacy_space-MNI152NLin6Asym_"
    "desc-congruentMinusIncongruent_z1-clusters.nii.gz"
)

EFFECT_DIR = Path("/data/pt_02825/MPCDF/univariate7mm/derivatives")
BEHAV_FILE = Path("/data/pt_02825/MPCDF/ddm_fit_results_simple.csv")

BEHAVIOUR_MEASURE = "drift_v"  # Column name in CSV
MIN_SESSION, MAX_SESSION = 1, 4
FDR_ALPHA = 0.05

LME_OUTPUT = "fmri_cluster_lme_within_person_results.csv"
CHANGE_OUTPUT = "baseline_activation_predicts_behaviour_change.csv"


# functions
def apply_fdr(df, p_col):
    """Applies Benjamini-Hochberg FDR correction to a specified p-value column."""
    df["p_FDR"] = np.nan
    df["significant_FDR"] = False
    valid = df[p_col].notna()
    if valid.any():
        reject, p_fdr = multipletests(df.loc[valid, p_col], alpha=FDR_ALPHA, method="fdr_bh")[:2]
        df.loc[valid, "p_FDR"] = p_fdr
        df.loc[valid, "significant_FDR"] = reject
    return df


def fit_lme_cluster(cluster_id, roi_data):
    """Fits LME with within/between activation decomposition & random session slope."""
    formula = f"{BEHAVIOUR_MEASURE} ~ activation_within + activation_between_c + session_c"
    model = mixedlm(formula, roi_data, groups=roi_data["subject"], re_formula="~session_c")
    
    fit, optimizer = None, None
    for opt in ["lbfgs", "powell"]:
        try:
            fit = model.fit(method=opt, reml=False, maxiter=5000, disp=False)
            optimizer = opt
            print(f"  Cluster {cluster_id}: Converged using {opt}.")
            break
        except Exception as e:
            print(f"  Cluster {cluster_id} - {opt} failed: {e}")
            
    # Fallback to random intercept only if random slope fails
    if fit is None:
        try:
            model_fallback = mixedlm(formula, roi_data, groups=roi_data["subject"])
            fit = model_fallback.fit(method="powell", reml=False, maxiter=5000, disp=False)
            optimizer = "powell_random_intercept_only"
            print(f"  Cluster {cluster_id}: Converged using random intercept fallback.")
        except Exception as e:
            raise RuntimeError(f"All optimizers failed: {e}")
        
    ci_low, ci_high = fit.conf_int().loc["activation_within"]
    return {
        "cluster": cluster_id,
        "optimizer": optimizer,
        "status": "converged",
        "beta_activation_within": fit.params["activation_within"],
        "CI_low": ci_low,
        "CI_high": ci_high,
        "p_activation_within": fit.pvalues["activation_within"],
        "beta_activation_between": fit.params["activation_between_c"],
        "p_activation_between": fit.pvalues["activation_between_c"],
        "beta_session": fit.params["session_c"],
        "p_session": fit.pvalues["session_c"],
        "n_subjects": roi_data["subject"].nunique(),
        "n_observations": len(roi_data)
    }


# load data
print("\n" + "="*50 + "\n1. LOADING MASK & BEHAVIOUR\n" + "="*50)
cluster_img = nib.load(CLUSTER_MASK)
cluster_data = cluster_img.get_fdata()
cluster_labels = np.unique(cluster_data)
cluster_labels = sorted([int(c) for c in cluster_labels if c > 0])

print(f"Clusters found ({len(cluster_labels)}): {cluster_labels}")

beh = pd.read_csv(BEHAV_FILE)
beh["subject"] = beh["subjID"].str.extract(r"^(.*)_sess")[0].str.lower().str.strip()
beh["session"] = beh["subjID"].str.extract(r"sess(\d+)")[0].astype(int)

beh = beh[beh["session"].between(MIN_SESSION, MAX_SESSION)].copy()


# get signal intensities
print("\n" + "="*50 + "\n2. EXTRACTING ROI ACTIVATION MAPS\n" + "="*50)
rows, missing = [], []

for _, row in beh.iterrows():
    subj, sess = row["subject"], row["session"]
    ses_str = f"{sess:02d}"

    effect_file = (
        EFFECT_DIR
        / f"sub-{subj}"
        / f"sub-{subj}_ses-{ses_str}_task-Literacy_FirstLevel_contrast-congruentMinusIncongruent_stat-effect_statmap.nii.gz"
    )

    if not effect_file.exists():
        missing.append(effect_file)
        continue

    effect_data = nib.load(effect_file).get_fdata()

    for cluster in cluster_labels:
        mask = cluster_data == cluster
        activation = np.nanmean(effect_data[mask])
        rows.append({
            "subject": subj,
            "session": sess,
            "cluster": cluster,
            "activation": activation
        })

activation_df = pd.DataFrame(rows)
print(f"Extracted {len(activation_df)} activation points. Missing effect files: {len(missing)}")


# merge
merged = pd.merge(activation_df, beh, on=["subject", "session"], how="inner")
merged = merged.dropna(subset=[BEHAVIOUR_MEASURE, "activation", "subject", "session"]).copy()

merged["session_c"] = merged["session"] - merged["session"].mean()

print(f"Merged Data: {len(merged)} obs | {merged['subject'].nunique()} unique subjects")


# within person longitudinal LME
print("\n" + "="*50 + "\n4. ANALYSIS 1: WITHIN-PERSON LONGITUDINAL LME\n" + "="*50)
lme_results = []

for cluster in cluster_labels:
    roi_data = merged[merged["cluster"] == cluster].copy()
    
    if roi_data["subject"].nunique() < 5:
        print(f"Cluster {cluster}: Skipped (N < 5 subjects)")
        continue

    # Disentangle within-person from between-person activation
    roi_data["activation_between"] = roi_data.groupby("subject")["activation"].transform("mean")
    roi_data["activation_within"] = roi_data["activation"] - roi_data["activation_between"]
    roi_data["activation_between_c"] = roi_data["activation_between"] - roi_data["activation_between"].mean()

    if roi_data["activation_within"].std() == 0:
        print(f"Cluster {cluster}: Skipped (No within-person variance)")
        continue

    try:
        res = fit_lme_cluster(cluster, roi_data)
        lme_results.append(res)
    except Exception as e:
        print(f"Cluster {cluster}: LME FAILED - {e}")
        lme_results.append({
            "cluster": cluster, "optimizer": "failed", "status": "failed",
            "beta_activation_within": np.nan, "CI_low": np.nan, "CI_high": np.nan,
            "p_activation_within": np.nan, "beta_activation_between": np.nan,
            "p_activation_between": np.nan, "beta_session": np.nan, "p_session": np.nan,
            "n_subjects": roi_data["subject"].nunique(), "n_observations": len(roi_data)
        })

lme_results_df = apply_fdr(pd.DataFrame(lme_results), "p_activation_within")
lme_results_df.to_csv(LME_OUTPUT, index=False)

print("\nLME Results Summary:")
print(lme_results_df[["cluster", "beta_activation_within", "p_activation_within", "p_FDR", "significant_FDR"]])


# baseline predicting behaviour
print("\n" + "="*50 + "\n5. ANALYSIS 2: BASELINE ACTIVATION -> BEHAVIOURAL CHANGE\n" + "="*50)

# Baseline data (Session = MIN_SESSION)
baseline_df = merged[merged["session"] == MIN_SESSION][["subject", "cluster", BEHAVIOUR_MEASURE, "activation"]].rename(
    columns={BEHAVIOUR_MEASURE: "baseline_behaviour", "activation": "baseline_activation"}
)

# Max available follow-up session per subject
followup_all = merged[merged["session"] > MIN_SESSION]
idx_max = followup_all.groupby(["subject", "cluster"])["session"].idxmax()
followup_df = followup_all.loc[idx_max, ["subject", "cluster", "session", BEHAVIOUR_MEASURE]].rename(
    columns={BEHAVIOUR_MEASURE: "followup_behaviour", "session": "endpoint_session"}
)

change_df = pd.merge(baseline_df, followup_df, on=["subject", "cluster"], how="inner")
change_df["delta_behaviour"] = change_df["followup_behaviour"] - change_df["baseline_behaviour"]
change_df["followup_duration"] = change_df["endpoint_session"] - MIN_SESSION

change_results = []

for cluster in cluster_labels:
    roi_data = change_df[change_df["cluster"] == cluster].dropna(
        subset=["delta_behaviour", "baseline_activation", "baseline_behaviour", "followup_duration"]
    ).copy()
    
    n_sub = roi_data["subject"].nunique()
    if n_sub < 5 or roi_data["baseline_activation"].std() == 0:
        continue

    # Standardise baseline variables
    roi_data["baseline_activation_z"] = (roi_data["baseline_activation"] - roi_data["baseline_activation"].mean()) / roi_data["baseline_activation"].std()
    b_std = roi_data["baseline_behaviour"].std()
    roi_data["baseline_behaviour_z"] = (roi_data["baseline_behaviour"] - roi_data["baseline_behaviour"].mean()) / b_std if b_std > 0 else 0

    X = sm.add_constant(roi_data[["baseline_activation_z", "baseline_behaviour_z", "followup_duration"]])
    y = roi_data["delta_behaviour"]

    try:
        ols = sm.OLS(y, X).fit()
        ci_low, ci_high = ols.conf_int().loc["baseline_activation_z"]
        change_results.append({
            "cluster": cluster,
            "beta_baseline_activation": ols.params["baseline_activation_z"],
            "CI_low": ci_low,
            "CI_high": ci_high,
            "p_activation": ols.pvalues["baseline_activation_z"],
            "beta_baseline_behaviour": ols.params["baseline_behaviour_z"],
            "p_baseline_behaviour": ols.pvalues["baseline_behaviour_z"],
            "beta_followup_duration": ols.params["followup_duration"],
            "p_followup_duration": ols.pvalues["followup_duration"],
            "R_squared": ols.rsquared,
            "n_subjects": n_sub
        })
    except Exception as e:
        print(f"OLS FAILED for Cluster {cluster}: {e}")

change_results_df = apply_fdr(pd.DataFrame(change_results), "p_activation")
change_results_df.to_csv(CHANGE_OUTPUT, index=False)

print("\nChange Results Summary:")
print(change_results_df[["cluster", "beta_baseline_activation", "p_activation", "p_FDR", "significant_FDR"]])


# plots
valid_change = change_results_df[change_results_df["p_activation"].notna()]

if not valid_change.empty:
    top_cluster = valid_change.sort_values("p_FDR").iloc[0]["cluster"]
    p_change = change_df[change_df["cluster"] == top_cluster].copy()

    # Residualise baseline activation and behavioral change against covariates
    p_change["baseline_activation_z"] = (p_change["baseline_activation"] - p_change["baseline_activation"].mean()) / p_change["baseline_activation"].std()
    p_change["baseline_behaviour_z"] = (p_change["baseline_behaviour"] - p_change["baseline_behaviour"].mean()) / p_change["baseline_behaviour"].std()

    X_cov = sm.add_constant(p_change[["baseline_behaviour_z", "followup_duration"]])
    
    p_change["act_resid"] = sm.OLS(p_change["baseline_activation_z"], X_cov).fit().resid
    p_change["change_resid"] = sm.OLS(p_change["delta_behaviour"], X_cov).fit().resid

    plt.figure(figsize=(8, 6))
    sns.regplot(data=p_change, x="act_resid", y="change_resid", scatter_kws={"s": 60}, line_kws={"color": "black"})
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.axvline(0, linestyle="--", linewidth=0.8)
    plt.xlabel("Baseline fMRI Activation (Adjusted Residuals)")
    plt.ylabel(f"Change in {BEHAVIOUR_MEASURE} (Adjusted Residuals)")
    plt.title(f"Cluster {top_cluster}\nBaseline Activation Predicts Behavioral Change")
    plt.tight_layout()
    plt.show()

print("\n" + "="*50 + "\nPROCESSING COMPLETE\n" + "="*50)
