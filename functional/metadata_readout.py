import pandas as pd

# Replace with your file/path
df = pd.read_csv(
    "/data/pt_02825/MPCDF/univariate7mm/metadata_task-Literacy.tsv",
    sep="\t"
)

print(df.columns.tolist())
print(df.head())
# Ensure datetime format
df["acq_time"] = pd.to_datetime(df["acq_time"])


# Keep only valid sessions
valid_df = df[df["good_ixs"] == "Y"].copy()


# Basic summary statistics
n_valid_participants = valid_df["subject"].nunique()
n_valid_sessions = len(valid_df)
n_valid_runs = valid_df["valid_runs"].sum()

# Days between sessions
valid_df = valid_df.sort_values(["subject", "session"])
valid_df["days_since_prev"] = (
    valid_df.groupby("subject")["acq_time"]
    .diff()
    .dt.total_seconds()
    / (24 * 60 * 60)
)

# Remove first session for each participant (NaN interval)
session_intervals = valid_df.dropna(subset=["days_since_prev"]).copy()
overall_mean_days = session_intervals["days_since_prev"].mean()
overall_sd_days = session_intervals["days_since_prev"].std()
transition_stats = (
    session_intervals.groupby("session")["days_since_prev"]
    .agg(["mean", "std", "count"])
    .rename(
        index={
            2: "ses01→ses02",
            3: "ses02→ses03",
            4: "ses03→ses04",
        }
    )
)


# Print results
print("=" * 60)
print("VALID DATA SUMMARY")
print("=" * 60)

print(f"Valid participants : {n_valid_participants}")
print(f"Valid sessions     : {n_valid_sessions}")
print(f"Valid runs         : {n_valid_runs}")

print("\n" + "=" * 60)
print("SESSION INTERVALS")
print("=" * 60)

print(
    f"Overall interval: "
    f"{overall_mean_days:.2f} ± {overall_sd_days:.2f} days "
    f"(n={len(session_intervals)})"
)

print("\nMean ± SD by transition:")
for transition, row in transition_stats.iterrows():
    print(
        f"{transition}: "
        f"{row['mean']:.2f} ± {row['std']:.2f} days "
        f"(n={int(row['count'])})"
    )

# Optional: participant-level mean intervals
participant_stats = (
    session_intervals.groupby("subject")["days_since_prev"]
    .agg(["mean", "std", "count"])
    .round(2)
)

print("\n" + "=" * 60)
print("PARTICIPANT-LEVEL INTERVALS")
print("=" * 60)
print(participant_stats)
