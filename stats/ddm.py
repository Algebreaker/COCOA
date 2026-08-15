import pandas as pd
import pyddm
from pyddm import Model, Fittable, Sample
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPointSourceCenter, Drift
from pyddm.functions import fit_adjust_model
from pyddm.models import Drift  # base class
from pyddm import Fittable
import pyddm.plot
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/home/johanna/DDM/ddm_input.csv", delimiter='\t')
print(df.columns)

# Recode: 2 = correct (→ 1), 1 = incorrect (→ 0)
df["response_code"] = df["choice"].map({1: 0, 2: 1})

# Drop any bad RTs
df = df[(df["RT"] > 0.2) & (df["RT"] < 5.0)].copy()

# Loop over each session ID (unique participant-session combo)
results = []

for sid, group in df.groupby("subjID"):
    print(f"Fitting model for {sid}...")

    # Create a PyDDM Sample
    sample = Sample.from_pandas_dataframe(
        group,
        rt_column_name="RT",
        choice_column_name="response_code"  # required by newer PyDDM
    )

    # Define and fit model
    m = pyddm.gddm(
    drift="v",
    noise=1,
    bound="a",
    nondecision="ndt",
    parameters={"v": (-5, 5), "a": (0.3, 3), "ndt": (0.1, 3)},
    T_dur=5.0 
    )


    # Fit model to participant-session sample
    m.fit(sample, verbose=False)

    params = m.parameters()
    print(f"Fitted parameter keys for {sid}: {list(params.keys())}")
    print(params)

    results.append({
    "subjID": sid,
    "drift_v": float(params['drift']['drift']),
    "bound_a": float(params['bound']['B']),
    "nondec_t": float(params['overlay']['nondectime'])
    })


# Save results
pd.DataFrame(results).to_csv("ddm_fit_results_simple.csv", index=False)
