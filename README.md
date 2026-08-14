


# Functional Analysis
All scripts are found in the 'functional' folder. 

The functional analysis was written for a HPC environment, so module calls etc. are environment-specific. The input are BIDS-formatted fmriPrep files. 

Software requirements include fmriPrep (for input processing), AFNI (24.3.08) for ACF estimation and cluster correction and Freesurfer.

run.py is the wrapper script calling part1.py, part2.py and part3.py which can be run separately and take care of 1) subject/session discovery + first-level GLM 2) ACF + cluster threshold 3) group-level mixed models respectively. Thus the pipeline looks as follows:
```text
BIDS + fMRIPrep
       │
       ▼
  ┌─────────┐
  │  part1  │
  └────┬────┘
       ├─► First-level GLM
       ├─► QC / confound filtering
       ├─► Residuals
       ├─► Contrast / effect maps
       └─► Group mask
               │
               ▼
          ┌─────────┐
          │  part2  │
          └────┬────┘
               ├─► AFNI 3dFWHMx
               └─► AFNI 3dClustSim
                       │
                       ▼
            Cluster-size threshold
                       │
                       ▼
                  ┌─────────┐
                  │  part3  │
                  └────┬────┘
                       ├─► Julia MixedModels
                       ├─► Voxelwise LMM
                       ├─► Beta / z maps
                       └─► Cluster results
```
##### Helper scripts:
clustersfinal.py - prints out the significant clusters based on output files with MNI coordinates etc.

metadata_readout.py - reads out mean intervals between sessions, valid runs etc. from the metadata.tsv file that is created in part1.
