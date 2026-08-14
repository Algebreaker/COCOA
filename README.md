clustersfinal.py - prints out the significant clusters based on output files with MNI coordinates etc.
metadata_readout.py - reads out mena intervals between sessions, valid runs etc. from metadata.tsv


Functional Analysis:

BIDS + fMRIPrep
       │
       ▼
   ┌────────┐
   │ part1  │
   └────────┘
       │
       ├── First-level GLM
       ├── QC/confound filtering
       ├── residuals
       ├── contrast/effect maps
       └── group mask
              │
              ▼
          ┌────────┐
          │ part2  │
          └────────┘
              │
              ├── AFNI 3dFWHMx
              └── AFNI 3dClustSim
                       │
                       ▼
              cluster-size threshold
                       │
                       ▼
                  ┌────────┐
                  │ part3  │
                  └────────┘
                       │
                       ├── Julia MixedModels
                       ├── voxelwise LMM
                       ├── beta/z maps
                       └── cluster results
run.py is the wrapper script calling part1.py, part2.py and part3.py which can be run separately and take care of 1) subject/session discovery + first-level GLM 2) ACF + cluster threshold 3) group-level mixed models respectively. 
