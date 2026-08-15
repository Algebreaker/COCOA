This is a short overview of the scripts that can be used to replicate the results from the paper 'Multisensory integration while learning to read: A longitudinal functional and structural MRI study' (https://doi.org/10.64898/2026.07.24.740626 ).


## Functional Analysis
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

func_plot.py generates the plots of significant clusters

func_behaviour.py runs the LME and regression analyses to look at the relationship of activation and behaviour.

## Structural Analysis
All scripts are found in the 'structural' folder. 

Prerequisites/Tip: Freesurfer should be activated before starting activating the Python or Matlab environment.

The scripts should be run in this order:

- run_single.py runs **recon-all** for a single subject
- recon_base.sh and recon_long.sh are next
- next the following lines should be run with Freesurfer, the prerequisite is having the qdec.table.dat file; it needs to be run once per measure (thickness, curv, volume, area) and hemisphere; here is an example running the right hemisphere with the volume metric:
  ```
  mris_preproc --qdec-long /data/pt_02825/code_freesurfer/qdec_long.table.dat --target fsaverage --hemi rh --meas volume--out rh.vol.mgh

  mri_surf2surf --hemi rh --s fsaverage --sval rh.vol.mgh --tval rh.vol_sm5.mgh --fwhm-trg 5 --cortex  --noreshape```
  
- long_lme.m is next - it's best to run it line-by-line and metrics currently need to be changed manually throughout the script
- finally run cluster_mask.sh

##### Helper scripts:
struct_plot.py plots the significant clusters across all metrics

struct_behaviour.py runs the LME and regression analyses to look at the relationship of morphology and behaviour.
