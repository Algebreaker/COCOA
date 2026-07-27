#!/bin/bash
#
#SBATCH --time 23:59:59                 
#SBATCH -c 4                            # Request 2 CPU cores
#SBATCH --mem 64G                       # Request 10 GB of memory

# Log output and error files.
#SBATCH --output=/data/pt_02825/MPCDF/log/job_output_%j.log       # Standard output log file (using %j for job ID).
#SBATCH --error=/data/pt_02825/MPCDF/log/job_error_%j.log         # Standard error log file (using %j for job ID).

export SUBJECTS_DIR=/data/pt_02825/MPCDF/freesurfer
mkdir -p "$SUBJECTS_DIR"

# Capture the dataset name from the command line argument
# Input subject ID
subject=$1 

# Run recon-all for each session
for ses in "01" "02" "03" "04"; do

    echo "Running recon-all for $freesurfer_id"
    bids_t1w="/data/pt_02825/MPCDF/BIDS/sub-${subject}/ses-${ses}/anat/sub-${subject}_ses-${ses}_run-2_T1w.nii.gz"
    freesurfer_id="sub-${subject}_ses-${ses}"
    FREESURFER recon-all -all -subjid $freesurfer_id -i $bids_t1w

done



