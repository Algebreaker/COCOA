#!/bin/bash
#
#SBATCH --time 23:59:59                 
#SBATCH -c 4                            # Request 4 CPU cores
#SBATCH --mem 64G                       # Request 64 GB of memory

# Log output and error files.
#SBATCH --output=/data/pt_02825/MPCDF/log/job_output_%j.log       # Standard output log file (using %j for job ID).
#SBATCH --error=/data/pt_02825/MPCDF/log/job_error_%j.log         # Standard error log file (using %j for job ID).

# Capture the dataset name from the command line argument
# Input subject ID
subject=$1 

# Run recon-all base to create within subject template

echo "Running recon-all base for $freesurfer_id"
freesurfer_id="sub-${subject}"
FREESURFER recon-all -base $freesurfer_id -tp "${freesurfer_id}_ses-01" -tp "${freesurfer_id}_ses-02" -tp "${freesurfer_id}_ses-03" -tp "${freesurfer_id}_ses-04" -all
