#!/bin/bash
#
#SBATCH --time 23:59:59                 
#SBATCH -c 4                            # Request 2 CPU cores
#SBATCH --mem 64G                       # Request 10 GB of memory

# Log output and error files.
#SBATCH --output=/data/pt_02825/MPCDF/log/job_output_%j.log       # Standard output log file (using %j for job ID).
#SBATCH --error=/data/pt_02825/MPCDF/log/job_error_%j.log         # Standard error log file (using %j for job ID).

# Capture the dataset name from the command line argument
# Input subject ID
subject=$1 

# Run recon-all for each session
for ses in "01" "02" "03" "04"; do

    base="sub-${subject}"
    freesurfer_id="sub-${subject}_ses-${ses}"
    
    echo "Running recon-all -long for $freesurfer_id"
    
    FREESURFER recon-all -long $freesurfer_id $base -all

done
