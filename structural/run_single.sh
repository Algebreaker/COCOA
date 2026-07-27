#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=7-00:00
#SBATCH --partition=long

subjects=("[insert ID]")

for subject in "${subjects[@]}"; do
    sbatch recon_command.sh "$subject"
done
