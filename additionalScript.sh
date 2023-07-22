#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --output=/home/achong1/book-classification/output/metrics_%j.txt
#SBATCH --partition=unowned
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=achong1@swarthmore.edu

cd $HOME/book-classification
srun python3 metrics.py
