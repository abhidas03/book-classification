#!/bin/bash
#SBATCH --job-name=classification-model
#SBATCH --output=/home/achong1/book-classification/output/output_%j.txt
#SBATCH --partition=unowned
#SBATCH --time=45:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=achong1@swarthmore.edu

cd $HOME/book-classification
srun python3 pretrained.py
