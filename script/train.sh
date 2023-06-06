#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liacz@bc.edu

#SBATCH --job-name=Training_Autoencoders
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --time=6:00:00
#SBATCH --output=output.log

# Load necessary modules or activate virtual environments
module load cuda-dcgm/2.2.9.1 
module load cuda11.2/toolkit/11.2.2 
module load cudnn8.1-cuda11.2/8.1.1.33
# Example:
export PATH="/usr/public/anaconda/2020.07-p3.8/bin/conda:$PATH"
source activate py39_dev
module load anaconda/2020.07-p3.8
which jupyter


cd CVAE_notebook

# Run the Jupyter notebook
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 test_load.ipynb --output AutoEncoders_executed_notebook.ipynb

# Optional: Convert the executed notebook to other formats
# Example:
# jupyter nbconvert --to html executed_notebook.ipynb
