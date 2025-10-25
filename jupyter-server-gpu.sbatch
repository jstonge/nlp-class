#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=jupyter
#SBATCH --mail-type=BEGIN,END,FAIL
# You may choose from BEGIN, END, FAIL to receive mail, or NONE to skip mail entirely.
#SBATCH --mail-user=yourNetID@uvm.edu

source ~/.bash_profile
source ~/.bashrc

conda activate llama_setup
module load gcc/13.3.0

# Print a hostname for your starter script
echo "Hostname: $(hostname -s)"

# Launch Jupyter
time jupyter notebook --no-browser --port=8900
