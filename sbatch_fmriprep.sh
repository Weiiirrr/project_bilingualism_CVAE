#!/bin/bash
#SBATCH --mail-type=ALL 			
#SBATCH --mail-user=xxx@bc.edu
#SBATCH --job-name=job_fmriprep_ds001796_1

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=128gb
#SBATCH --array=0-10

module load singularity

#this_dir=$(pwd);echo $this_dir
cd $HOME

data_dir='Data/ds001796-download'

sing_image=fmriprep_21.0.1.sif

cd $data_dir
files=(213 214 215 216 217 218 219 220 221 222 223)
#cd $this_dir

cd $HOME

output_dir='Data/output'

fs_lic='Data/license.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub

echo ${files[$SLURM_ARRAY_TASK_ID]}

singularity run --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --ignore slicetiming