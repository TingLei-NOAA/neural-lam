#!/bin/bash -l
  
#SBATCH -A fv3-cam
#SBATCH -J fge-create_parameter 
#SBATCH -o debo.log
#SBATCH -e debe.log
#SBATCH --qos=gpuwf
#SBATCH --partition=fge
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=0

__conda_setup="$('/scratch2/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/scratch2/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/etc/profile.d/conda.sh" ]; then
        . "/scratch2/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/scratch2/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate test-env3

srun -l  -n 4 python -m neural_lam.debug --data_config data_config_3km.yaml --distributed=1  >& deb.out 

conda deactivate

exit

