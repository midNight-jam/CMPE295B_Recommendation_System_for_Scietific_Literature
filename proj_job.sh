#!/bin/bash
#
#SBATCH --job-name=orion
SBATCH -o  /data/cmpe295-Eirinaki/Rec_Sys_Sci_Litr/scripts/logs.txt
SBATCH -e /data/cmpe295-Eirinaki/Rec_Sys_Sci_Litr/scripts/errors.txt
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu

#SBATCH --gres=gpu:1
SBATCH --time=24:00:00

SBATCH --mail-type=ALL
SBATCH --mail-user=jayam.malviya@sjsu.edu


# scripts path :::: /data/cmpe295-Eirinaki/Rec_Sys_Sci_Litr/scripts

python /data/cmpe295-Eirinaki/Rec_Sys_Sci_Litr/scripts/sample_svd_tensorflow.py