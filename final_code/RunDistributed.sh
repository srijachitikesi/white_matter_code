#!/bin/bash

#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=300g
#SBATCH --gres=gpu:A40:1
#SBATCH -p qTRDGPU
#SBATCH -J ConvLr00001
#SBATCH -e error%A.err
#SBATCH -o gpu4_%A.txt
#SBATCH -A trends396s109
#SBATCH -t 1-10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijachitikesi01@gmail.com
#SBATCH --oversubscribe
#SBATCH --export=NONE

/home/users/schitikesi1/miniconda3/bin/python3 crossval_pred_test.py