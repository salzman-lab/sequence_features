#!/bin/sh
#SBATCH -p horence
#SBATCH --time=100:00:00
#SBATCH --mem=336000      # In MB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --job-name=periodic     # job name

## Argument 1: Input file path. 
## Argument 2: Desired output file path. 
## Argument 3: Column name on which to perform search (ex. 'compactor'). 

source /oak/stanford/groups/horence/george/dog/bin/activate

python3 /oak/stanford/groups/horence/george/splash_utils/periodic_repeat.py $1 $2 $3