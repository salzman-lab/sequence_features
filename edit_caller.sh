#!/bin/sh
#SBATCH -p horence
#SBATCH --time=100:00:00
#SBATCH --mem=268000      # In MB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --job-name=edit_calling     # job name

## Argument 1: Input file path. 
## Argument 2: Desired output file path. 
## Argument 3: Output file path, with results outer merged onto input. 

inputDt=$1
outputDt=$2
inputWithOutputDt=$3

source /oak/stanford/groups/horence/george/dog/bin/activate

python3 /oak/stanford/groups/horence/george/splash_utils/edit_caller.py ${inputDt} ${outputDt} ${inputWithOutputDt}