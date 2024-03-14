#!/bin/sh
#SBATCH -p horence
#SBATCH --time=40:00:00
#SBATCH --mem=100000      # In MB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=orfipy     # job name

/oak/stanford/groups/horence/george/splash_utils/orfipy --partial-5 --partial-3 --procs 40 --dna DNA_out.fasta --pep peptide_out.fasta --min 36 $1
