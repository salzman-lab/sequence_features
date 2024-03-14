#!/bin/sh
#SBATCH -p horence
#SBATCH --time=16:00:00
#SBATCH --mem=120000      # In MB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=mmseqs2     # job name

/oak/stanford/groups/horence/MMseqs2/mmseqs/bin/mmseqs createdb $1 DB_$2 --dbtype 1

/oak/stanford/groups/horence/MMseqs2/mmseqs/bin/mmseqs cluster DB_$2 DB_clu_$2 $2_tmp -c $3 --cov-mode 0

/oak/stanford/groups/horence/MMseqs2/mmseqs/bin/mmseqs createtsv DB_$2 DB_$2 DB_clu_$2 DB_clu_$2.tsv