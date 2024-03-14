#!/bin/sh
#SBATCH -p horence
#SBATCH --time=160:00:00
#SBATCH --mem=340000      # In MB
#SBATCH --job-name=mmseqs2_search     # job name

source /oak/stanford/groups/horence/george/dog/bin/activate

input_fasta=$1
input_database=$2

#### Create an MMseqs2 database using the query FASTA. 
/oak/stanford/groups/horence/MMseqs2/mmseqs/bin/mmseqs createdb "$input_fasta" queryDB

#### Search the query database against the target database. 
/oak/stanford/groups/horence/MMseqs2/mmseqs/bin/mmseqs search -a 1 queryDB "$input_database" resultDB tmp

### Convert the result to a rich, tabular format. 
/oak/stanford/groups/horence/MMseqs2/mmseqs/bin/mmseqs convertalis --format-output "query,target,qheader,theader,evalue,fident,qstart,qend,qlen,tstart,tend,tlen,raw,bits,qseq,tseq,qcov,tcov,mismatch,qframe" queryDB "$input_database" resultDB resultDB.out
