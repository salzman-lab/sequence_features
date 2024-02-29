import numpy as np
import pandas as pd
import os
from itertools import product
import multiprocessing as mp
import sys

def hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def generate_word_combinations(elements, length):
    return [''.join(combination) for combination in product(elements, repeat=length)]

def repeating_subsequence(s, K):

    ## Sort candidate k in descending order. 
    _K = sorted(K, reverse=True)
    for k in _K:
        
        ## The start position must be >= sequence length - 3k. 
        ## We will iterate through start positions. 
        for start in range(0,len(s)-(3*k)+1):
            
            ## Get the list of valid intervals.
            ## This is found via 3k + 2i <= s.
            max_j = ((len(s)-start) - (3*k)) // 2
            
            ## Begin with the maximum possible interval size given the start position and kmer size.
            for j in reversed(range(0, max_j + 1)):
                
                ## Extract the sequence to test. 
                us = s[start:k+start]
                
                ## Extract the remaining sequences to test. 
                remaining_s = s[start+k+j:]
                
                ## Get the remaining sequences that are of the appropriate length. 
                them = [remaining_s[(j+k)*i:(j+k)*i+k] for i in range(0, ((len(remaining_s)+k+j)//(2*k+j))+1)]
                
                ## Compute the Hamming distance between the tested and remaining kmers. 
                ham_dists = [hamming_distance(they,us) for they in them]
                hams = ''.join([str(1*(i<3)) for i in ham_dists])

                ## Check for at least 3 contiguous passes in the sequence.
                if '111' in hams or hams[:2]=='11':
                    
                    ## Report: the repeat, the period, 
                    ## the 0-indexed coordinates of the repeat in the sequence,
                    ## the Hamming distances, the average Hamming distance. 
                    return us, j, ham_dists, [start]+[start+k+j + ((j+k)*i) for i in range(0, ((len(remaining_s)+k+j)//(2*k+j))+1)], np.mean(ham_dists), len(us)
                
    return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan


def sequence_entropy(string,k):
    
    ## Get the sequence length.
    leng = len(string)
    
    ## If the subsequence size we want to investigate is greater than the sequence length, return NaN. 
    if k > leng: 
        return np.nan
    
    ## If the subsequence size = sequence size, we know the entropy will be 0.
    elif k == leng: 
        return 0 
    
    ## Otherwise, 
    else: 
        dicti = dict()
        
        ## Pass through the sequence one position at a time, incrementing a dictionary with its # of occurrences. 
        for kmer in range(len(string) - k + 1): 
            stri = string[kmer:kmer+k]
            if stri in dicti.keys():
                dicti[stri] += 1
            else: 
                dicti[stri] = 1
                
    ## Use the values of the dictionary (each kmer's count) to compute Shannon entropy. 
    p_vect = np.array(list(dicti.values())) / np.sum(np.array(list(dicti.values())))
    return -(np.dot(p_vect, np.log10(p_vect)))


data = pd.read_csv(sys.argv[1],sep='\t')

K = [33, 30, 27, 24, 21, 18, 15]

inputiter = []
for i in range(len(data)):
    inputiter.append((data[sys.argv[3]][i],K))
    
workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) 

if __name__ == "__main__":
    with mp.Pool(workers) as p:
        outs = p.starmap(repeating_subsequence, inputiter)

outs = pd.DataFrame(outs)

data['periodic_repeat'] = outs[0]
data['repeat_period'] = outs[1]
data['periodic_repeat_Hamming_distances'] = outs[2]
data['periodic_repeat_positions'] = outs[3]
data['periodic_repeat_mean_Hamming_distance'] = outs[4]
data['periodic_repeat_length'] = outs[5]
        
data['periodic_repeat'] = data['periodic_repeat'].fillna('NONE')
unique_pr = data['periodic_repeat'].unique()

trois_periodique = dict()
trois_periodique['NONE'] = 0
for i in unique_pr:
    if i not in trois_periodique.keys():
        trois_periodique[i] = sequence_entropy(i,3)
        
catre_periodique = dict()
catre_periodique['NONE'] = 0
for i in unique_pr:
    if i not in catre_periodique.keys():
        catre_periodique[i] = sequence_entropy(i,4)
        
cinque_periodique = dict()
cinque_periodique['NONE'] = 0
for i in unique_pr:
    if i not in cinque_periodique.keys():
        cinque_periodique[i] = sequence_entropy(i,5)
    
data['periodic_repeat_3mer_entropy'] = [trois_periodique[i] for i in data['periodic_repeat']]
data['periodic_repeat_4mer_entropy'] = [catre_periodique[i] for i in data['periodic_repeat']]
data['periodic_repeat_5mer_entropy'] = [cinque_periodique[i] for i in data['periodic_repeat']]

data.to_csv(sys.argv[2],sep='\t',index=None)
