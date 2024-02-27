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

def find_largest_k(s, K):

    _K = sorted(K, reverse=True)
    
    for k in _K:
        
        for i in range(len(s) - k):
            
            positions, hammings = [], []

            window_i = s[i:i+k]

            for j in range(i+k, len(s) - k + 1):

                window_j = s[j:j+k]

                ham = hamming_distance(window_i, window_j)

                positions.append(j)
                hammings.append(ham)

            boolarr = np.array(hammings) < 3

            take_pos = np.array(positions)[boolarr]
            take_ham = np.array(hammings)[boolarr]

            if not len(take_pos):
                
                continue

            take_df = pd.DataFrame(take_ham,take_pos).sort_values(by=0)

            that = set(range(i,i+k))

            report_pos, report_ham = [i], []

            for index in take_df.index: 

                this = set(range(index, index + k))

                if not len(this & that):

                    that = this | that 

                    report_pos.append(index)

                    report_ham.append(take_df[0][index])
                    
            return window_i, report_ham, report_pos, np.mean(report_ham), k
    return np.nan, np.nan, np.nan, np.nan, np.nan
                

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


data = pd.read_csv(sys.argv[1], sep='\t')
K = [33, 30, 27, 24, 21, 18, 15]


inputiter = []
for i in range(len(data)):
    inputiter.append((data[sys.argv[3]][i],K))
    
workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) 

if __name__ == "__main__":
    with mp.Pool(workers) as p:
        outs = p.starmap(find_largest_k, inputiter)
        
outs = pd.DataFrame(outs)

data['generic_repeat'] = outs[0]
data['generic_repeat_size'] = outs[4]
data['generic_repeat_positions'] = outs[2]
data['generic_repeat_Hamming_distances'] = outs[1]
data['generic_repeat_mean_Hamming_distance'] = outs[3]


trois_generique = dict()
for i in data['generic_repeat'].unique():
    if i not in trois_generique.keys():
        trois_generique[i] = sequence_entropy(i,3)

catre_generique = dict()
for i in data['generic_repeat'].unique():
    if i not in catre_generique.keys():
        catre_generique[i] = sequence_entropy(i,4)

cinque_generique = dict()
for i in data['generic_repeat'].unique():
    if i not in cinque_generique.keys():
        cinque_generique[i] = sequence_entropy(i,5)
        
    
data['generic_repeat_3mer_entropy'] = [trois_generique[i] for i in data['generic_repeat']]
data['generic_repeat_4mer_entropy'] = [catre_generique[i] for i in data['generic_repeat']]
data['generic_repeat_5mer_entropy'] = [cinque_generique[i] for i in data['generic_repeat']]

data.to_csv(sys.argv[2],sep='\t',index=None)