import numpy as np
import pandas as pd
import os 
import glob
import multiprocessing as mp
import sys

def edit_finder(inputdt,target_fraction_threshold=0.05):

    num_edits = [0]
    
    tl = np.array([np.array(list(i)) for i in inputdt['target'].tolist()])
    tfl = np.array(inputdt['target_fraction'])
    
    tl = tl[tfl > target_fraction_threshold]
    tl_out = np.array(inputdt['target'])[tfl > target_fraction_threshold]
    
    anchor, dataset = inputdt['anchor'].unique()[0], inputdt['dataset'].unique()[0]
    
    if len(tl) < 3: 
        return 0
    
    base = tl[0]

    tracker = []

    boolarr = tl[1] != base
    num_edits.append(np.sum(boolarr))
    base_bp = np.unique(base[boolarr])
    test_bp = np.unique(tl[1][boolarr])

    if len(base_bp) > 1 or len(test_bp) > 1:
        return 0

    tracker = [base_bp,test_bp]

    for i in tl[2:]:

        boolarr = i != base
        num_edits.append(np.sum(boolarr))
        base_bp = np.unique(base[boolarr])
        test_bp = np.unique(i[boolarr])

        if len(base_bp) > 1 or len(test_bp) > 1:
            return 0

        if base_bp[0] != tracker[0] or test_bp[0] != tracker[1]:
            return 0

    outputDt = pd.DataFrame({'target':tl_out,'num_edits':num_edits})
    outputDt['anchor'] = anchor
    outputDt['dataset'] = dataset
    outputDt['edited_from'] = base_bp[0]
    outputDt['edited_to'] = test_bp[0]
    
    return outputDt

def applyParallel(dfGrouped, func):
    
    with mp.Pool(int(os.environ['SLURM_JOB_CPUS_PER_NODE'])) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
        
    return ret_list

def main(): 
    
    a = pd.read_csv(sys.argv[1],sep='\t')
    if 'target' not in a.columns and 'extendor' in a.columns:
        a['target'] = a['extendor'].str[27:]
    a['target_fraction'] = a['target_count'] / a['anchor_count']
    gpby = a.groupby(['anchor','dataset'])[['target','dataset','anchor','target_fraction']]
    outs = applyParallel(gpby, edit_finder)
    b = [i for i in outs if type(i) != int]
    b = pd.concat(b).reset_index(drop=True)
    b.to_csv(sys.argv[2],sep='\t',index=None)
    a.merge(b,how='left').to_csv(sys.argv[3],sep='\t',index=None)

    return 

main()