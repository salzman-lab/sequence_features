# sequence_features
A repository to store scripts identifying features such as repeats and editing in nucleotide sequences.

### edit_caller.py
A script to identify anchors for which all targets 2...n are differentiated from target 1 via a single kind of single-nucleotide variation, i.e. A to G mutations. <br><br> Usage: sbatch edit_caller.sh ${```inputDt```} ${```intermediateOutDt```} ${```outDt```} <br> 1. ```inputDt``` - a file having columns ```anchor```, ```target``` or ```extendor```, ```dataset```, ```anchor_count```, ```target_count```. <br> 2. ```intermediateOutdt``` - intermediate output reporting, for anchor, dataset pairs having edit calls: 
```anchor```, ```dataset```, ```target```, edited base (```edited_from```), resulting base (```edited_to```), the # edited positions for each target (```num_edits```). <br> 3. ```outDt``` - final output, reporting fields in ```intermediateOutdt```, but left merged onto the input.

### generic_repeat.py
A script to identify nonoverlapping repeats occurring in compactor sequences. The largest and left-most k-mer is reported such that, without overlapping, the k-mer appears at least once elsewhere in the compactor sequence with Hamming distance <= 2 to its left-most occurrence. If there are multiple such k-mers, those having the lowest Hamming distance and furthest to the left are selected and their sequence coordinates 'blocked out' until no further k-mers with Hamming distance <= 2 can be added to the set we report on the basis of introducing overlaps or exceeding the Hamming distance threshold.   <br><br> 
### periodic_repeat.py
