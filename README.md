# sequence_features
A repository to store scripts identifying features such as repeats and editing in nucleotide sequences.

### edit_caller.py
A script to identify anchors for which all targets 2...n are differentiated from target 1 via a single kind of single-nucleotide variation, i.e. A to G mutations. <br><br> Usage: sbatch edit_caller.sh ${```inputDt```} ${```intermediateOutDt```} ${```outDt```} <br> 1. ```inputDt``` - a file having columns ```anchor```, ```target``` or ```extendor```, ```dataset```, ```anchor_count```, ```target_count```. <br> 2. ```intermediateOutdt``` - intermediate output reporting, for anchor, dataset pairs having edit calls: 
```anchor```, ```dataset```, ```target```, edited base (```edited_from```), resulting base (```edited_to```), the # edited positions for each target (```num_edits```). <br> 3. ```outDt``` - final output, reporting fields in ```intermediateOutdt```, but left merged onto the input.

### generic_repeat.py
A script to identify nonoverlapping repeats occurring in nucleotide sequences. The largest and left-most k-mer is reported such that, without overlapping, the k-mer appears at least once elsewhere in the compactor sequence with Hamming distance <= 2 to its left-most occurrence. If there are multiple such k-mers, those having the lowest Hamming distance and furthest to the left are selected and their sequence coordinates 'blocked out' until no further k-mers with Hamming distance <= 2 can be added to the set we report on the basis of introducing overlaps or exceeding the Hamming distance threshold.   <br><br> Usage: sbatch generic_repeat.sh ${```inputDt```} ${```outputDt```} ${```sequence_column_name```} <br> 1. ```inputDt``` - A file having at minimum the column ```sequence_column_name```. <br> 2. ```outputDt``` - ```inputDt```, modified to have the following columns: <br>a. ```generic_repeat``` the 'left-most' repeat sequence; b. <br>```generic_repeat_size``` the repeat's k-mer size;<br> c. ```generic_repeat_positions``` the 0-indexed sequence coordinates at which the repeat appears;<br> d. ```generic_repeat_Hamming_distances``` the Hamming distances of the repeats whose positions are reported in ```generic_repeat_positions```; <br>e. ```generic_repeat_mean_Hamming_distance``` the mean of the reported Hamming distances. Note that Hamming distances are not reported circularly, or in comparing the left-most repeat occurrence to itself; <br> f.```generic_repeat_Xmer_entropy``` for X in [3,4,5], the Shannon entropy of the distribution of kmer counts when tiled from the repeat.


### periodic_repeat.py
A script to identify nonoverlapping repeats occurring at a fixed interval in nucleotide sequences. The largest and left-most k-mer is reported such that, at a fixed interval, the k-mer occurs downstream in the sequence at least 3 times contiguously (meaning repeat-interval-repeat-interval-repeat) where the repeat's occurrences are Hamming distance <= 2 to its left-most occurrence. The search is performed such that the k-mer size and interval are maximized. 
<br><br> Usage: sbatch periodic_repeat.sh Usage: sbatch generic_repeat.sh ${```inputDt```} ${```outputDt```} ${```sequence_column_name```} <br> 1. ```inputDt``` - A file having at minimum the column ```sequence_column_name```. <br> 2. ```outputDt``` - ```inputDt```, modified to have the following columns: <br>a. ```periodic_repeat``` the 'left-most' repeat sequence; b. <br>```repeat period``` the interval between which repeats appear;<br> c. ```periodic_repeat_positions``` the 0-indexed sequence coordinates at which the repeat appears;<br> d. ```periodic_repeat_Hamming_distances``` the Hamming distances of the repeats whose positions are reported in ```periodic_repeat_positions```; <br>e. ```periodic_repeat_mean_Hamming_distance``` the mean of the reported Hamming distances. Note that Hamming distances are not reported circularly, or in comparing the left-most repeat occurrence to itself; <br> f.```periodic_repeat_Xmer_entropy``` for X in [3,4,5], the Shannon entropy of the distribution of kmer counts when tiled from the repeat; <g> ```periodic_repeat_length``` the length of ```periodic_repeat```.

