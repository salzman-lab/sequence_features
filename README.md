# sequence_features
A repository to store scripts identifying features such as repeats and editing in nucleotide sequences.

### edit_caller.py
A function to identify anchors for which all targets 2...n are differentiated from target 1 via a single kind of single-nucleotide variation, i.e. A to G mutations. <br><br> Usage: sbatch edit_caller.sh ${```inputDt```} ${```intermediateOutDt```} ${```outDt```} <br> 1. ```inputDt``` - a file having columns ```anchor```, ```target``` or ```extendor```, ```dataset```, ```anchor_count```, ```target_count```. <br> 2. ```intermediateOutdt``` - intermediate output reporting, for anchor, dataset pairs having edit calls: 
```anchor```, ```dataset```, ```target```, edited base (```edited_from```), resulting base (```edited_to```), the # edited positions for each target (```num_edits```). <br> 3. ```outDt``` - final output, reporting fields in ```intermediateOutdt```, but left merged onto the input.



### generic_repeat.py

### periodic_repeat.py
