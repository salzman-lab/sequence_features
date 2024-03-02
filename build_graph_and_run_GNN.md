```python
import numpy as np
import pandas as pd
import os
import glob
import networkx
from nltk import edit_distance
from itertools import combinations
```


```python
satcs = glob.glob('*_2/*')
satc_ = pd.read_csv('tabula_sapiens_10x_unpacked_2/TSP_10x_gap_0_unpublished_SPLIT_TSP21_SPLIT_BoneMarrowbin88.satc_unpacked',sep='\t',header=None)
satc_ = satc_[satc_[4]>1]
satc = satc_[satc_[2]=='AGAAGAACTAATGTTAGTATAAGTAAC']
satc_.shape, satc_[2].value_counts().reset_index()
```




    ((220244, 5),
                                 index     2
     0     AGAAGAACTAATGTTAGTATAAGTAAC  4578
     1     GAGCCTGGTGATAGCTGGTTGTCCAAG  4219
     2     CGCATAAGCCTGCGTCAGATTAAAACA  4179
     3     CTCCTCACACCCAATTGGACCAATCTA  3829
     4     ATTGGACCAATCTATCACCCTATAGAA  3807
     ...                           ...   ...
     1321  TGGAATTCTTTGTCTTTGACTTTTGAC     2
     1322  AGCGTGGCCGTTGGCTGCCTCGCACAG     2
     1323  CCGACGGCACCTACGGCTCAACATTTT     2
     1324  GCTCATATGCGAGCGCTAATTCTGTGG     2
     1325  CTCAGGGAGTGCATCCGCCCCAACCCT     2
     
     [1326 rows x 2 columns])




```python
satc = satc.rename(columns={1:'sample',2:'anchor',3:'target',4:'count'})
satc = satc[['sample','anchor','target','count']]
```

### Define configurations for a target-centered graph. 


graphName : (anchor, class)

nodeType : "target" 

nodeFeatures : [ "seqComp" , "sampleFraction" ]

edgeFeatures : [ "targetHamming" , "targetLevenshtein" , "corrSampleFractions" , 
                 "corrBoolSampleFractions" , "naiveMSA" ]

connectedness: [ "full" , "sampleCorrelated" , "toleratedHamming_"+{float} , "toleratedLevenshtein_"+{float} ] 



### Define configurations for a sample-centered graph. 

graphName : (anchor, class)

nodeType : "sample" 

nodeFeatures : [ "maskedSeqComp" , "sampleFraction" ]

edgeFeatures : [ "fullTargetHamming" , "fullTargetLevenshtein" ,"aggTargetHamming" , "aggTargetLevenshtein" ,
                 "corrSampleFractions" ,  "corrBoolSampleFractions" , "naiveMSA" ]

connectedness: [ "full" , "sampleCorrelated" , "toleratedMeanHamming_"+{float} , "toleratedMeanLevenshtein_"+{float} ] 
    


```python
def node_featurization(data, nodeType, nodeFeatures, ordering):
    
    nodeFeatureDict = dict()
    
    if 'seqComp' in nodeFeatures:
        
        nodeFeatureDict['seqComp'] = dict()
        
        for target in data['target'].unique():
        
            seq = np.array(list(target))
            mat = np.zeros(shape=(4,len(seq)))
            mat[0] = np.array([1*(seq == 'A')])
            mat[1] = np.array([1*(seq == 'C')])
            mat[2] = np.array([1*(seq == 'G')])
            mat[3] = np.array([1*(seq == 'T')])
            nodeFeatureDict['seqComp'][target] = mat
            
    if 'sampleFraction' in nodeFeatures:
    
        nodeFeatureDict['sampleFraction'] = dict()
        
        if nodeType == 'sample':
            
            full = pd.DataFrame({'target':ordering})
            
            for sample in data['sample'].unique():
                
                this_full = full.merge(data[data['sample']==sample][['sample','target','count']],how='left').fillna(0)
                nodeFeatureDict['sampleFraction'][sample] = np.array(this_full['count'] / this_full['count'].sum())
        
        if nodeType == 'target': 
          
            full = pd.DataFrame({'sample':ordering})
            
            for target in data['target'].unique():
                
                this_full = full.merge(data[data['target']==target][['sample','target','count']],how='left').fillna(0)
                nodeFeatureDict['sampleFraction'][target] = np.array(this_full['count'] / this_full['count'].sum())
             

    if 'maskedSeqComp' in nodeFeatures:
     
        if 'seqComp' in nodeFeatures:
            internalFeatureDict = nodeFeatureDict
        
        else:
            internalFeatureDict = dict()
        
        for target in data['target'].unique():
         
            seq = np.array(list(target))
            mat = np.zeros(shape=(4,len(seq)))
            mat[0] = np.array([1*(seq == 'A')])
            mat[1] = np.array([1*(seq == 'C')])
            mat[2] = np.array([1*(seq == 'G')])
            mat[3] = np.array([1*(seq == 'T')])
            internalFeatureDict['seqComp'][target] = mat
        
        for sample in data['sample'].unique():
               
            nodeFeatureDict['maskedSeqComp'][sample] = np.zeros(shape=(len(ordering),4,len(ordering[0])))
            sample_targets = set(data[data['sample']==sample]['target'].unique())
            ordering_set = set(ordering)
            
            for index in range(len(ordering)):
                
                verbose = ordering[index]
                
                if verbose in sample_targets:
                    
                    nodeFeatureDict['maskedSeqComp'][sample][index] = internalFeatureDict['seqComp'][verbose]
        
    return nodeFeatureDict
```


```python
def pairwise_similarities(values, function):
    
    distanceMatrix = np.zeros(shape=(len(values),len(values)))

    for i in range(len(values)): 

        for j in range(len(values)):

            distanceMatrix[i][j] = function(values[i], values[j])
    
    return distanceMatrix


def hamming_distance(seq1, seq2):
    
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def correlation_matrix(vectorDict, keys, isBoolMasked):
    
    correlationMatrix = np.zeros(shape=(len(keys),len(keys)))
    setList = []

    if not isBoolMasked:
    
        for i in range(len(keys)): 

            for j in range(len(keys)): 
            
                corr = np.dot(vectorDict[keys[i]], vectorDict[keys[j]])
                correlationMatrix[i][j] = corr
                
                if corr > 0: 
                    
                    s = set([keys[i],keys[j]])
                    
                    if len(s) > 1 and s not in setList:
                        
                        setList.append(s)
        
        return correlationMatrix, [tuple(i) for i in setList]
    
    if isBoolMasked is True: 
        
        for i in range(len(keys)): 
            
            for j in range(len(keys)): 
                
                correlationMatrix[i][j] = np.dot(1*(vectorDict[keys[i]]>0), 1*(vectorDict[keys[j]]>0))

        return correlationMatrix
```


```python
def edgeFeatures(data, nodeType, edges, edgeFeatureList, nodeOrdering, ordering, hamDistanceMatrix, levDistanceMatrix, correlationMatrix, boolCorrelationMatrix):    
    
    edgeFeatureDict = dict()
        
    for edge in edges: 

        i, j = np.nonzero(nodeOrdering==edge[0])[0][0], np.nonzero(nodeOrdering==edge[1])[0][0]
        edgeFeatureDict[edge] = dict()

        if 'corrBoolSampleFractions' in edgeFeatureList:
            
            edgeFeatureDict[edge]['corrBoolSampleFractions'] = boolCorrelationMatrix[i][j]
        
        if 'corrSampleFractions' in edgeFeatureList:
            
            edgeFeatureDict[edge]['corrSampleFractions'] = correlationMatrix[i][j]
            
        if 'targetHamming' in edgeFeatureList:
            
            edgeFeatureDict[edge]['targetHamming'] = hamDistanceMatrix[i][j]
            
        if 'targetLevenshtein' in edgeFeatureList:
            
            edgeFeatureDict[edge]['targetLevenshtein'] = levDistanceMatrix[i][j]
            
    return edgeFeatureDict
```


```python
def construct_graph(data, graphName, nodeType, nodeFeatures, edgeFeatureList, connectedness, ordering):
    
    output_dict = dict()
    
    output_dict['nodeFeatures'] = node_featurization(data, nodeType, nodeFeatures, ordering)
    
    if nodeType == 'target': 
        
        node_value_arr = data['target'].unique()
        
    if nodeType == 'sample': 
        
        node_value_arr = data['sample'].unique()
        
    ####
    ####
    ####
    #### If needed, precompute all pairwise Hamming and/or Levenshtein distances.
        
    hamDistanceMatrix = 0
    
    if 'Hamming' in ''.join(edgeFeatureList) or ''.join(connectedness):
        
        if nodeType == 'target': 
            
            hamDistanceMatrix = pairwise_similarities(node_value_arr, hamming_distance)
        
        if nodeType == 'sample': 
            
            hamDistanceMatrix = pairwise_similarities(ordering, hamming_distance)
            
    levDistanceMatrix = 0
        
    if 'Levenshtein' in ''.join(edgeFeatureList) or ''.join(connectedness):
        
        if nodeType == 'target': 
            
            levDistanceMatrix = pairwise_similarities(node_value_arr, edit_distance)
            
        if nodeType == 'sample': 
            
            levDistanceMatrix = pairwise_similarities(ordering, edit_distance)
            
    ####
    ####
    ####
    #### Define edge sets prior to constructing edge features. 
    
    if 'full' in connectedness: 
        
        output_dict['edges'] = list(combinations(node_value_arr, 2))
        
    correlationMatrix = 0
    
    if 'sampleCorrelated' in connectedness or 'corr' in ''.join(edgeFeatureList): 
        
        correlationMatrix, output_dict['edges'] = correlation_matrix(output_dict['nodeFeatures']['sampleFraction'], node_value_arr, False)
        
    boolCorrelationMatrix = 0
    
    if 'Bool' in ''.join(edgeFeatureList):
        
        boolCorrelationMatrix = correlation_matrix(output_dict['nodeFeatures']['sampleFraction'], node_value_arr, True)   
        
    output_dict['edgeFeatures'] = edgeFeatures(data, nodeType, output_dict['edges'], edgeFeatureList, node_value_arr, ordering, hamDistanceMatrix, levDistanceMatrix, correlationMatrix, boolCorrelationMatrix)
            
    return output_dict
```


```python
a = construct_graph(select, 'test', 'target', ['seqComp','sampleFraction'], ["targetHamming" , "targetLevenshtein" , "corrSampleFractions" , "corrBoolSampleFractions" ], ['sampleCorrelated'],satc['sample'].unique())

```


```python
satc.to_csv('vibrio_satc/satc.tsv',sep='\t',index=None)
```


```python
import torch
from torch_geometric.data import Data
```


```python
list(v[v['anchor']=='AAAAACCTAGTGAGGCAATAACTCAGA']['class'])[0]
```




    'unclassified'




```python
## Define one sample ordering for all anchors. 
sample_ord = satc['sample'].unique()
store_dict_and_data = dict()

for anchor in satc['anchor'].unique():

    label = list(v[v['anchor']==anchor]['class'])[0]
    select = satc[satc['anchor']==anchor]
    a = construct_graph(select, 'test', 'target', ['seqComp','sampleFraction'], ["targetHamming" , "targetLevenshtein" , "corrSampleFractions" , "corrBoolSampleFractions" ], ['sampleCorrelated'],satc['sample'].unique())

    ## Define one target ordering for this anchor.
    ## The index of each entry defines its node ID. 
    target_ord = np.array(list(a['nodeFeatures']['seqComp'].keys()))

    ## Extract node features. 
    nodeFeatureList = []
    for target in target_ord:
        nodeFeatureList.append(list(np.append(a['nodeFeatures']['seqComp'][target].flatten(),\
        a['nodeFeatures']['sampleFraction'][target])))

    ## Define the tensor corresponding to node features. 
    x = torch.tensor(nodeFeatureList)

    ## Extract edges and edge features. 
    edgeFeatureListOrder = []
    edge_index_top, edge_index_bot = [], []
    edgeFeaturesVector = []
    for edge in a['edges']:
        i, j = np.nonzero(target_ord==edge[0])[0][0], np.nonzero(target_ord==edge[1])[0][0]
        edge_index_top.append(i)
        edge_index_top.append(j)
        edge_index_bot.append(j)
        edge_index_bot.append(i)
        if not len(edgeFeatureListOrder):
            edgeFeatureListOrder = list(a['edgeFeatures'][edge].keys())
        extractedFeatures = []
        for eF in edgeFeatureListOrder:
            extractedFeatures.append(a['edgeFeatures'][edge][eF])
        edgeFeaturesVector.append(extractedFeatures) 
        edgeFeaturesVector.append(extractedFeatures)

    ## Define the tensors corresponding to edges and edge features. 
    edge_attributes = torch.tensor(edgeFeaturesVector)
    edge_index = torch.tensor([edge_index_top,edge_index_bot])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=label)

    store_dict_and_data[anchor] = label, a, data
    
```


```python
len(store_dict_and_data.keys())
```




    3428




```python
## Given the set of anchors, take 80 percent of CRISPR and MGEs into training;  
## take 20% of CRISPR and MGEs into test. 

```


```python
cri = v[v['class']=='CRISPR']
mge = v[v['class']=='MGE']
unc = v[v['class']=='unclassified']
v.groupby(['class'])['anchor'].nunique()
```




    class
    CRISPR            67
    MGE              119
    unclassified    3242
    Name: anchor, dtype: int64



#### Training split; CRISPR, MGE, unclassified.


```python
int((0.8*67)//1), int((0.8*119)//1), int(((67+119)/2)//1)
```




    (53, 95, 93)




```python
53 / (53 + 95 + 93), 95 / (53 + 95 + 93), 93 / (53 + 95 + 93)
```




    (0.21991701244813278, 0.3941908713692946, 0.38589211618257263)



#### Testing split; CRISPR, MGE, unclassified.


```python
(14, 24, 42)
```




    (14, 24, 42)




```python
14 / (14 + 24 + 42 ), 24 / (14 + 24 + 42 ), 42 / (14 + 24 + 42 )

```




    (0.175, 0.3, 0.525)




```python
train_cri_anchs = np.random.choice(cri['anchor'].tolist(),int((0.8*67)//1),replace=False)
train_mge_anchs = np.random.choice(mge['anchor'].tolist(),int((0.8*119)//1),replace=False)
train_unc_anchs = np.random.choice(unc['anchor'].tolist(),int(((67+119)/2)//1),replace=False)



```


```python
train_anchs = []
train_anchs.extend(train_cri_anchs)
train_anchs.extend(train_mge_anchs)
train_anchs.extend(train_unc_anchs)

```


```python
v_positive_anchs = list(v[(~v['anchor'].isin(train_anchs))&(v['class']!='unclassified')]['anchor'])
len(v_positive_anchs), len(train_anchs)
v_negative_anchs = np.random.choice(v[(~v['anchor'].isin(train_anchs))&(v['class']=='unclassified')]['anchor'].tolist(),int(42),replace=False)
test_anchs = []
test_anchs.extend(v_positive_anchs)
test_anchs.extend(v_negative_anchs)

```

<br><br><br><br>

### Beginning with the attentive graph convolutional network! 

https://medium.com/stanford-cs224w/incorporating-edge-features-into-graph-neural-networks-for-country-gdp-predictions-1d4dea68337d

https://discuss.pytorch.org/t/pytorch-geometric-how-make-own-dataset-of-multiple-graphs/125130   
   
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=CN3sRVuaQ88l


```python
torch.manual_seed(163)
from torch_geometric.utils import add_self_loops

for i in train_anchs: 
    add_self_loops(store_dict_and_data[i][2].edge_index, store_dict_and_data[i][2].edge_attr)
    
for i in test_anchs: 
    add_self_loops(store_dict_and_data[i][2].edge_index, store_dict_and_data[i][2].edge_attr)
    
train_dataset = [store_dict_and_data[i][2] for i in train_anchs]
test_dataset = [store_dict_and_data[i][2] for i in test_anchs]

    
```


```python
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

```


```python
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(163)
        self.conv1 = GATv2Conv(train_dataset[0].num_node_features, hidden_channels,edge_dim = train_dataset[0].num_edge_features,add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels,edge_dim = train_dataset[0].num_edge_features,add_self_loops=True)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels,edge_dim = train_dataset[0].num_edge_features,add_self_loops=False)
        self.conv4 = GATv2Conv(hidden_channels, hidden_channels,edge_dim = train_dataset[0].num_edge_features,add_self_loops=False)
        self.conv5 = GATv2Conv(hidden_channels, hidden_channels,edge_dim = train_dataset[0].num_edge_features,add_self_loops=True)

        self.lin = Linear(hidden_channels, 3) # 3 = num_classes
        
    #def forward(self, x, edge_index, edge_attr, batch):
    def forward(self, data, batch):
        x, edge_index, edge_attr = data.x.float(), data.edge_index.long(), data.edge_attr.float()

        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv5(x, edge_index, edge_attr=edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=64)

print(model)
```

    GCN(
      (conv1): GATv2Conv(342, 64, heads=1)
      (conv2): GATv2Conv(64, 64, heads=1)
      (conv3): GATv2Conv(64, 64, heads=1)
      (conv4): GATv2Conv(64, 64, heads=1)
      (conv5): GATv2Conv(64, 64, heads=1)
      (lin): Linear(in_features=64, out_features=3, bias=True)
    )



```python
track = dict()
```


```python
torch.manual_seed(163)
model = GCN(hidden_channels=16)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track[16] = [train_accs, test_accs]
    
```

    Epoch: 001, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 002, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 003, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 004, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 005, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 006, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 007, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 008, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 009, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 010, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 011, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 012, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 013, Train Acc: 0.3983, Test Acc: 0.3000
    Epoch: 014, Train Acc: 0.3983, Test Acc: 0.3000
    Epoch: 015, Train Acc: 0.4025, Test Acc: 0.3375
    Epoch: 016, Train Acc: 0.4274, Test Acc: 0.3375
    Epoch: 017, Train Acc: 0.4398, Test Acc: 0.3250
    Epoch: 018, Train Acc: 0.4606, Test Acc: 0.3500
    Epoch: 019, Train Acc: 0.4772, Test Acc: 0.3625
    Epoch: 020, Train Acc: 0.4772, Test Acc: 0.3750
    Epoch: 021, Train Acc: 0.5021, Test Acc: 0.3750
    Epoch: 022, Train Acc: 0.5228, Test Acc: 0.3875
    Epoch: 023, Train Acc: 0.5311, Test Acc: 0.4375
    Epoch: 024, Train Acc: 0.5519, Test Acc: 0.4625
    Epoch: 025, Train Acc: 0.5519, Test Acc: 0.4500
    Epoch: 026, Train Acc: 0.5685, Test Acc: 0.4500
    Epoch: 027, Train Acc: 0.5768, Test Acc: 0.4500
    Epoch: 028, Train Acc: 0.5768, Test Acc: 0.4750
    Epoch: 029, Train Acc: 0.5685, Test Acc: 0.5000
    Epoch: 030, Train Acc: 0.5726, Test Acc: 0.5125
    Epoch: 031, Train Acc: 0.5726, Test Acc: 0.5125
    Epoch: 032, Train Acc: 0.5809, Test Acc: 0.5125
    Epoch: 033, Train Acc: 0.5851, Test Acc: 0.4875
    Epoch: 034, Train Acc: 0.5809, Test Acc: 0.4875
    Epoch: 035, Train Acc: 0.5768, Test Acc: 0.5000
    Epoch: 036, Train Acc: 0.5726, Test Acc: 0.5250
    Epoch: 037, Train Acc: 0.5768, Test Acc: 0.5375
    Epoch: 038, Train Acc: 0.5851, Test Acc: 0.5500
    Epoch: 039, Train Acc: 0.5934, Test Acc: 0.5625
    Epoch: 040, Train Acc: 0.6017, Test Acc: 0.5625
    Epoch: 041, Train Acc: 0.5975, Test Acc: 0.5625
    Epoch: 042, Train Acc: 0.5851, Test Acc: 0.5500
    Epoch: 043, Train Acc: 0.5809, Test Acc: 0.5625
    Epoch: 044, Train Acc: 0.5975, Test Acc: 0.5625
    Epoch: 045, Train Acc: 0.6017, Test Acc: 0.5625
    Epoch: 046, Train Acc: 0.6100, Test Acc: 0.5625
    Epoch: 047, Train Acc: 0.6100, Test Acc: 0.5750
    Epoch: 048, Train Acc: 0.6349, Test Acc: 0.6000
    Epoch: 049, Train Acc: 0.6515, Test Acc: 0.5875
    Epoch: 050, Train Acc: 0.6556, Test Acc: 0.5875
    Epoch: 051, Train Acc: 0.6763, Test Acc: 0.5875
    Epoch: 052, Train Acc: 0.6888, Test Acc: 0.6125
    Epoch: 053, Train Acc: 0.7178, Test Acc: 0.6125
    Epoch: 054, Train Acc: 0.7261, Test Acc: 0.6125
    Epoch: 055, Train Acc: 0.7220, Test Acc: 0.6250
    Epoch: 056, Train Acc: 0.7261, Test Acc: 0.6250
    Epoch: 057, Train Acc: 0.7510, Test Acc: 0.6250
    Epoch: 058, Train Acc: 0.7593, Test Acc: 0.6250
    Epoch: 059, Train Acc: 0.7718, Test Acc: 0.6250
    Epoch: 060, Train Acc: 0.7842, Test Acc: 0.6000
    Epoch: 061, Train Acc: 0.7884, Test Acc: 0.6000
    Epoch: 062, Train Acc: 0.7925, Test Acc: 0.6250
    Epoch: 063, Train Acc: 0.8050, Test Acc: 0.6250
    Epoch: 064, Train Acc: 0.8050, Test Acc: 0.6125
    Epoch: 065, Train Acc: 0.8340, Test Acc: 0.6000
    Epoch: 066, Train Acc: 0.8382, Test Acc: 0.6125
    Epoch: 067, Train Acc: 0.8465, Test Acc: 0.6125
    Epoch: 068, Train Acc: 0.8382, Test Acc: 0.6125
    Epoch: 069, Train Acc: 0.8299, Test Acc: 0.6000
    Epoch: 070, Train Acc: 0.8340, Test Acc: 0.6000
    Epoch: 071, Train Acc: 0.8340, Test Acc: 0.6000
    Epoch: 072, Train Acc: 0.8465, Test Acc: 0.6000
    Epoch: 073, Train Acc: 0.8465, Test Acc: 0.6125
    Epoch: 074, Train Acc: 0.8589, Test Acc: 0.6125
    Epoch: 075, Train Acc: 0.8506, Test Acc: 0.6250
    Epoch: 076, Train Acc: 0.8548, Test Acc: 0.6250
    Epoch: 077, Train Acc: 0.8589, Test Acc: 0.6375
    Epoch: 078, Train Acc: 0.8631, Test Acc: 0.6125
    Epoch: 079, Train Acc: 0.8631, Test Acc: 0.5875
    Epoch: 080, Train Acc: 0.8714, Test Acc: 0.6125
    Epoch: 081, Train Acc: 0.8672, Test Acc: 0.6000
    Epoch: 082, Train Acc: 0.8797, Test Acc: 0.6000
    Epoch: 083, Train Acc: 0.8797, Test Acc: 0.6000
    Epoch: 084, Train Acc: 0.8921, Test Acc: 0.6000
    Epoch: 085, Train Acc: 0.8963, Test Acc: 0.6125
    Epoch: 086, Train Acc: 0.9004, Test Acc: 0.6000
    Epoch: 087, Train Acc: 0.9004, Test Acc: 0.6000
    Epoch: 088, Train Acc: 0.8963, Test Acc: 0.6000
    Epoch: 089, Train Acc: 0.9004, Test Acc: 0.6000
    Epoch: 090, Train Acc: 0.9046, Test Acc: 0.5875
    Epoch: 091, Train Acc: 0.9087, Test Acc: 0.5875
    Epoch: 092, Train Acc: 0.9046, Test Acc: 0.5875
    Epoch: 093, Train Acc: 0.9129, Test Acc: 0.6000
    Epoch: 094, Train Acc: 0.9129, Test Acc: 0.6000
    Epoch: 095, Train Acc: 0.9170, Test Acc: 0.5875
    Epoch: 096, Train Acc: 0.9253, Test Acc: 0.6000
    Epoch: 097, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 098, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 099, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 100, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 101, Train Acc: 0.9419, Test Acc: 0.5875
    Epoch: 102, Train Acc: 0.9419, Test Acc: 0.5875
    Epoch: 103, Train Acc: 0.9419, Test Acc: 0.5875
    Epoch: 104, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 105, Train Acc: 0.9461, Test Acc: 0.5875
    Epoch: 106, Train Acc: 0.9461, Test Acc: 0.5875
    Epoch: 107, Train Acc: 0.9461, Test Acc: 0.5875
    Epoch: 108, Train Acc: 0.9502, Test Acc: 0.5875
    Epoch: 109, Train Acc: 0.9544, Test Acc: 0.5875
    Epoch: 110, Train Acc: 0.9627, Test Acc: 0.5875
    Epoch: 111, Train Acc: 0.9627, Test Acc: 0.6000
    Epoch: 112, Train Acc: 0.9585, Test Acc: 0.6000
    Epoch: 113, Train Acc: 0.9627, Test Acc: 0.6000
    Epoch: 114, Train Acc: 0.9668, Test Acc: 0.6250
    Epoch: 115, Train Acc: 0.9668, Test Acc: 0.6250
    Epoch: 116, Train Acc: 0.9710, Test Acc: 0.6125
    Epoch: 117, Train Acc: 0.9751, Test Acc: 0.6000
    Epoch: 118, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 119, Train Acc: 0.9710, Test Acc: 0.6125
    Epoch: 120, Train Acc: 0.9793, Test Acc: 0.6250
    Epoch: 121, Train Acc: 0.9834, Test Acc: 0.6125
    Epoch: 122, Train Acc: 0.9793, Test Acc: 0.6375
    Epoch: 123, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 124, Train Acc: 0.9917, Test Acc: 0.6375
    Epoch: 125, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 126, Train Acc: 0.9917, Test Acc: 0.6500
    Epoch: 127, Train Acc: 0.9917, Test Acc: 0.6625
    Epoch: 128, Train Acc: 0.9917, Test Acc: 0.6375
    Epoch: 129, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 130, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 131, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 132, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 133, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 134, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 135, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 136, Train Acc: 0.9959, Test Acc: 0.6500
    Epoch: 137, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 138, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 139, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 140, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 141, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 142, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 143, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 144, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 145, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 146, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 147, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 148, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 149, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 150, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 151, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 152, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 153, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 154, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 155, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 156, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 157, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 158, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 159, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 160, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 161, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 162, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 163, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 164, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 165, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 166, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 167, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 168, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 169, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 170, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 171, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 172, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 173, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 174, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 175, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 176, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 177, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 178, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 179, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 180, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 181, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 182, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 183, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 184, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 185, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 186, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 187, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 188, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 189, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 190, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 191, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 192, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 193, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 194, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 195, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 196, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 197, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 198, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 199, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 200, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 201, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 202, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 203, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 204, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 205, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 206, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 207, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 208, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 209, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 210, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 211, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 212, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 213, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 214, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 215, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 216, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 217, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 218, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 219, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 220, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 221, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 222, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 223, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 224, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 225, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 226, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 227, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 228, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 229, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 230, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 231, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 232, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 233, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 234, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 235, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 236, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 237, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 238, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 239, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 240, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 241, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 242, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 243, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 244, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 245, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 246, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 247, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 248, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 249, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 250, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 251, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 252, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 253, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 254, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 255, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 256, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 257, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 258, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 259, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 260, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 261, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 262, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 263, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 264, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 265, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 266, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 267, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 268, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 269, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 270, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 271, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 272, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 273, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 274, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 275, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 276, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 277, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 278, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 279, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 280, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 281, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 282, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 283, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 284, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 285, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 286, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 287, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 288, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 289, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 290, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 291, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 292, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 293, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 294, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 295, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 296, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 297, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 298, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 299, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 300, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 301, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 302, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 303, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 304, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 305, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 306, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 307, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 308, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 309, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 310, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 311, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 312, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 313, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 314, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 315, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 316, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 317, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 318, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 319, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 320, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 321, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 322, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 323, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 324, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 325, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 326, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 327, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 328, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 329, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 330, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 331, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 332, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 333, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 334, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 335, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 336, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 337, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 338, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 339, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 340, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 341, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 342, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 343, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 344, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 345, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 346, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 347, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 348, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 349, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 350, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 351, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 352, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 353, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 354, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 355, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 356, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 357, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 358, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 359, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 360, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 361, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 362, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 363, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 364, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 365, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 366, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 367, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 368, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 369, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 370, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 371, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 372, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 373, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 374, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 375, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 376, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 377, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 378, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 379, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 380, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 381, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 382, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 383, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 384, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 385, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 386, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 387, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 388, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 389, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 390, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 391, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 392, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 393, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 394, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 395, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 396, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 397, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 398, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 399, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 400, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 401, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 402, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 403, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 404, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 405, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 406, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 407, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 408, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 409, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 410, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 411, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 412, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 413, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 414, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 415, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 416, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 417, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 418, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 419, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 420, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 421, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 422, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 423, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 424, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 425, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 426, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 427, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 428, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 429, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 430, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 431, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 432, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 433, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 434, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 435, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 436, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 437, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 438, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 439, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 440, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 441, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 442, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 443, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 444, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 445, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 446, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 447, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 448, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 449, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 450, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 451, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 452, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 453, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 454, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 455, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 456, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 457, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 458, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 459, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 460, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 461, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 462, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 463, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 464, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 465, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 466, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 467, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 468, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 469, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 470, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 471, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 472, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 473, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 474, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 475, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 476, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 477, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 478, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 479, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 480, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 481, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 482, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 483, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 484, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 485, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 486, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 487, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 488, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 489, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 490, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 491, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 492, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 493, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 494, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 495, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 496, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 497, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 498, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 499, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 500, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 501, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 502, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 503, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 504, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 505, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 506, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 507, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 508, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 509, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 510, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 511, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 512, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 513, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 514, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 515, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 516, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 517, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 518, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 519, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 520, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 521, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 522, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 523, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 524, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 525, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 526, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 527, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 528, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 529, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 530, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 531, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 532, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 533, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 534, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 535, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 536, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 537, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 538, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 539, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 540, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 541, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 542, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 543, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 544, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 545, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 546, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 547, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 548, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 549, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 550, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 551, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 552, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 553, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 554, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 555, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 556, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 557, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 558, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 559, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 560, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 561, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 562, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 563, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 564, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 565, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 566, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 567, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 568, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 569, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 570, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.6750
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6125



```python
torch.manual_seed(163)
model = GCN(hidden_channels=8)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track[8] = [train_accs, test_accs]
```

    Epoch: 001, Train Acc: 0.2739, Test Acc: 0.1875
    Epoch: 002, Train Acc: 0.2905, Test Acc: 0.1875
    Epoch: 003, Train Acc: 0.3112, Test Acc: 0.2000
    Epoch: 004, Train Acc: 0.3195, Test Acc: 0.2000
    Epoch: 005, Train Acc: 0.3610, Test Acc: 0.2500
    Epoch: 006, Train Acc: 0.3942, Test Acc: 0.2875
    Epoch: 007, Train Acc: 0.3817, Test Acc: 0.2750
    Epoch: 008, Train Acc: 0.3859, Test Acc: 0.2875
    Epoch: 009, Train Acc: 0.3900, Test Acc: 0.3000
    Epoch: 010, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 011, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 012, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 013, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 014, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 015, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 016, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 017, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 018, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 019, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 020, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 021, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 022, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 023, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 024, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 025, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 026, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 027, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 028, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 029, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 030, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 031, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 032, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 033, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 034, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 035, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 036, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 037, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 038, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 039, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 040, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 041, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 042, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 043, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 044, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 045, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 046, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 047, Train Acc: 0.3983, Test Acc: 0.3125
    Epoch: 048, Train Acc: 0.4066, Test Acc: 0.3250
    Epoch: 049, Train Acc: 0.4066, Test Acc: 0.3375
    Epoch: 050, Train Acc: 0.4149, Test Acc: 0.3375
    Epoch: 051, Train Acc: 0.4191, Test Acc: 0.3375
    Epoch: 052, Train Acc: 0.4191, Test Acc: 0.3375
    Epoch: 053, Train Acc: 0.4232, Test Acc: 0.3500
    Epoch: 054, Train Acc: 0.4232, Test Acc: 0.3625
    Epoch: 055, Train Acc: 0.4232, Test Acc: 0.3750
    Epoch: 056, Train Acc: 0.4274, Test Acc: 0.3875
    Epoch: 057, Train Acc: 0.4315, Test Acc: 0.4000
    Epoch: 058, Train Acc: 0.4315, Test Acc: 0.4125
    Epoch: 059, Train Acc: 0.4481, Test Acc: 0.4000
    Epoch: 060, Train Acc: 0.4481, Test Acc: 0.3875
    Epoch: 061, Train Acc: 0.4523, Test Acc: 0.4000
    Epoch: 062, Train Acc: 0.4606, Test Acc: 0.4000
    Epoch: 063, Train Acc: 0.4689, Test Acc: 0.4125
    Epoch: 064, Train Acc: 0.4855, Test Acc: 0.4000
    Epoch: 065, Train Acc: 0.4896, Test Acc: 0.4375
    Epoch: 066, Train Acc: 0.4938, Test Acc: 0.4250
    Epoch: 067, Train Acc: 0.5187, Test Acc: 0.4500
    Epoch: 068, Train Acc: 0.5353, Test Acc: 0.4625
    Epoch: 069, Train Acc: 0.5311, Test Acc: 0.4625
    Epoch: 070, Train Acc: 0.5560, Test Acc: 0.4625
    Epoch: 071, Train Acc: 0.5519, Test Acc: 0.4625
    Epoch: 072, Train Acc: 0.5519, Test Acc: 0.4625
    Epoch: 073, Train Acc: 0.5477, Test Acc: 0.4625
    Epoch: 074, Train Acc: 0.5436, Test Acc: 0.4625
    Epoch: 075, Train Acc: 0.5477, Test Acc: 0.4500
    Epoch: 076, Train Acc: 0.5477, Test Acc: 0.4500
    Epoch: 077, Train Acc: 0.5394, Test Acc: 0.4625
    Epoch: 078, Train Acc: 0.5602, Test Acc: 0.5000
    Epoch: 079, Train Acc: 0.5726, Test Acc: 0.5250
    Epoch: 080, Train Acc: 0.5851, Test Acc: 0.5250
    Epoch: 081, Train Acc: 0.5809, Test Acc: 0.5250
    Epoch: 082, Train Acc: 0.5851, Test Acc: 0.5125
    Epoch: 083, Train Acc: 0.5851, Test Acc: 0.5125
    Epoch: 084, Train Acc: 0.5892, Test Acc: 0.5125
    Epoch: 085, Train Acc: 0.5892, Test Acc: 0.5000
    Epoch: 086, Train Acc: 0.5851, Test Acc: 0.5250
    Epoch: 087, Train Acc: 0.5726, Test Acc: 0.5250
    Epoch: 088, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 089, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 090, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 091, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 092, Train Acc: 0.5809, Test Acc: 0.5500
    Epoch: 093, Train Acc: 0.5768, Test Acc: 0.5625
    Epoch: 094, Train Acc: 0.5851, Test Acc: 0.5625
    Epoch: 095, Train Acc: 0.5851, Test Acc: 0.5625
    Epoch: 096, Train Acc: 0.5892, Test Acc: 0.5625
    Epoch: 097, Train Acc: 0.5892, Test Acc: 0.5375
    Epoch: 098, Train Acc: 0.5934, Test Acc: 0.5375
    Epoch: 099, Train Acc: 0.6141, Test Acc: 0.5375
    Epoch: 100, Train Acc: 0.6183, Test Acc: 0.5375
    Epoch: 101, Train Acc: 0.6017, Test Acc: 0.5375
    Epoch: 102, Train Acc: 0.6100, Test Acc: 0.5500
    Epoch: 103, Train Acc: 0.6100, Test Acc: 0.5625
    Epoch: 104, Train Acc: 0.6183, Test Acc: 0.5500
    Epoch: 105, Train Acc: 0.6141, Test Acc: 0.5625
    Epoch: 106, Train Acc: 0.6307, Test Acc: 0.5625
    Epoch: 107, Train Acc: 0.6349, Test Acc: 0.5875
    Epoch: 108, Train Acc: 0.6556, Test Acc: 0.6000
    Epoch: 109, Train Acc: 0.6598, Test Acc: 0.6000
    Epoch: 110, Train Acc: 0.6639, Test Acc: 0.5875
    Epoch: 111, Train Acc: 0.6763, Test Acc: 0.6000
    Epoch: 112, Train Acc: 0.6888, Test Acc: 0.6000
    Epoch: 113, Train Acc: 0.6846, Test Acc: 0.6000
    Epoch: 114, Train Acc: 0.6888, Test Acc: 0.5875
    Epoch: 115, Train Acc: 0.7012, Test Acc: 0.5750
    Epoch: 116, Train Acc: 0.7012, Test Acc: 0.5875
    Epoch: 117, Train Acc: 0.7054, Test Acc: 0.5875
    Epoch: 118, Train Acc: 0.7012, Test Acc: 0.5875
    Epoch: 119, Train Acc: 0.7137, Test Acc: 0.5875
    Epoch: 120, Train Acc: 0.7178, Test Acc: 0.5625
    Epoch: 121, Train Acc: 0.7220, Test Acc: 0.5500
    Epoch: 122, Train Acc: 0.7261, Test Acc: 0.5250
    Epoch: 123, Train Acc: 0.7427, Test Acc: 0.5500
    Epoch: 124, Train Acc: 0.7427, Test Acc: 0.5625
    Epoch: 125, Train Acc: 0.7469, Test Acc: 0.5375
    Epoch: 126, Train Acc: 0.7635, Test Acc: 0.5250
    Epoch: 127, Train Acc: 0.7676, Test Acc: 0.5250
    Epoch: 128, Train Acc: 0.7759, Test Acc: 0.5375
    Epoch: 129, Train Acc: 0.7759, Test Acc: 0.5375
    Epoch: 130, Train Acc: 0.7842, Test Acc: 0.5375
    Epoch: 131, Train Acc: 0.7801, Test Acc: 0.5375
    Epoch: 132, Train Acc: 0.7801, Test Acc: 0.5375
    Epoch: 133, Train Acc: 0.7801, Test Acc: 0.5375
    Epoch: 134, Train Acc: 0.7925, Test Acc: 0.5375
    Epoch: 135, Train Acc: 0.8008, Test Acc: 0.5375
    Epoch: 136, Train Acc: 0.7884, Test Acc: 0.5500
    Epoch: 137, Train Acc: 0.7967, Test Acc: 0.5500
    Epoch: 138, Train Acc: 0.8133, Test Acc: 0.5500
    Epoch: 139, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 140, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 141, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 142, Train Acc: 0.8091, Test Acc: 0.5500
    Epoch: 143, Train Acc: 0.8133, Test Acc: 0.5500
    Epoch: 144, Train Acc: 0.7967, Test Acc: 0.5500
    Epoch: 145, Train Acc: 0.8050, Test Acc: 0.5625
    Epoch: 146, Train Acc: 0.8133, Test Acc: 0.5500
    Epoch: 147, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 148, Train Acc: 0.8133, Test Acc: 0.5625
    Epoch: 149, Train Acc: 0.8216, Test Acc: 0.5625
    Epoch: 150, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 151, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 152, Train Acc: 0.8216, Test Acc: 0.5500
    Epoch: 153, Train Acc: 0.8216, Test Acc: 0.5500
    Epoch: 154, Train Acc: 0.8257, Test Acc: 0.5625
    Epoch: 155, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 156, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 157, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 158, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 159, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 160, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 161, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 162, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 163, Train Acc: 0.8257, Test Acc: 0.5375
    Epoch: 164, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 165, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 166, Train Acc: 0.8465, Test Acc: 0.5375
    Epoch: 167, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 168, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 169, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 170, Train Acc: 0.8423, Test Acc: 0.5500
    Epoch: 171, Train Acc: 0.8340, Test Acc: 0.5625
    Epoch: 172, Train Acc: 0.8506, Test Acc: 0.5375
    Epoch: 173, Train Acc: 0.8589, Test Acc: 0.5500
    Epoch: 174, Train Acc: 0.8548, Test Acc: 0.5375
    Epoch: 175, Train Acc: 0.8589, Test Acc: 0.5375
    Epoch: 176, Train Acc: 0.8589, Test Acc: 0.5500
    Epoch: 177, Train Acc: 0.8589, Test Acc: 0.5625
    Epoch: 178, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 179, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 180, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 181, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 182, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 183, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 184, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 185, Train Acc: 0.8714, Test Acc: 0.5500
    Epoch: 186, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 187, Train Acc: 0.8631, Test Acc: 0.5625
    Epoch: 188, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 189, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 190, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 191, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 192, Train Acc: 0.8714, Test Acc: 0.5375
    Epoch: 193, Train Acc: 0.8672, Test Acc: 0.5375
    Epoch: 194, Train Acc: 0.8672, Test Acc: 0.5375
    Epoch: 195, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 196, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 197, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 198, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 199, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 200, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 201, Train Acc: 0.8714, Test Acc: 0.5375
    Epoch: 202, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 203, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 204, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 205, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 206, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 207, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 208, Train Acc: 0.8838, Test Acc: 0.5500
    Epoch: 209, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 210, Train Acc: 0.8880, Test Acc: 0.5375
    Epoch: 211, Train Acc: 0.8921, Test Acc: 0.5375
    Epoch: 212, Train Acc: 0.8880, Test Acc: 0.5375
    Epoch: 213, Train Acc: 0.8880, Test Acc: 0.5375
    Epoch: 214, Train Acc: 0.8921, Test Acc: 0.5375
    Epoch: 215, Train Acc: 0.8921, Test Acc: 0.5125
    Epoch: 216, Train Acc: 0.9046, Test Acc: 0.5125
    Epoch: 217, Train Acc: 0.9087, Test Acc: 0.5250
    Epoch: 218, Train Acc: 0.9004, Test Acc: 0.5125
    Epoch: 219, Train Acc: 0.9004, Test Acc: 0.5125
    Epoch: 220, Train Acc: 0.9046, Test Acc: 0.5125
    Epoch: 221, Train Acc: 0.9004, Test Acc: 0.5250
    Epoch: 222, Train Acc: 0.9046, Test Acc: 0.5250
    Epoch: 223, Train Acc: 0.9046, Test Acc: 0.5250
    Epoch: 224, Train Acc: 0.9129, Test Acc: 0.5250
    Epoch: 225, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 226, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 227, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 228, Train Acc: 0.9087, Test Acc: 0.5375
    Epoch: 229, Train Acc: 0.9170, Test Acc: 0.5500
    Epoch: 230, Train Acc: 0.9129, Test Acc: 0.5375
    Epoch: 231, Train Acc: 0.9087, Test Acc: 0.5250
    Epoch: 232, Train Acc: 0.9170, Test Acc: 0.5250
    Epoch: 233, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 234, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 235, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 236, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 237, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 238, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 239, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 240, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 241, Train Acc: 0.9295, Test Acc: 0.5250
    Epoch: 242, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 243, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 244, Train Acc: 0.9295, Test Acc: 0.5250
    Epoch: 245, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 246, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 247, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 248, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 249, Train Acc: 0.9419, Test Acc: 0.5250
    Epoch: 250, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 251, Train Acc: 0.9336, Test Acc: 0.5125
    Epoch: 252, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 253, Train Acc: 0.9378, Test Acc: 0.5250
    Epoch: 254, Train Acc: 0.9461, Test Acc: 0.5250
    Epoch: 255, Train Acc: 0.9419, Test Acc: 0.5250
    Epoch: 256, Train Acc: 0.9502, Test Acc: 0.5125
    Epoch: 257, Train Acc: 0.9502, Test Acc: 0.5250
    Epoch: 258, Train Acc: 0.9461, Test Acc: 0.5250
    Epoch: 259, Train Acc: 0.9461, Test Acc: 0.5250
    Epoch: 260, Train Acc: 0.9502, Test Acc: 0.5250
    Epoch: 261, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 262, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 263, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 264, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 265, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 266, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 267, Train Acc: 0.9585, Test Acc: 0.5125
    Epoch: 268, Train Acc: 0.9585, Test Acc: 0.5125
    Epoch: 269, Train Acc: 0.9627, Test Acc: 0.5250
    Epoch: 270, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 271, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 272, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 273, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 274, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 275, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 276, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 277, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 278, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 279, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 280, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 281, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 282, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 283, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 284, Train Acc: 0.9627, Test Acc: 0.5625
    Epoch: 285, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 286, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 287, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 288, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 289, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 290, Train Acc: 0.9585, Test Acc: 0.5750
    Epoch: 291, Train Acc: 0.9627, Test Acc: 0.5625
    Epoch: 292, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 293, Train Acc: 0.9627, Test Acc: 0.5750
    Epoch: 294, Train Acc: 0.9627, Test Acc: 0.5750
    Epoch: 295, Train Acc: 0.9627, Test Acc: 0.5750
    Epoch: 296, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 297, Train Acc: 0.9710, Test Acc: 0.5750
    Epoch: 298, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 299, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 300, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 301, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 302, Train Acc: 0.9710, Test Acc: 0.5750
    Epoch: 303, Train Acc: 0.9751, Test Acc: 0.5500
    Epoch: 304, Train Acc: 0.9710, Test Acc: 0.5625
    Epoch: 305, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 306, Train Acc: 0.9710, Test Acc: 0.5750
    Epoch: 307, Train Acc: 0.9751, Test Acc: 0.5750
    Epoch: 308, Train Acc: 0.9751, Test Acc: 0.5750
    Epoch: 309, Train Acc: 0.9793, Test Acc: 0.5500
    Epoch: 310, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 311, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 312, Train Acc: 0.9751, Test Acc: 0.5625
    Epoch: 313, Train Acc: 0.9710, Test Acc: 0.6000
    Epoch: 314, Train Acc: 0.9751, Test Acc: 0.5750
    Epoch: 315, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 316, Train Acc: 0.9793, Test Acc: 0.5750
    Epoch: 317, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 318, Train Acc: 0.9793, Test Acc: 0.5750
    Epoch: 319, Train Acc: 0.9793, Test Acc: 0.6000
    Epoch: 320, Train Acc: 0.9793, Test Acc: 0.6000
    Epoch: 321, Train Acc: 0.9793, Test Acc: 0.6000
    Epoch: 322, Train Acc: 0.9793, Test Acc: 0.5500
    Epoch: 323, Train Acc: 0.9834, Test Acc: 0.5500
    Epoch: 324, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 325, Train Acc: 0.9793, Test Acc: 0.5875
    Epoch: 326, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 327, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 328, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 329, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 330, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 331, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 332, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 333, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 334, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 335, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 336, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 337, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 338, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 339, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 340, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 341, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 342, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 343, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 344, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 345, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 346, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 347, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 348, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 349, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 350, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 351, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 352, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 353, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 354, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 355, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 356, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 357, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 358, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 359, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 360, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 361, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 362, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 363, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 364, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 365, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 366, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 367, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 368, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 369, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 370, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 371, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 372, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 373, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 374, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 375, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 376, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 377, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 378, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 379, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 380, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 381, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 382, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 383, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 384, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 385, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 386, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 387, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 388, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 389, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 390, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 391, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 392, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 393, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 394, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 395, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 396, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 397, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 398, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 399, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 400, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 401, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 402, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 403, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 404, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 405, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 406, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 407, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 408, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 409, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 410, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 411, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 412, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 413, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 414, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 415, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 416, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 417, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 418, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 419, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 420, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 421, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 422, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 423, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 424, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 425, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 426, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 427, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 428, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 429, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 430, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 431, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 432, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 433, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 434, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 435, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 436, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 437, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 438, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 439, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 440, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 441, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 442, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 443, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 444, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 445, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 446, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 447, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 448, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 449, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 450, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 451, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 452, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 453, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 454, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 455, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 456, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 457, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 458, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 459, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 460, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 461, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 462, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 463, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 464, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 465, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 466, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 467, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 468, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 469, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 470, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 471, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 472, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 473, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 474, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 475, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 476, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 477, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 478, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 479, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 480, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 481, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 482, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 483, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 484, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 485, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 486, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 487, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 488, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 489, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 490, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 491, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 492, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 493, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 494, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 495, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 496, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 497, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 498, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 499, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 500, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 501, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 502, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 503, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 504, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 505, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 506, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 507, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 508, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 509, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 510, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 511, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 512, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 513, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 514, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 515, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 516, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 517, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 518, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 519, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 520, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 521, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 522, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 523, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 524, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 525, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 526, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 527, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 528, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 529, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 530, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 531, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 532, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 533, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 534, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 535, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 536, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 537, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 538, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 539, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 540, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 541, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 542, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 543, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 544, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 545, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 546, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 547, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 548, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 549, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 550, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 551, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 552, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 553, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 554, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 555, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 556, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 557, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 558, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 559, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 560, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 561, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 562, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 563, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 564, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 565, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 566, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 567, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 568, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 569, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 570, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6000



```python
torch.manual_seed(163)
model = GCN(hidden_channels=32)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track[32] = [train_accs, test_accs]
```

    Epoch: 001, Train Acc: 0.3859, Test Acc: 0.5125
    Epoch: 002, Train Acc: 0.3942, Test Acc: 0.5250
    Epoch: 003, Train Acc: 0.4025, Test Acc: 0.5250
    Epoch: 004, Train Acc: 0.3983, Test Acc: 0.5375
    Epoch: 005, Train Acc: 0.4149, Test Acc: 0.5500
    Epoch: 006, Train Acc: 0.4606, Test Acc: 0.5625
    Epoch: 007, Train Acc: 0.4896, Test Acc: 0.5375
    Epoch: 008, Train Acc: 0.4855, Test Acc: 0.5375
    Epoch: 009, Train Acc: 0.4647, Test Acc: 0.5125
    Epoch: 010, Train Acc: 0.4979, Test Acc: 0.5500
    Epoch: 011, Train Acc: 0.5062, Test Acc: 0.5625
    Epoch: 012, Train Acc: 0.5145, Test Acc: 0.5750
    Epoch: 013, Train Acc: 0.5311, Test Acc: 0.5875
    Epoch: 014, Train Acc: 0.5436, Test Acc: 0.5875
    Epoch: 015, Train Acc: 0.5394, Test Acc: 0.5625
    Epoch: 016, Train Acc: 0.5519, Test Acc: 0.6000
    Epoch: 017, Train Acc: 0.5394, Test Acc: 0.6125
    Epoch: 018, Train Acc: 0.5436, Test Acc: 0.5875
    Epoch: 019, Train Acc: 0.5519, Test Acc: 0.5875
    Epoch: 020, Train Acc: 0.5643, Test Acc: 0.6000
    Epoch: 021, Train Acc: 0.5726, Test Acc: 0.6000
    Epoch: 022, Train Acc: 0.5851, Test Acc: 0.6125
    Epoch: 023, Train Acc: 0.5851, Test Acc: 0.6125
    Epoch: 024, Train Acc: 0.5934, Test Acc: 0.6000
    Epoch: 025, Train Acc: 0.6100, Test Acc: 0.6375
    Epoch: 026, Train Acc: 0.6349, Test Acc: 0.6375
    Epoch: 027, Train Acc: 0.6349, Test Acc: 0.6375
    Epoch: 028, Train Acc: 0.6390, Test Acc: 0.6375
    Epoch: 029, Train Acc: 0.6307, Test Acc: 0.6125
    Epoch: 030, Train Acc: 0.6349, Test Acc: 0.6125
    Epoch: 031, Train Acc: 0.6515, Test Acc: 0.6000
    Epoch: 032, Train Acc: 0.6390, Test Acc: 0.6000
    Epoch: 033, Train Acc: 0.6515, Test Acc: 0.5875
    Epoch: 034, Train Acc: 0.6473, Test Acc: 0.6000
    Epoch: 035, Train Acc: 0.6680, Test Acc: 0.6000
    Epoch: 036, Train Acc: 0.6846, Test Acc: 0.6000
    Epoch: 037, Train Acc: 0.7054, Test Acc: 0.6125
    Epoch: 038, Train Acc: 0.7220, Test Acc: 0.6000
    Epoch: 039, Train Acc: 0.7552, Test Acc: 0.6125
    Epoch: 040, Train Acc: 0.7593, Test Acc: 0.5875
    Epoch: 041, Train Acc: 0.7676, Test Acc: 0.5875
    Epoch: 042, Train Acc: 0.7884, Test Acc: 0.6250
    Epoch: 043, Train Acc: 0.8133, Test Acc: 0.6250
    Epoch: 044, Train Acc: 0.8133, Test Acc: 0.6250
    Epoch: 045, Train Acc: 0.8216, Test Acc: 0.6125
    Epoch: 046, Train Acc: 0.8382, Test Acc: 0.6000
    Epoch: 047, Train Acc: 0.8631, Test Acc: 0.6125
    Epoch: 048, Train Acc: 0.8672, Test Acc: 0.6250
    Epoch: 049, Train Acc: 0.8797, Test Acc: 0.6125
    Epoch: 050, Train Acc: 0.8838, Test Acc: 0.6250
    Epoch: 051, Train Acc: 0.8880, Test Acc: 0.6375
    Epoch: 052, Train Acc: 0.9046, Test Acc: 0.6375
    Epoch: 053, Train Acc: 0.9004, Test Acc: 0.6250
    Epoch: 054, Train Acc: 0.9212, Test Acc: 0.6375
    Epoch: 055, Train Acc: 0.9253, Test Acc: 0.6125
    Epoch: 056, Train Acc: 0.9336, Test Acc: 0.6250
    Epoch: 057, Train Acc: 0.9419, Test Acc: 0.6250
    Epoch: 058, Train Acc: 0.9502, Test Acc: 0.6250
    Epoch: 059, Train Acc: 0.9502, Test Acc: 0.6125
    Epoch: 060, Train Acc: 0.9502, Test Acc: 0.6250
    Epoch: 061, Train Acc: 0.9544, Test Acc: 0.6250
    Epoch: 062, Train Acc: 0.9544, Test Acc: 0.6250
    Epoch: 063, Train Acc: 0.9585, Test Acc: 0.6375
    Epoch: 064, Train Acc: 0.9627, Test Acc: 0.6375
    Epoch: 065, Train Acc: 0.9544, Test Acc: 0.6375
    Epoch: 066, Train Acc: 0.9627, Test Acc: 0.6250
    Epoch: 067, Train Acc: 0.9627, Test Acc: 0.6250
    Epoch: 068, Train Acc: 0.9668, Test Acc: 0.6500
    Epoch: 069, Train Acc: 0.9627, Test Acc: 0.6375
    Epoch: 070, Train Acc: 0.9668, Test Acc: 0.6375
    Epoch: 071, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 072, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 073, Train Acc: 0.9710, Test Acc: 0.6375
    Epoch: 074, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 075, Train Acc: 0.9751, Test Acc: 0.6375
    Epoch: 076, Train Acc: 0.9751, Test Acc: 0.6500
    Epoch: 077, Train Acc: 0.9751, Test Acc: 0.6500
    Epoch: 078, Train Acc: 0.9751, Test Acc: 0.6250
    Epoch: 079, Train Acc: 0.9793, Test Acc: 0.6250
    Epoch: 080, Train Acc: 0.9793, Test Acc: 0.6375
    Epoch: 081, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 082, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 083, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 084, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 085, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 086, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 087, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 088, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 089, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 090, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 091, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 092, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 093, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 094, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 095, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 096, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 097, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 098, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 099, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 100, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 101, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 102, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 103, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 104, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 105, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 106, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 107, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 108, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 109, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 110, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 111, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 112, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 113, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 114, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 115, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 116, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 117, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 118, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 119, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 120, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 121, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 122, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 123, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 124, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 125, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 126, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 127, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 128, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 129, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 130, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 131, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 132, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 133, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 134, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 135, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 136, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 137, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 138, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 139, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 140, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 141, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 142, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 143, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 144, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 145, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 146, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 147, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 148, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 149, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 150, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 151, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 152, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 153, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 154, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 155, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 156, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 157, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 158, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 159, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 160, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 161, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 162, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 163, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 164, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 165, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 166, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 167, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 168, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 169, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 170, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 171, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 172, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 173, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 174, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 175, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 176, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 177, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 178, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 179, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 180, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 181, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 182, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 183, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 184, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 185, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 186, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 187, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 188, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 189, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 190, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 191, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 192, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 193, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 194, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 195, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 196, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 197, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 198, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 199, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 200, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 201, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 202, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 203, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 204, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 205, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 206, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 207, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 208, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 209, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 210, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 211, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 212, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 213, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 214, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 215, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 216, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 217, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 218, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 219, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 220, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 221, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 222, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 223, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 224, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 225, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 226, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 227, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 228, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 229, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 230, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 231, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 232, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 233, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 234, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 235, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 236, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 237, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 238, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 239, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 240, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 241, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 242, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 243, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 244, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 245, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 246, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 247, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 248, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 249, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 250, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 251, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 252, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 253, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 254, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 255, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 256, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 257, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 258, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 259, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 260, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 261, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 262, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 263, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 264, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 265, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 266, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 267, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 268, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 269, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 270, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 271, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 272, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 273, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 274, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 275, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 276, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 277, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 278, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 279, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 280, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 281, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 282, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 283, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 284, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 285, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 286, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 287, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 288, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 289, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 290, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 291, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 292, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 293, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 294, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 295, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 296, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 297, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 298, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 299, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 300, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 301, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 302, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 303, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 304, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 305, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 306, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 307, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 308, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 309, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 310, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 311, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 312, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 313, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 314, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 315, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 316, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 317, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 318, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 319, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 320, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 321, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 322, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 323, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 324, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 325, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 326, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 327, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 328, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 329, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 330, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 331, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 332, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 333, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 334, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 335, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 336, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 337, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 338, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 339, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 340, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 341, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 342, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 343, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 344, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 345, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 346, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 347, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 348, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 349, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 350, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 351, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 352, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 353, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 354, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 355, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 356, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 357, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 358, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 359, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 360, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 361, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 362, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 363, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 364, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 365, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 366, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 367, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 368, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 369, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 370, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 371, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 372, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 373, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 374, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 375, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 376, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 377, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 378, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 379, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 380, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 381, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 382, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 383, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 384, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 385, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 386, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 387, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 388, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 389, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 390, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 391, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 392, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 393, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 394, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 395, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 396, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 397, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 398, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 399, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 400, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 401, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 402, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 403, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 404, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 405, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 406, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 407, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 408, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 409, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 410, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 411, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 412, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 413, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 414, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 415, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 416, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 417, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 418, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 419, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 420, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 421, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 422, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 423, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 424, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 425, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 426, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 427, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 428, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 429, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 430, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 431, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 432, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 433, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 434, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 435, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 436, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 437, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 438, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 439, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 440, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 441, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 442, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 443, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 444, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 445, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 446, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 447, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 448, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 449, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 450, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 451, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 452, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 453, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 454, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 455, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 456, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 457, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 458, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 459, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 460, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 461, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 462, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 463, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 464, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 465, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 466, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 467, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 468, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 469, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 470, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 471, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 472, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 473, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 474, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 475, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 476, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 477, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 478, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 479, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 480, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 481, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 482, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 483, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 484, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 485, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 486, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 487, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 488, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 489, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 490, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 491, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 492, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 493, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 494, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 495, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 496, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 497, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 498, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 499, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 500, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 501, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 502, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 503, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 504, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 505, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 506, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 507, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 508, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 509, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 510, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 511, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 512, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 513, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 514, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 515, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 516, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 517, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 518, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 519, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 520, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 521, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 522, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 523, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 524, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 525, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 526, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 527, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 528, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 529, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 530, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 531, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 532, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 533, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 534, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 535, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 536, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 537, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 538, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 539, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 540, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 541, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 542, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 543, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 544, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 545, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 546, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 547, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 548, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 549, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 550, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 551, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 552, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 553, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 554, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 555, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 556, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 557, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 558, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 559, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 560, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 561, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 562, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 563, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 564, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 565, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 566, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 567, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 568, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 569, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 570, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6500



```python
torch.manual_seed(163)
model = GCN(hidden_channels=64)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track[64] = [train_accs, test_accs]
```

    Epoch: 001, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 002, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 003, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 004, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 005, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 006, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 007, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 008, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 009, Train Acc: 0.4025, Test Acc: 0.3000
    Epoch: 010, Train Acc: 0.4025, Test Acc: 0.3125
    Epoch: 011, Train Acc: 0.4066, Test Acc: 0.3125
    Epoch: 012, Train Acc: 0.4066, Test Acc: 0.3125
    Epoch: 013, Train Acc: 0.4523, Test Acc: 0.3250
    Epoch: 014, Train Acc: 0.5311, Test Acc: 0.4000
    Epoch: 015, Train Acc: 0.5436, Test Acc: 0.4000
    Epoch: 016, Train Acc: 0.5726, Test Acc: 0.4750
    Epoch: 017, Train Acc: 0.5809, Test Acc: 0.5250
    Epoch: 018, Train Acc: 0.5851, Test Acc: 0.5500
    Epoch: 019, Train Acc: 0.6017, Test Acc: 0.5375
    Epoch: 020, Train Acc: 0.5934, Test Acc: 0.5500
    Epoch: 021, Train Acc: 0.5809, Test Acc: 0.5500
    Epoch: 022, Train Acc: 0.5975, Test Acc: 0.5500
    Epoch: 023, Train Acc: 0.5892, Test Acc: 0.5750
    Epoch: 024, Train Acc: 0.5892, Test Acc: 0.5750
    Epoch: 025, Train Acc: 0.6058, Test Acc: 0.5875
    Epoch: 026, Train Acc: 0.6515, Test Acc: 0.6375
    Epoch: 027, Train Acc: 0.6846, Test Acc: 0.6375
    Epoch: 028, Train Acc: 0.7054, Test Acc: 0.6250
    Epoch: 029, Train Acc: 0.7386, Test Acc: 0.6250
    Epoch: 030, Train Acc: 0.7718, Test Acc: 0.6000
    Epoch: 031, Train Acc: 0.7925, Test Acc: 0.6125
    Epoch: 032, Train Acc: 0.8050, Test Acc: 0.6375
    Epoch: 033, Train Acc: 0.8091, Test Acc: 0.6250
    Epoch: 034, Train Acc: 0.8216, Test Acc: 0.5750
    Epoch: 035, Train Acc: 0.8465, Test Acc: 0.6500
    Epoch: 036, Train Acc: 0.8465, Test Acc: 0.6125
    Epoch: 037, Train Acc: 0.8589, Test Acc: 0.6125
    Epoch: 038, Train Acc: 0.8714, Test Acc: 0.6250
    Epoch: 039, Train Acc: 0.8797, Test Acc: 0.6125
    Epoch: 040, Train Acc: 0.8921, Test Acc: 0.6375
    Epoch: 041, Train Acc: 0.9004, Test Acc: 0.6250
    Epoch: 042, Train Acc: 0.9087, Test Acc: 0.6125
    Epoch: 043, Train Acc: 0.9253, Test Acc: 0.6250
    Epoch: 044, Train Acc: 0.9336, Test Acc: 0.6500
    Epoch: 045, Train Acc: 0.9336, Test Acc: 0.6000
    Epoch: 046, Train Acc: 0.9378, Test Acc: 0.6250
    Epoch: 047, Train Acc: 0.9461, Test Acc: 0.6000
    Epoch: 048, Train Acc: 0.9502, Test Acc: 0.5875
    Epoch: 049, Train Acc: 0.9502, Test Acc: 0.5750
    Epoch: 050, Train Acc: 0.9585, Test Acc: 0.5750
    Epoch: 051, Train Acc: 0.9585, Test Acc: 0.5875
    Epoch: 052, Train Acc: 0.9710, Test Acc: 0.6000
    Epoch: 053, Train Acc: 0.9710, Test Acc: 0.6000
    Epoch: 054, Train Acc: 0.9793, Test Acc: 0.6125
    Epoch: 055, Train Acc: 0.9834, Test Acc: 0.6000
    Epoch: 056, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 057, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 058, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 059, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 060, Train Acc: 0.9876, Test Acc: 0.6250
    Epoch: 061, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 062, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 063, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 064, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 065, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 066, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 067, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 068, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 069, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 070, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 071, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 072, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 073, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 074, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 075, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 076, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 077, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 078, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 079, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 080, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 081, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 082, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 083, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 084, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 085, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 086, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 087, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 088, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 089, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 090, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 091, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 092, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 093, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 094, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 095, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 096, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 097, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 098, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 099, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 100, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 101, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 102, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 103, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 104, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 105, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 106, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 107, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 108, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 109, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 110, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 111, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 112, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 113, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 114, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 115, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 116, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 117, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 118, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 119, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 120, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 121, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 122, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 123, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 124, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 125, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 126, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 127, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 128, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 129, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 130, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 131, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 132, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 133, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 134, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 135, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 136, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 137, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 138, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 139, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 140, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 141, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 142, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 143, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 144, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 145, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 146, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 147, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 148, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 149, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 150, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 151, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 152, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 153, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 154, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 155, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 156, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 157, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 158, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 159, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 160, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 161, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 162, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 163, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 164, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 165, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 166, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 167, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 168, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 169, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 170, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 171, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 172, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 173, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 174, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 175, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 176, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 177, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 178, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 179, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 180, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 181, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 182, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 183, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 184, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 185, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 186, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 187, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 188, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 189, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 190, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 191, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 192, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 193, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 194, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 195, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 196, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 197, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 198, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 199, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 200, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 201, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 202, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 203, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 204, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 205, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 206, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 207, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 208, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 209, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 210, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 211, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 212, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 213, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 214, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 215, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 216, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 217, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 218, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 219, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 220, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 221, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 222, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 223, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 224, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 225, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 226, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 227, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 228, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 229, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 230, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 231, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 232, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 233, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 234, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 235, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 236, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 237, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 238, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 239, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 240, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 241, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 242, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 243, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 244, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 245, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 246, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 247, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 248, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 249, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 250, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 251, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 252, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 253, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 254, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 255, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 256, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 257, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 258, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 259, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 260, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 261, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 262, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 263, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 264, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 265, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 266, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 267, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 268, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 269, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 270, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 271, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 272, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 273, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 274, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 275, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 276, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 277, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 278, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 279, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 280, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 281, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 282, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 283, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 284, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 285, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 286, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 287, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 288, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 289, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 290, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 291, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 292, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 293, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 294, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 295, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 296, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 297, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 298, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 299, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 300, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 301, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 302, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 303, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 304, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 305, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 306, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 307, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 308, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 309, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 310, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 311, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 312, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 313, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 314, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 315, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 316, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 317, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 318, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 319, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 320, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 321, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 322, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 323, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 324, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 325, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 326, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 327, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 328, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 329, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 330, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 331, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 332, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 333, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 334, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 335, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 336, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 337, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 338, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 339, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 340, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 341, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 342, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 343, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 344, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 345, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 346, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 347, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 348, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 349, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 350, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 351, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 352, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 353, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 354, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 355, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 356, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 357, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 358, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 359, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 360, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 361, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 362, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 363, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 364, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 365, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 366, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 367, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 368, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 369, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 370, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 371, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 372, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 373, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 374, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 375, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 376, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 377, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 378, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 379, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 380, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 381, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 382, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 383, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 384, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 385, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 386, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 387, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 388, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 389, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 390, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 391, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 392, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 393, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 394, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 395, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 396, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 397, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 398, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 399, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 400, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 401, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 402, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 403, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 404, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 405, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 406, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 407, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 408, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 409, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 410, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 411, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 412, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 413, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 414, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 415, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 416, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 417, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 418, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 419, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 420, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 421, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 422, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 423, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 424, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 425, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 426, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 427, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 428, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 429, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 430, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 431, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 432, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 433, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 434, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 435, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 436, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 437, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 438, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 439, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 440, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 441, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 442, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 443, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 444, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 445, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 446, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 447, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 448, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 449, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 450, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 451, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 452, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 453, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 454, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 455, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 456, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 457, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 458, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 459, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 460, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 461, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 462, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 463, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 464, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 465, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 466, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 467, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 468, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 469, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 470, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 471, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 472, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 473, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 474, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 475, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 476, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 477, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 478, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 479, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 480, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 481, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 482, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 483, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 484, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 485, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 486, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 487, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 488, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 489, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 490, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 491, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 492, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 493, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 494, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 495, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 496, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 497, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 498, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 499, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 500, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 501, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 502, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 503, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 504, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 505, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 506, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 507, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 508, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 509, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 510, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 511, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 512, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 513, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 514, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 515, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 516, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 517, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 518, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 519, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 520, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 521, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 522, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 523, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 524, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 525, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 526, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 527, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 528, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 529, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 530, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 531, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 532, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 533, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 534, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 535, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 536, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 537, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 538, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 539, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 540, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 541, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 542, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 543, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 544, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 545, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 546, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 547, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 548, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 549, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 550, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 551, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 552, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 553, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 554, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 555, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 556, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 557, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 558, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 559, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 560, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 561, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 562, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 563, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 564, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 565, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 566, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 567, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 568, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 569, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 570, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6000



```python
track.keys()


```




    dict_keys([16, 8, 32, 64])



### Try again, this time with a fully-connected graph!


```python
## Define one sample ordering for all anchors. 
sample_ord = satc['sample'].unique()
store_dict_and_data_FULLY_CONNECTED = dict()
import time
start = time.time()
for anchor in satc['anchor'].unique():

    label = list(v[v['anchor']==anchor]['class'])[0]
    select = satc[satc['anchor']==anchor]
    a = construct_graph(select, 'test', 'target', ['seqComp','sampleFraction'], ["targetHamming" , "targetLevenshtein" , "corrSampleFractions" , "corrBoolSampleFractions" ], ['full'],satc['sample'].unique())

    ## Define one target ordering for this anchor.
    ## The index of each entry defines its node ID. 
    target_ord = np.array(list(a['nodeFeatures']['seqComp'].keys()))

    ## Extract node features. 
    nodeFeatureList = []
    for target in target_ord:
        nodeFeatureList.append(list(np.append(a['nodeFeatures']['seqComp'][target].flatten(),\
        a['nodeFeatures']['sampleFraction'][target])))

    ## Define the tensor corresponding to node features. 
    x = torch.tensor(nodeFeatureList)

    ## Extract edges and edge features. 
    edgeFeatureListOrder = []
    edge_index_top, edge_index_bot = [], []
    edgeFeaturesVector = []
    for edge in a['edges']:
        i, j = np.nonzero(target_ord==edge[0])[0][0], np.nonzero(target_ord==edge[1])[0][0]
        edge_index_top.append(i)
        edge_index_top.append(j)
        edge_index_bot.append(j)
        edge_index_bot.append(i)
        if not len(edgeFeatureListOrder):
            edgeFeatureListOrder = list(a['edgeFeatures'][edge].keys())
        extractedFeatures = []
        for eF in edgeFeatureListOrder:
            extractedFeatures.append(a['edgeFeatures'][edge][eF])
        edgeFeaturesVector.append(extractedFeatures) 
        edgeFeaturesVector.append(extractedFeatures)

    ## Define the tensors corresponding to edges and edge features. 
    edge_attributes = torch.tensor(edgeFeaturesVector)
    edge_index = torch.tensor([edge_index_top,edge_index_bot])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=label)

    store_dict_and_data_FULLY_CONNECTED[anchor] = label, a, data
    print(time.time()-start)
```

    0.0738379955291748
    0.21361255645751953
    0.3127877712249756
    0.3956437110900879
    0.44136476516723633
    0.5505409240722656
    0.5901758670806885
    0.7094306945800781
    0.818770170211792
    0.871807336807251
    0.9802563190460205
    1.014251947402954
    1.088026523590088
    1.2163214683532715
    1.2891972064971924
    1.3295328617095947
    1.3634302616119385
    1.4030394554138184
    1.4484343528747559
    1.5687496662139893
    1.6976776123046875
    1.7431797981262207
    1.802659034729004
    1.886312484741211
    1.9319682121276855
    1.9981849193572998
    2.043687343597412
    2.096057891845703
    2.1310181617736816
    2.1766653060913086
    2.240400552749634
    2.318197727203369
    2.6815714836120605
    2.709150791168213
    2.7625882625579834
    2.8374719619750977
    2.868666887283325
    2.9371418952941895
    3.054932117462158
    3.0966079235076904
    3.138141393661499
    3.1932272911071777
    3.2351205348968506
    3.327998161315918
    3.3703017234802246
    3.4074361324310303
    3.461714029312134
    3.503988742828369
    3.540592908859253
    3.588742971420288
    3.636680841445923
    3.6977198123931885
    3.7463924884796143
    3.7781100273132324
    3.8048791885375977
    3.841282606124878
    3.9031786918640137
    3.945774555206299
    3.98218035697937
    4.018409729003906
    4.049556493759155
    4.103362560272217
    4.172407865524292
    4.208317995071411
    4.2451770305633545
    4.467778444290161
    4.491472005844116
    4.582469940185547
    4.614819765090942
    4.662976026535034
    4.7051379680633545
    4.788599491119385
    4.890301704406738
    5.075302362442017
    5.112220048904419
    5.1596596240997314
    5.235518217086792
    5.277284383773804
    5.408183574676514
    5.463044166564941
    5.504750728607178
    10.559147357940674
    10.586951494216919
    10.613199710845947
    10.703626871109009
    10.72960615158081
    10.760485887527466
    10.845106363296509
    11.006437063217163
    11.043422222137451
    11.104307413101196
    11.152819156646729
    11.206649780273438
    11.261064052581787
    11.292895078659058
    11.377959489822388
    11.604317665100098
    11.644022226333618
    11.712136268615723
    11.822721719741821
    11.864408493041992
    11.91919469833374
    11.96086835861206
    12.021476030349731
    12.06890869140625
    12.14389967918396
    12.204528570175171
    12.235219240188599
    12.27688980102539
    12.319029092788696
    12.53125548362732
    12.579471588134766
    12.742643117904663
    12.779258251190186
    12.815903902053833
    12.853074550628662
    12.936747074127197
    13.005369901657104
    13.042176008224487
    13.07344651222229
    13.123045206069946
    13.199373245239258
    13.24800157546997
    13.285246849060059
    13.311641931533813
    13.38113284111023
    13.435643672943115
    13.622159242630005
    13.669525384902954
    13.738363981246948
    13.961786270141602
    14.081796646118164
    14.157701253890991
    14.194626092910767
    14.255441427230835
    14.283162593841553
    14.31423020362854
    14.499710083007812
    14.60037636756897
    14.637305974960327
    14.684990882873535
    14.733541250228882
    14.794543504714966
    15.043385028839111
    17.031700611114502
    17.074584245681763
    17.13435387611389
    17.262821674346924
    17.316593885421753
    17.347373247146606
    19.436562061309814
    19.468669176101685
    19.559938430786133
    19.602368354797363
    19.81372594833374
    19.867936372756958
    19.904304027557373
    20.045966625213623
    20.10063099861145
    20.136656045913696
    20.184491395950317
    20.221404790878296
    20.30667471885681
    20.407607316970825
    20.439194679260254
    20.486557245254517
    20.517577648162842
    20.59371304512024
    20.641007900238037
    20.689356088638306
    20.71515917778015
    20.76395058631897
    20.811660289764404
    20.880086660385132
    20.948335647583008
    20.974271297454834
    21.00085139274597
    21.055580854415894
    21.085973262786865
    21.401190996170044
    21.43276619911194
    21.542240142822266
    21.610935926437378
    21.64780879020691
    21.67836356163025
    21.714640140533447
    21.751148223876953
    21.78750777244568
    21.855733394622803
    21.92312455177307
    21.95857882499695
    21.98927593231201
    22.18277406692505
    22.299950122833252
    22.37408709526062
    22.421508312225342
    22.45709991455078
    22.522918462753296
    22.604463815689087
    22.670844554901123
    22.70717144012451
    22.833576440811157
    22.893847942352295
    22.916234254837036
    22.957351446151733
    22.988218307495117
    23.0411639213562
    23.07621431350708
    23.11199140548706
    23.218103408813477
    23.24839973449707
    23.27814555168152
    23.318873643875122
    23.340264797210693
    23.366069555282593
    23.396273851394653
    23.437284231185913
    23.490386962890625
    23.571489810943604
    24.260764837265015
    24.33570146560669
    24.555020093917847
    24.62334442138672
    24.664726972579956
    24.724534273147583
    24.760223627090454
    24.790921926498413
    24.89056658744812
    24.921072959899902
    24.968159198760986
    30.16742968559265
    30.227688312530518
    30.27531123161316
    30.30170178413391
    30.332621097564697
    30.415709257125854
    30.492329835891724
    30.528828382492065
    30.582966327667236
    30.7114999294281
    30.748514413833618
    30.779380083084106
    30.839340686798096
    30.87595295906067
    30.930063724517822
    30.971938133239746
    31.002724409103394
    31.12090015411377
    31.180378675460815
    31.233869791030884
    31.281119108200073
    31.335476636886597
    31.402005434036255
    31.455071449279785
    31.49642324447632
    31.556601762771606
    31.587199926376343
    31.62421154975891
    31.67824077606201
    31.719799518585205
    31.761327743530273
    31.813992738723755
    31.849923610687256
    31.891589641571045
    31.957977294921875
    32.0053026676178
    32.13259792327881
    32.18543982505798
    32.22097897529602
    32.2566351890564
    32.31652903556824
    32.33847784996033
    32.37983465194702
    32.42624306678772
    32.49331998825073
    32.546969175338745
    32.58861303329468
    32.629149436950684
    32.66477584838867
    32.7064266204834
    32.766589641571045
    32.81479859352112
    32.85082244873047
    32.88657569885254
    32.912803649902344
    32.93923568725586
    33.014227867126465
    33.06820058822632
    33.099565267562866
    33.135512828826904
    33.17126178741455
    33.23864269256592
    33.28717923164368
    33.31829571723938
    33.35942339897156
    33.40739464759827
    33.44899368286133
    33.577332496643066
    33.65240478515625
    33.688239097595215
    33.73041820526123
    33.813042402267456
    33.922565937042236
    34.03227639198303
    34.06863617897034
    34.11015558242798
    34.14707541465759
    34.18253254890442
    34.21310877799988
    34.244712591171265
    34.2812237739563
    34.32253575325012
    34.35335731506348
    34.38456845283508
    34.460357904434204
    34.487348556518555
    34.518993854522705
    34.567140102386475
    34.65166115760803
    34.69408082962036
    34.74271011352539
    34.76948070526123
    34.86112380027771
    34.92976403236389
    34.96143364906311
    34.9977023601532
    35.02425289154053
    35.055885791778564
    35.09757947921753
    41.74967622756958
    41.78883123397827
    41.83132219314575
    41.87947702407837
    41.902689933776855
    41.933475732803345
    41.96795892715454
    42.00512933731079
    42.08957767486572
    42.127262353897095
    42.16401028633118
    42.20719599723816
    42.2923583984375
    42.33494520187378
    42.403929710388184
    42.460665702819824
    42.484097480773926
    42.58614230155945
    42.62309646606445
    42.68596792221069
    42.79764127731323
    42.83535671234131
    49.300331830978394
    49.33522701263428
    49.371577978134155
    49.41308927536011
    49.48062586784363
    49.51170015335083
    49.6113600730896
    49.67902851104736
    49.70082902908325
    49.77669811248779
    49.81951570510864
    49.87388062477112
    49.915247678756714
    49.97532939910889
    50.08591914176941
    50.11783194541931
    50.178743839263916
    50.215471506118774
    50.26401424407959
    50.301037073135376
    50.337408781051636
    50.3783917427063
    50.42023587226868
    50.46214199066162
    50.510507345199585
    50.57140111923218
    50.60792684555054
    50.64497971534729
    50.69321942329407
    50.74137544631958
    50.77745795249939
    50.80826807022095
    50.83869552612305
    50.913323163986206
    50.974987745285034
    51.001980781555176
    51.269190073013306
    51.30674982070923
    51.33813667297363
    51.415085315704346
    51.47007513046265
    51.49863839149475
    51.56789302825928
    51.59530282020569
    51.64358305931091
    51.72944903373718
    51.771602630615234
    51.81994867324829
    51.86283588409424
    51.92442536354065
    52.13640832901001
    52.19889426231384
    52.22141122817993
    52.28267216682434
    52.45670032501221
    52.50003528594971
    52.56226706504822
    52.77638864517212
    52.846115589141846
    52.884472608566284
    52.97016930580139
    52.99341559410095
    53.0867383480072
    53.13554358482361
    53.19923949241638
    53.2318549156189
    53.27506184577942
    53.302650451660156
    53.335469245910645
    53.430521965026855
    53.486509561538696
    53.523553133010864
    53.55563497543335
    53.591962814331055
    53.66168451309204
    53.71124076843262
    53.78885459899902
    53.83888101577759
    53.87163996696472
    53.95791745185852
    53.98992967605591
    54.12193703651428
    54.1657178401947
    54.228673696517944
    54.255929946899414
    54.29349970817566
    54.33076214790344
    54.38691830635071
    54.4891631603241
    54.55184984207153
    54.579031229019165
    54.62341642379761
    54.6798460483551
    54.717461585998535
    54.76015114784241
    54.79702615737915
    54.83478593826294
    54.87767028808594
    54.93172860145569
    55.016173124313354
    55.07064366340637
    55.11357402801514
    55.16819620132446
    55.229355573654175
    55.270981550216675
    55.32506346702576
    55.36160588264465
    55.41600704193115
    55.45222520828247
    55.50047755241394
    55.568355083465576
    55.61092829704285
    55.64215087890625
    55.67799663543701
    55.718632221221924
    55.92570114135742
    55.96213483810425
    56.04391694068909
    56.118157625198364
    56.14490032196045
    56.18002414703369
    56.20629596710205
    56.280972480773926
    56.312384843826294
    56.395431995391846
    56.46298384666443
    56.504252433776855
    56.5457649230957
    56.576146841049194
    56.61723589897156
    56.69066572189331
    56.72558617591858
    56.77187156677246
    56.83850026130676
    56.875022888183594
    56.91085863113403
    56.9530713558197
    57.00034856796265
    57.09910297393799
    57.12540888786316
    57.172404289245605
    57.25470781326294
    57.295767307281494
    57.355732917785645
    57.42352557182312
    57.52118539810181
    57.574469327926636
    57.61021399497986
    57.68348217010498
    57.724841356277466
    57.905601978302
    57.942599058151245
    58.136271953582764
    58.18305468559265
    58.20896863937378
    58.26355314254761
    58.32467770576477
    58.37313890457153
    58.44031620025635
    58.507779359817505
    58.56853652000427
    58.66877818107605
    58.72278952598572
    58.76386833190918
    58.816935777664185
    58.89296889305115
    58.92425274848938
    59.050923109054565
    59.077515840530396
    59.10985255241394
    59.218446493148804
    59.25479030609131
    59.36257553100586
    59.42300534248352
    62.304723024368286
    62.35291075706482
    62.398810148239136
    62.45125436782837
    62.61779046058655
    62.676573514938354
    62.70282554626465
    62.73786950111389
    62.797707080841064
    62.83417344093323
    62.87543964385986
    62.90634751319885
    63.01289463043213
    63.07205104827881
    63.11229705810547
    63.15929460525513
    63.20001745223999
    63.26610994338989
    63.35603308677673
    63.39119338989258
    63.49685716629028
    63.537575483322144
    63.5679886341095
    63.82537245750427
    63.8618483543396
    63.89240646362305
    63.938953161239624
    64.01241636276245
    64.0488862991333
    64.07084703445435
    64.10169792175293
    64.14352631568909
    64.16909956932068
    64.20461750030518
    64.27793049812317
    64.31327104568481
    64.34346556663513
    64.36932826042175
    64.435964345932
    64.49619936943054
    64.56252002716064
    64.62943363189697
    64.69665908813477
    64.7326352596283
    64.79253339767456
    64.8662486076355
    64.8973171710968
    64.9195065498352
    64.94642901420593
    65.01394128799438
    65.04915738105774
    65.09024310112
    65.22802114486694
    65.26954889297485
    65.3168637752533
    65.36442375183105
    65.42408752441406
    65.49182105064392
    65.52385401725769
    65.57109713554382
    65.64593267440796
    65.68676233291626
    65.72828674316406
    65.75969433784485
    65.80082392692566
    65.83703851699829
    65.8967170715332
    65.97986435890198
    66.0274875164032
    66.0943865776062
    66.14368486404419
    66.18542456626892
    66.21200609207153
    66.32906556129456
    66.39651584625244
    66.44497537612915
    66.54371976852417
    66.59722971916199
    66.687096118927
    66.74774503707886
    66.78441047668457
    66.85927557945251
    66.8911464214325
    66.95157408714294
    67.0052592754364
    67.18752431869507
    67.22427916526794
    67.25510549545288
    67.28148865699768
    67.31224751472473
    67.35960602760315
    67.42175602912903
    67.45864748954773
    67.48437595367432
    67.53776431083679
    67.568776845932
    67.59956121444702
    67.63631010055542
    67.74534916877747
    67.93040347099304
    67.97877740859985
    68.0398313999176
    68.07575988769531
    68.11064076423645
    68.1409285068512
    68.17682981491089
    68.21283316612244
    68.44509172439575
    68.57218527793884
    68.60299634933472
    68.6290352344513
    68.67013716697693
    68.75070595741272
    68.78117990493774
    68.811838388443
    68.90228080749512
    68.99964046478271
    69.05179595947266
    69.1878433227539
    69.22350597381592
    69.32035040855408
    69.34190917015076
    69.3833076953888
    69.40528130531311
    69.48648452758789
    69.51676845550537
    69.54722213745117
    69.5778603553772
    69.63732123374939
    69.66790390014648
    69.749098777771
    69.80151295661926
    69.97137212753296
    70.00306725502014
    70.07772016525269
    70.11888027191162
    70.1406033039093
    70.37235021591187
    70.40910816192627
    70.50209856033325
    70.53383255004883
    70.59413695335388
    70.62477612495422
    70.6556670665741
    70.690842628479
    70.80757927894592
    70.84350609779358
    70.90319848060608
    70.95007634162903
    71.02543091773987
    71.09213781356812
    71.12833571434021
    71.16950726509094
    71.2516860961914
    71.287682056427
    71.31970357894897
    71.36843872070312
    71.56557750701904
    71.60197806358337
    71.63279366493225
    71.67389750480652
    71.70994830131531
    71.75715041160583
    71.83442044258118
    71.89700484275818
    71.93338966369629
    71.95980954170227
    72.07760977745056
    72.27510190010071
    72.33166337013245
    72.39460778236389
    72.4423258304596
    72.53245806694031
    72.57414674758911
    72.6001648902893
    72.67433023452759
    72.72261810302734
    72.7722487449646
    72.8107430934906
    72.84870052337646
    75.41627502441406
    75.53745627403259
    75.60919141769409
    75.6527419090271
    75.70086574554443
    75.74258470535278
    75.77332711219788
    75.82048320770264
    75.86268353462219
    75.94603514671326
    76.02203392982483
    76.05493116378784
    76.1397750377655
    76.18108487129211
    76.22211337089539
    76.30333185195923
    76.43942975997925
    76.4613573551178
    76.50287508964539
    76.57018399238586
    76.60595560073853
    76.67886519432068
    76.73166728019714
    76.77162218093872
    76.80186319351196
    76.8681526184082
    76.90932774543762
    76.9936306476593
    77.03725528717041
    77.07489156723022
    77.10533404350281
    77.13632273674011
    77.18321084976196
    77.23037552833557
    77.28990650177002
    77.3208155632019
    77.37446546554565
    77.40503692626953
    77.45897102355957
    77.5357735157013
    77.61097574234009
    77.65239405632019
    77.68279790878296
    77.74203324317932
    77.80226135253906
    77.83857655525208
    77.90569686889648
    78.90022444725037
    78.95029640197754
    78.99702215194702
    79.07169771194458
    79.10738825798035
    79.14862108230591
    80.84191346168518
    80.87881064414978
    80.90951442718506
    80.95682978630066
    80.98770475387573
    81.04032516479492
    81.08112215995789
    81.11595320701599
    81.15066409111023
    81.18683552742004
    81.21329307556152
    81.24843645095825
    81.28921294212341
    81.34231567382812
    81.4253740310669
    81.4558458328247
    81.52941036224365
    81.65869212150574
    81.68762111663818
    81.72414541244507
    81.79196953773499
    81.8531539440155
    81.89527583122253
    81.93657040596008
    82.02625179290771
    82.08685493469238
    82.14107871055603
    82.17684578895569
    82.23593878746033
    82.2770745754242
    82.33716416358948
    82.36818933486938
    82.40960645675659
    82.45667219161987
    82.70202732086182
    82.74890971183777
    82.79455018043518
    82.83016276359558
    82.86026883125305
    82.94980502128601
    83.14541602134705
    83.17729258537292
    83.21871137619019
    83.29165267944336
    83.33263683319092
    83.65203666687012
    83.69949698448181
    83.7651298046112
    83.82448816299438
    83.85489273071289
    83.8963234424591
    83.9380087852478
    83.98025822639465
    84.08918046951294
    84.12534976005554
    84.15121650695801
    84.17715644836426
    84.2373526096344
    84.29095244407654
    84.3485906124115
    84.37541818618774
    84.41262125968933
    84.44432187080383
    84.49999690055847
    84.53094744682312
    84.57835936546326
    84.64455437660217
    84.67513608932495
    84.83457350730896
    84.86648607254028
    84.90448474884033
    84.97334885597229
    85.02061915397644
    85.06814551353455
    85.1087498664856
    85.14967131614685
    85.19572949409485
    85.23629856109619
    85.27750825881958
    85.33138918876648
    85.37383556365967
    85.4161171913147
    85.49015951156616
    85.52532982826233
    85.57155299186707
    85.66840887069702
    85.74102401733398
    85.79376721382141
    85.84700036048889
    86.00485610961914
    86.0577039718628
    86.10418772697449
    86.19243311882019
    86.22860383987427
    86.33769488334656
    86.39732432365417
    86.42310357093811
    90.50460290908813
    90.54159927368164
    90.56745505332947
    90.61362051963806
    90.67317724227905
    90.75478458404541
    90.80154466629028
    90.83783078193665
    92.94098424911499
    92.97311329841614
    92.99914956092834
    93.04013013839722
    93.12171387672424
    93.14782166481018
    93.24210953712463
    93.28475999832153
    93.53134346008301
    93.56783413887024
    93.62299299240112
    93.65532398223877
    93.86390566825867
    93.92418146133423
    93.95474886894226
    93.98555278778076
    94.01206588745117
    94.0381588935852
    94.0645227432251
    94.12772178649902
    94.17086458206177
    94.20201206207275
    94.23314952850342
    94.28029203414917
    94.31110191345215
    94.37840580940247
    94.42598867416382
    94.52488422393799
    94.57357788085938
    94.65157723426819
    94.69323468208313
    94.72426223754883
    94.75073170661926
    94.77281308174133
    94.84793663024902
    94.88964319229126
    94.93079710006714
    94.96149802207947
    94.98770236968994
    95.04943180084229
    95.10604047775269
    95.14760732650757
    95.21465182304382
    95.26822686195374
    95.30996227264404
    95.35168218612671
    95.39944911003113
    95.43061137199402
    95.4785692691803
    95.51592302322388
    95.59285473823547
    95.65335297584534
    95.67940258979797
    95.77047228813171
    95.80145335197449
    95.83247399330139
    95.92338514328003
    95.96746802330017
    96.03069519996643
    96.06153297424316
    96.12151789665222
    96.16291284561157
    96.20426201820374
    96.27179789543152
    96.31898379325867
    96.35016989707947
    96.39847993850708
    96.46894717216492
    96.50634217262268
    96.57259345054626
    96.64637970924377
    96.68233299255371
    96.70881605148315
    96.76156044006348
    96.7921416759491
    96.83932614326477
    96.91582107543945
    96.95887160301208
    97.04093861579895
    97.07576084136963
    97.11065459251404
    97.15120577812195
    97.24081707000732
    97.3223967552185
    97.36992526054382
    97.40679383277893
    97.4603853225708
    97.49077367782593
    97.52585625648499
    97.56114339828491
    97.6428759098053
    97.67347192764282
    97.70355486869812
    97.74385976791382
    97.79040670394897
    97.82586097717285
    97.86142683029175
    97.92040467262268
    97.9557843208313
    97.98158383369446
    98.00728344917297
    98.04240417480469
    98.07783079147339
    98.11866617202759
    98.15453219413757
    98.1954996585846
    98.23771810531616
    98.42042231559753
    98.46189427375793
    98.50248146057129
    98.54329824447632
    98.60216498374939
    98.6689600944519
    98.71109795570374
    98.73900246620178
    98.80626630783081
    98.853280544281
    98.92076587677002
    98.96772384643555
    99.00370001792908
    99.19174861907959
    99.22497773170471
    99.39604663848877
    99.48445653915405
    99.66726350784302
    99.69471883773804
    99.71626400947571
    99.74644899368286
    99.79862236976624
    99.83439016342163
    99.87534737586975
    99.91705894470215
    99.95869159698486
    99.99938893318176
    100.22250938415527
    100.27055406570435
    100.33009958267212
    100.3562080860138
    100.38264298439026
    100.46526551246643
    100.55840349197388
    100.58158254623413
    100.69072818756104
    100.73777389526367
    100.81230473518372
    100.89409327507019
    100.91693115234375
    100.95363306999207
    100.9845769405365
    101.00612878799438
    101.03201961517334
    101.06780576705933
    101.1091046333313
    101.20816469192505
    101.25594806671143
    101.30971240997314
    101.53028988838196
    101.6331262588501
    101.66954636573792
    101.71059536933899
    101.78463220596313
    101.8157286643982
    101.92369961738586
    102.05581092834473
    102.09249377250671
    102.22832727432251
    102.26348280906677
    102.29933476448059
    102.34021806716919
    102.38708996772766
    102.4486677646637
    102.50327110290527
    102.57762885093689
    102.61308979988098
    102.66569256782532
    102.70738577842712
    102.75442910194397
    102.78507232666016
    102.81548261642456
    102.97817492485046
    103.01979613304138
    103.06033825874329
    103.08643388748169
    103.12752366065979
    103.16815400123596
    103.19443893432617
    103.22461581230164
    103.25965332984924
    103.31930136680603
    103.43139886856079
    103.47279858589172
    103.65360260009766
    103.70719695091248
    103.73745441436768
    103.77240085601807
    103.81873607635498
    103.85945081710815
    103.89083099365234
    103.94451236724854
    104.02584934234619
    104.14163780212402
    104.2311601638794
    104.26709771156311
    104.31601977348328
    104.36507892608643
    104.42459654808044
    104.47138905525208
    104.5074303150177
    104.60593104362488
    104.6876266002655
    104.72335529327393
    104.78349995613098
    104.88372993469238
    104.91942477226257
    104.95497727394104
    104.99597024917603
    105.01750183105469
    105.09244322776794
    105.13967251777649
    105.19285154342651
    105.22510004043579
    105.2811131477356
    105.31278133392334
    105.34343218803406
    105.37403297424316
    105.40487432479858
    105.47877740859985
    105.56078505516052
    105.5967276096344
    105.62319707870483
    105.67054486274719
    105.70253777503967
    105.72560358047485
    105.84471917152405
    105.87588167190552
    105.92933464050293
    106.00279784202576
    106.04362034797668
    106.1354672908783
    106.16313338279724
    106.2181670665741
    106.24900794029236
    106.28428769111633
    106.31946015357971
    106.38624358177185
    106.43934798240662
    106.48044466972351
    106.5398530960083
    106.57093691825867
    106.63322901725769
    106.67668294906616
    106.73004269599915
    106.76097846031189
    107.01864862442017
    107.06675267219543
    107.09908771514893
    107.1259069442749
    107.22371649742126
    107.2768702507019
    107.55119442939758
    107.60671257972717
    107.79014444351196
    107.8166856765747
    107.86357593536377
    107.92312097549438
    107.95924878120422
    108.12022757530212
    108.17441582679749
    108.22754454612732
    108.2877836227417
    108.33592367172241
    108.37756156921387
    108.4786434173584
    108.72792363166809
    108.76077461242676
    108.8984489440918
    108.94292116165161
    108.97097587585449
    109.04759669303894
    109.08363747596741
    109.17444348335266
    109.22784447669983
    109.28143525123596
    109.32865166664124
    109.36509609222412
    109.40391874313354
    109.43698501586914
    109.49738907814026
    109.53334093093872
    109.669429063797
    109.7053017616272
    109.8321590423584
    109.88839435577393
    109.93757009506226
    109.9689929485321
    110.00511145591736
    110.03158521652222
    110.13180017471313
    110.16817903518677
    110.24320888519287
    110.28558611869812
    110.36490082740784
    110.41311836242676
    110.48715853691101
    110.52274441719055
    110.60459208488464
    110.72072792053223
    110.74320650100708
    110.78026080131531
    110.87103152275085
    110.94448614120483
    110.97986817359924
    111.01509046554565
    111.13027811050415
    111.18259525299072
    111.21879386901855
    111.27854037284851
    111.36779761314392
    111.44034767150879
    111.4864296913147
    111.51653361320496
    111.54643702507019
    111.58732223510742
    111.63946485519409
    111.71530342102051
    111.74717020988464
    111.88210344314575
    111.93558502197266
    111.98224139213562
    112.07228994369507
    112.12609910964966
    112.15736985206604
    112.1998553276062
    113.07813000679016
    113.13492155075073
    113.18210458755493
    113.2178213596344
    113.31770277023315
    113.36522674560547
    113.48197531700134
    113.52447295188904
    113.56757593154907
    113.60508322715759
    113.72254586219788
    113.79004502296448
    113.81607747077942
    113.85795164108276
    113.88889813423157
    113.91986393928528
    113.96185851097107
    113.98437094688416
    114.02861952781677
    114.06623768806458
    114.1263062953949
    114.1481614112854
    114.40888595581055
    114.47882151603699
    114.52758646011353
    117.01759576797485
    117.07242894172668
    117.11345362663269
    117.1600661277771
    117.2075707912445
    117.23427987098694
    117.29724860191345
    117.33940100669861
    117.3749053478241
    117.40039920806885
    117.42596220970154
    117.45222568511963
    117.55979514122009
    117.59048914909363
    117.63170623779297
    117.7164511680603
    117.74988555908203
    117.79141449928284
    117.83825039863586
    117.9211778640747
    117.97551321983337
    118.05038738250732
    118.08647751808167
    118.2748498916626
    118.36638069152832
    118.39264678955078
    118.52997612953186
    118.59115743637085
    118.70428943634033
    118.73655033111572
    118.77855205535889
    118.82588458061218
    118.86702847480774
    118.89792203903198
    118.93950843811035
    119.02145195007324
    119.04838609695435
    119.1031744480133
    119.14028239250183
    119.2490758895874
    119.28471302986145
    119.3514928817749
    119.41845703125
    119.48657703399658
    119.52958345413208
    119.56190252304077
    119.59445333480835
    119.63740420341492
    119.7050530910492
    119.74669027328491
    119.78280377388
    119.80867767333984
    119.85600757598877
    119.89757776260376
    119.95126914978027
    120.11519527435303
    120.16350483894348
    120.20490050315857
    120.25259947776794
    120.31861209869385
    120.36447143554688
    120.40551161766052
    120.49534010887146
    120.54365348815918
    120.56956958770752
    120.60486960411072
    120.63141679763794
    120.66695523262024
    120.74247169494629
    120.77808666229248
    120.808673620224
    120.89214825630188
    120.92343521118164
    120.96511816978455
    120.99179720878601
    121.0338397026062
    121.07501173019409
    121.12144422531128
    121.16801714897156
    121.21397233009338
    121.2489824295044
    121.28955459594727
    121.3555953502655
    121.39791989326477
    121.42970323562622
    121.47215414047241
    121.58208417892456
    121.64241671562195
    121.67814517021179
    121.72466254234314
    121.76545357704163
    121.84797358512878
    121.87957215309143
    121.92762017250061
    122.0276288986206
    122.06962323188782
    122.10626173019409
    122.15323901176453
    122.30155730247498
    122.36439609527588
    122.39610981941223
    122.4319384098053
    122.4916889667511
    122.55237913131714
    122.58345627784729
    122.63069009780884
    122.66199541091919
    122.69745802879333
    122.80751156806946
    122.94957613945007
    122.97616076469421
    123.00233674049377
    123.03384184837341
    123.0754508972168
    123.14321970939636
    123.18476295471191
    123.25356221199036
    123.28721070289612
    123.33041095733643
    123.36165189743042
    123.39248180389404
    123.42299699783325
    123.46477580070496
    123.54721093177795
    123.58322334289551
    123.62510657310486
    123.67888498306274
    123.75494766235352
    123.79842495918274
    123.83095526695251
    123.87222623825073
    123.9077718257904
    123.94854736328125
    123.98968601226807
    124.02070188522339
    124.06212282180786
    124.12269711494446
    124.17652559280396
    124.30838322639465
    124.38275694847107
    124.44944214820862
    124.5090560913086
    124.59078979492188
    124.617116689682
    124.65296196937561
    124.77061080932617
    124.8066565990448
    124.84224009513855
    124.90145325660706
    124.93209195137024
    125.25903487205505
    125.30068588256836
    125.42569589614868
    125.4511866569519
    125.49206185340881
    125.5380916595459
    125.62738370895386
    125.68007922172546
    125.80481719970703
    125.85770654678345
    125.9392397403717
    125.97498512268066
    126.14493703842163
    126.20512533187866
    126.25808787345886
    126.2934021949768
    126.32862854003906
    186.67356514930725
    186.73402547836304
    186.83218836784363
    186.94279861450195
    186.97098660469055
    187.0897922515869
    187.1207299232483
    187.1566903591156
    187.21084713935852
    187.25156044960022
    187.35112929344177
    187.41553735733032
    187.47819566726685
    187.5312533378601
    187.62087965011597
    187.6679346561432
    187.72090911865234
    187.7564766407013
    187.8311414718628
    187.8582010269165
    187.8960084915161
    187.92279982566833
    187.94478702545166
    187.99778723716736
    188.03338074684143
    188.0740921497345
    188.12761044502258
    188.174001455307
    188.22070932388306
    188.319757938385
    188.37438583374023
    188.41016030311584
    188.45635151863098
    188.49107456207275
    188.5719985961914
    188.6023919582367
    188.63758039474487
    188.69694328308105
    188.7639217376709
    188.8316674232483
    188.8921995162964
    188.95180320739746
    189.0049068927765
    189.08552622795105
    189.15053486824036
    189.17609667778015
    189.24294686317444
    189.2741949558258
    189.34107947349548
    189.37626910209656
    189.4345052242279
    189.45956087112427
    189.5243215560913
    189.60414218902588
    189.65646624565125
    189.68679666519165
    189.76697945594788
    189.8029556274414
    189.85566186904907
    189.88614892959595
    189.91645121574402
    189.98164534568787
    190.01674628257751
    190.0468716621399
    190.08728861808777
    190.1409797668457
    190.1827256679535
    190.22971105575562
    190.26458072662354
    191.6473731994629
    191.71431469917297
    191.76043796539307
    191.81991338729858
    191.85075974464417
    191.8872995376587
    191.92851972579956
    191.98389101028442
    192.0952558517456
    192.1488118171692
    192.19547724723816
    192.24272680282593
    192.26870846748352
    192.29493236541748
    192.36092519760132
    192.41488647460938
    192.49176168441772
    192.52798509597778
    192.57463192939758
    192.62160348892212
    192.7804515361786
    192.82129096984863
    192.8629069328308
    192.88901615142822
    192.9258120059967
    192.96237349510193
    193.01531648635864
    193.16016578674316
    193.20065784454346
    193.23157596588135
    193.30520844459534
    193.3589129447937
    193.41251707077026
    193.4534306526184
    193.52913784980774
    193.5641450881958
    193.59460473060608
    193.64730405807495
    193.68342566490173
    193.71862888336182
    193.77884531021118
    193.84777903556824
    193.8753001689911
    193.94309282302856
    194.02558159828186
    194.1008644104004
    194.14238715171814
    194.17339372634888
    194.2336287498474
    194.2814211845398
    194.32320141792297
    194.3448190689087
    194.443350315094
    194.48430132865906
    194.54396653175354
    194.6257462501526
    194.65174078941345
    194.69917511940002
    194.77544045448303
    194.84393048286438
    194.8803551197052
    197.4892144203186
    197.51705861091614
    197.57641577720642
    197.6226999759674
    197.81542825698853
    197.86850118637085
    197.92149543762207
    197.99672102928162
    198.04470086097717
    198.1709544658661
    198.22379350662231
    198.2599003314972
    198.29040956497192
    198.32610487937927
    198.35765743255615
    198.40027356147766
    198.44966745376587
    198.4814214706421
    198.51253461837769
    198.5431432723999
    198.63339948654175
    198.6872251033783
    198.72889304161072
    198.75478601455688
    198.78534579277039
    198.81262302398682
    198.83543133735657
    198.87837600708008
    198.92695426940918
    198.95755887031555
    198.98838472366333
    215.54189443588257
    215.58469414710999
    215.6196448802948
    215.67209601402283
    215.7310667037964
    215.81436014175415
    215.84139728546143
    215.88854789733887
    215.9544816017151
    217.60569953918457
    217.62845396995544
    217.88813519477844
    218.0464222431183
    218.09421920776367
    218.125470161438
    218.2073414325714
    218.265474319458
    218.30603885650635
    218.34122133255005
    218.3868248462677
    218.4123456478119
    218.45865297317505
    218.49911975860596
    348.565265417099
    348.6024270057678
    352.54056787490845
    352.5780363082886
    352.60470175743103
    352.64640283584595
    352.71344804763794
    352.73995757102966
    352.7822775840759
    352.83556962013245
    352.8890264034271
    352.92546701431274
    352.96120953559875
    353.0153913497925
    353.0466537475586
    353.0826413631439
    353.1189053058624
    353.2021143436432
    353.2442181110382
    353.2707452774048
    353.2969720363617
    353.47857332229614
    353.5099022388458
    353.5514328479767
    353.58801007270813
    353.6234459877014
    353.72316098213196
    353.76481199264526
    353.80659675598145
    353.8670959472656
    353.9416546821594
    353.983434677124
    354.04400515556335
    354.1047945022583
    354.16567611694336
    354.19720339775085
    354.3343777656555
    354.3653476238251
    354.4129726886749
    354.44922733306885
    354.47126936912537
    354.5327217578888
    354.5696783065796
    354.596275806427
    354.62821412086487
    354.6650037765503
    354.6871771812439
    354.7283570766449
    354.7825107574463
    354.8186147212982
    354.8604280948639
    354.8916540145874
    354.93334341049194
    354.9690845012665
    355.0228819847107
    355.0491976737976
    355.1029567718506
    355.1560847759247
    355.19707584381104
    355.2495279312134
    355.2800977230072
    355.32684206962585
    355.3572781085968
    355.4103374481201
    355.4458329677582
    355.58203864097595
    355.62351059913635
    355.68346309661865
    355.71910429000854
    355.801066160202
    355.83218145370483
    355.87386536598206
    355.9406430721283
    355.9765090942383
    356.0242669582367
    356.0600402355194
    388.33009004592896
    388.41785955429077
    388.4786138534546
    388.52529788017273
    388.5475368499756
    388.707097530365
    388.7909517288208
    456.7313940525055
    456.7978529930115
    456.8514757156372
    456.88730096817017
    456.96230840682983
    457.0151665210724
    457.06827664375305
    457.2063982486725
    457.2374289035797
    457.28434777259827
    457.331750869751
    457.406046628952
    457.5546839237213
    457.5909652709961
    457.62755942344666
    457.66903281211853
    457.7102394104004
    457.77768206596375
    457.8256251811981
    457.86753249168396
    457.89882254600525
    458.0299837589264
    458.0840084552765
    458.1201899051666
    458.2493760585785
    458.35655999183655
    458.41588044166565
    458.48227739334106
    458.5555167198181
    458.5812566280365
    458.6482105255127
    458.77476596832275
    458.8911962509155
    458.9325382709503
    458.95418977737427
    459.0018274784088
    459.0427849292755
    459.0891418457031
    459.12472772598267
    459.19173407554626
    459.21803760528564
    459.271906375885
    459.2978513240814
    459.3335826396942
    459.36420798301697
    459.41732001304626
    459.4530827999115
    459.4839918613434
    459.5303237438202
    459.5566942691803
    459.5875382423401
    459.6341724395752
    459.660249710083
    459.7129406929016
    459.7538893222809
    459.79524779319763
    459.82578778266907
    459.8924400806427
    459.9596977233887
    460.0264935493469
    460.1630187034607
    460.1991446018219
    460.2732915878296
    460.30372500419617
    460.38463711738586
    460.437801361084
    460.4733262062073
    460.4948868751526
    460.55053901672363
    460.59210753440857
    460.62798523902893
    460.70929312705994
    460.9660978317261
    461.00214409828186
    461.0561273097992
    461.1087770462036
    461.19920682907104
    461.23506236076355
    461.2873592376709
    461.3339126110077
    461.4069573879242
    461.46593737602234
    461.51840591430664
    461.5398151874542
    461.5987296104431
    461.72321105003357
    461.76962995529175
    461.82843470573425
    461.87535762786865
    461.90573716163635
    462.1251654624939
    462.1921410560608
    462.2393171787262
    462.3224310874939
    462.3491725921631
    462.39075231552124
    462.4325611591339
    462.48586797714233
    462.5215799808502
    462.54765367507935
    462.57839131355286
    462.64495611190796
    462.7698771953583
    462.8233160972595
    462.8587369918823
    462.8992145061493
    462.95144629478455
    463.0404191017151
    463.09994864463806
    463.15326619148254
    463.19984006881714
    463.25982213020325
    463.3340108394623
    463.38740372657776
    463.41824889183044
    463.4599618911743
    463.4963388442993
    463.5629382133484
    463.5988235473633
    463.64598536491394
    463.68178129196167
    463.71799659729004
    463.74407982826233
    463.7702202796936
    463.8302540779114
    464.03655791282654
    464.09035778045654
    464.1164791584015
    464.14745259284973
    464.2217471599579
    464.2482421398163
    464.31550431251526
    464.3820822238922
    464.44865441322327
    464.4897699356079
    464.5118992328644
    464.57194542884827
    464.6313545703888
    464.697984457016
    464.74247002601624
    464.8124256134033
    464.88487815856934
    464.9287917613983
    464.9821774959564
    465.0280432701111
    465.18579936027527
    465.24710392951965
    465.7403521537781
    465.8154900074005
    465.85178995132446
    465.91831517219543
    465.97710037231445
    466.0193524360657
    466.1324071884155
    466.32087564468384
    466.35661244392395
    466.41286611557007
    466.4553008079529
    466.5113754272461
    466.59355640411377
    466.68803429603577
    466.73639369010925
    466.7922902107239
    466.8224632740021
    466.8786368370056
    466.92647099494934
    466.9614100456238
    467.0092980861664
    467.0901379585266
    467.1317365169525
    467.172815322876
    467.2207806110382
    467.2619745731354
    467.3174743652344
    467.3879256248474
    467.43580651283264
    467.5457308292389
    467.60759377479553
    467.6544449329376
    467.7081604003906
    467.7545349597931
    467.86154341697693
    467.90835666656494
    467.9424352645874
    468.00465965270996
    468.0453362464905
    468.10077714920044
    468.1487064361572
    468.22816801071167
    468.46818017959595
    468.56649708747864
    468.6209716796875
    468.68463826179504
    468.7401807308197
    468.79450249671936
    468.8289415836334
    468.89913606643677
    468.97007846832275
    469.0251579284668
    469.3155982494354
    469.35704135894775
    469.4654495716095
    469.5132567882538
    469.55443382263184
    469.61543893814087
    469.76486587524414
    469.8107178211212
    469.8510322570801
    469.96705770492554
    470.02780842781067
    470.0806567668915
    470.11483097076416
    470.14814281463623
    470.21641421318054
    470.2637484073639
    470.3030273914337
    470.3485863208771
    470.4336097240448
    470.48732018470764
    470.5273542404175
    470.6332585811615
    470.6799750328064
    470.72018241882324
    470.7892904281616
    470.8288793563843
    470.8889973163605
    470.91750931739807
    471.0243353843689
    471.1635603904724
    471.28059935569763
    471.3092665672302
    471.36261796951294
    471.40965938568115
    471.44331431388855
    471.56048011779785
    471.6006052494049
    471.958025932312
    472.0273714065552
    472.1229009628296
    472.15113830566406
    472.19748544692993
    472.2310199737549
    472.2773699760437
    472.3231499195099
    472.46198892593384
    472.4905250072479
    472.53011322021484
    472.5824484825134
    472.6874690055847
    472.71796321868896
    472.77731704711914
    472.83603620529175
    472.90279722213745
    472.9361979961395
    472.9950850009918
    473.0617706775665
    473.10666513442993
    473.17395639419556
    473.2258515357971
    473.2709972858429
    473.30854177474976
    473.37276911735535
    473.6246495246887
    479.36902022361755
    479.41037225723267
    479.53041315078735
    479.57007598876953
    479.6103582382202
    479.7011561393738
    479.7467601299286
    479.79319953918457
    479.8399395942688
    479.96359276771545
    480.0049433708191
    480.05302357673645
    480.1480724811554
    480.26087284088135
    480.3012430667877
    480.3545346260071
    480.42988085746765
    480.4972665309906
    480.54422068595886
    480.60479974746704
    480.6514501571655
    480.6981794834137
    480.72764325141907
    480.78184032440186
    480.9533200263977
    480.9833092689514
    481.06041073799133
    481.1291136741638
    481.19012999534607
    481.22531294822693
    481.3034076690674
    481.3332681655884
    481.37511110305786
    481.4109025001526
    481.505243062973
    481.5597403049469
    481.62096762657166
    481.6565799713135
    481.70427918434143
    481.74021768569946
    481.78203868865967
    481.8293299674988
    481.86861658096313
    481.9214210510254
    482.0348436832428
    482.21573424339294
    482.25145983695984
    482.31196880340576
    482.3722610473633
    482.4568283557892
    482.4920208454132
    482.527019739151
    482.5622229576111
    482.622385263443
    482.8157057762146
    482.85059666633606
    482.91908979415894
    482.95419430732727
    483.01525235176086
    483.10829639434814
    485.36187505722046
    485.4311809539795
    485.4842338562012
    485.5311634540558
    485.5848181247711
    485.61935806274414
    485.6594159603119
    485.6936914920807
    485.7608082294464
    485.789705991745
    485.90080189704895
    485.93531703948975
    485.95992064476013
    486.01196241378784
    486.0712676048279
    486.1002736091614
    486.14707589149475
    486.186975479126
    486.2329308986664
    486.27904057502747
    486.31914591789246
    486.36013293266296
    486.4065270423889
    486.4467589855194
    486.5058960914612
    486.5517461299896
    486.6051344871521
    486.6722266674042
    486.73903632164
    486.79164481163025
    486.82673478126526
    486.8731918334961
    486.91994047164917
    486.9660949707031
    486.99543380737305
    487.04263639450073
    487.08284735679626
    487.117240190506
    487.14659881591797
    487.2293405532837
    487.2546410560608
    487.32270765304565
    487.423965215683
    487.45330238342285
    487.5070343017578
    487.5602927207947
    487.64517188072205
    487.6994159221649
    487.8101978302002
    487.85041308403015
    487.8958022594452
    488.0050106048584
    488.05087518692017
    488.1028971672058
    488.1481628417969
    488.19317603111267
    488.2448480129242
    488.28449177742004
    488.37472009658813
    488.44815039634705
    488.48223900794983
    488.52776765823364
    488.57943177223206
    488.6128933429718
    488.6463141441345
    488.69175839424133
    488.9615602493286
    488.9958176612854
    489.0346667766571
    489.0796010494232
    489.16094422340393
    489.65219354629517
    489.69052147865295
    489.74707078933716
    489.78507018089294
    489.83484840393066
    489.8790326118469
    489.91170382499695
    489.9496991634369
    489.99911546707153
    490.0780861377716
    490.14929699897766
    490.20576763153076
    490.2561774253845
    490.27941513061523
    490.34270453453064
    490.37625551223755
    490.42055201530457
    490.44844365119934
    490.50494146347046
    490.5551800727844
    492.7210147380829
    492.78009247779846
    492.84374952316284
    492.8872947692871
    492.94443106651306
    493.0595555305481
    493.09263706207275
    493.2071707248688
    493.2403070926666
    493.39894437789917
    493.4278039932251
    493.471312046051
    493.54950308799744
    493.6054127216339
    496.6369137763977
    496.67912888526917
    496.7153458595276
    496.8445358276367
    496.91274642944336
    496.96105670928955
    496.99714970588684
    497.0477924346924
    497.08483386039734
    497.1708698272705
    497.2082214355469
    497.2349178791046
    497.30439352989197
    497.35953402519226
    497.38657784461975
    497.42343306541443
    497.47704124450684
    497.5128364562988
    497.55442929267883
    497.6449010372162
    497.697701215744
    497.73303842544556
    497.76828384399414
    497.82114458084106
    497.88109159469604
    497.9338746070862
    497.98121190071106
    498.03470158576965
    498.06531858444214
    498.1068081855774
    498.26593375205994
    498.3133189678192
    498.345006942749
    498.4189155101776
    498.4719352722168
    498.50722670555115
    498.5605764389038
    498.60208654403687
    498.6492660045624
    498.67558431625366
    498.7579219341278
    498.7893362045288
    498.8306427001953
    498.86649227142334
    498.9261977672577
    498.97318840026855
    499.05523228645325
    499.10889744758606
    499.1560616493225
    499.4451034069061
    499.57315850257874
    499.61467146873474
    499.66910552978516
    499.7005763053894
    499.78326892852783
    499.83015418052673
    499.897762298584
    499.951340675354
    500.0500214099884
    500.0805604457855
    500.1632888317108
    500.2157220840454
    500.25055384635925
    500.3035886287689
    500.3499159812927
    500.39692306518555
    500.449747800827
    500.4798185825348
    500.5322651863098
    500.5673727989197
    500.6142122745514
    500.8185589313507
    501.00903511047363
    501.0557060241699
    501.10235929489136
    501.13276052474976
    501.1586859226227
    504.05407977104187
    504.09057545661926
    504.1568281650543
    504.1871728897095
    504.23353910446167
    504.28626132011414
    504.3272457122803
    504.34885454177856
    504.4145016670227
    504.43985056877136
    504.5887713432312
    504.615257024765
    504.64615845680237
    504.6819429397583
    504.75609135627747
    504.79185009002686
    504.833110332489
    505.01329255104065
    505.07983684539795
    505.12043857574463
    505.1730971336365
    505.2258574962616
    505.2998459339142
    505.3464319705963
    505.39908266067505
    507.34228134155273
    507.4027395248413
    507.4548375606537
    507.4767029285431
    507.5363473892212
    507.5776090621948
    507.6678075790405
    507.70945858955383
    507.75102972984314
    507.78180980682373
    507.8354949951172
    509.27922892570496
    509.3464345932007
    509.3762414455414
    509.42253017425537
    509.4573097229004
    509.5303432941437
    509.56587052345276
    509.836186170578
    509.87719559669495
    509.90297293663025
    509.9435601234436
    510.0240397453308
    510.05480337142944
    510.13581705093384
    510.1824781894684
    510.2177290916443
    510.2535376548767
    510.3355875015259
    510.37598729133606
    510.416419506073
    510.4761629104614
    510.58454871177673
    512.0019285678864
    512.0453431606293
    512.1064021587372
    512.199248790741
    512.2365472316742
    512.2790246009827
    512.3158824443817
    512.4093651771545
    512.458366394043
    512.5199496746063
    512.5562343597412
    512.5823729038239
    512.6134519577026
    512.6606137752533
    512.7079153060913
    512.782897233963
    512.818825006485
    512.8661386966705
    512.9076397418976
    513.1302149295807
    513.1779520511627
    513.2138164043427
    513.2495744228363
    513.2967839241028
    513.3381187915802
    513.3795771598816
    513.420746088028
    513.4511260986328
    513.504369020462
    513.5303435325623
    513.5773918628693
    513.6082203388214
    513.6437900066376
    513.8093156814575
    513.9455881118774
    514.1396336555481
    514.1929306983948
    514.2234737873077
    514.2490427494049
    514.3006806373596
    514.3405647277832
    514.3752725124359
    514.4105134010315
    514.6136646270752
    514.6661076545715
    514.6876745223999
    514.7181348800659
    514.7839324474335
    514.8240096569061
    514.8591365814209
    514.9243495464325
    514.9651470184326
    515.0238726139069
    515.049439907074
    515.1466701030731
    515.2270743846893
    515.3356084823608
    515.3947591781616
    515.440703868866
    515.4664807319641
    515.4921538829803
    515.5444991588593
    515.5849759578705
    515.6313660144806
    515.6719701290131
    515.7371826171875
    517.7907266616821
    517.8386952877045
    517.8793544769287
    517.9143404960632
    517.9602015018463
    518.0004315376282
    518.0407552719116
    518.0754852294922
    518.1101942062378
    518.1900322437286
    518.2557351589203
    518.307895898819
    518.3430299758911
    518.373025894165
    518.4134364128113
    518.4347009658813
    518.465145111084
    518.5002565383911
    518.5592255592346
    518.6117472648621
    518.6522023677826
    518.6826229095459
    518.8596451282501
    518.9009299278259
    518.9669871330261
    518.9973664283752
    519.0279030799866
    519.0629394054413
    519.2178058624268
    519.2639474868774
    519.3161377906799
    519.3565311431885
    519.4157013893127
    519.4563553333282
    519.4870226383209
    519.5333013534546
    519.5793192386627
    519.6006593704224
    519.6264600753784
    519.6737804412842
    519.7040679454803
    519.7561542987823
    519.7865836620331
    530.097599029541
    530.2444767951965
    530.334136724472
    530.3694903850555
    530.4158918857574
    530.4894833564758
    530.5249221324921
    530.5841617584229
    530.6194581985474
    530.6546785831451
    530.7348999977112
    530.7866859436035
    530.8171124458313
    530.8629837036133
    530.9086661338806
    530.9544847011566
    531.0061128139496
    531.0939249992371
    531.1196496486664
    531.1496176719666
    531.1951775550842
    531.3106877803802
    531.3362958431244
    531.3713400363922
    531.4235138893127
    531.5194344520569
    531.5656492710114
    531.6006655693054
    531.6357581615448
    531.7083406448364
    531.7432396411896
    531.7688419818878
    531.8039774894714
    531.8392381668091
    531.9047040939331
    532.0287823677063
    532.0591771602631
    532.1115500926971
    532.1918435096741
    532.297349691391
    532.327821969986
    532.3578202724457
    532.4462263584137
    532.4919543266296
    532.5225324630737
    532.5813066959381
    532.621738910675
    532.6802217960358
    532.7205431461334
    532.7614462375641
    532.8017868995667
    532.8419878482819
    532.872394323349
    532.9180109500885
    532.976568698883
    533.0170521736145
    533.1319034099579
    533.1621460914612
    533.2147541046143
    533.2500092983246
    533.2961957454681
    533.318119764328
    533.3642852306366
    533.4104039669037
    533.4718523025513
    533.5025036334991
    533.5328199863434
    533.7341465950012
    533.8146142959595
    533.8550789356232
    533.9011852741241
    533.9365723133087
    533.9775395393372
    534.0182085037231
    534.0482280254364
    534.0786821842194
    534.1308388710022
    534.2202639579773
    534.2791924476624
    534.534551858902
    534.5752918720245
    534.6104063987732
    534.6462800502777
    534.6869101524353
    534.7680585384369
    534.8089182376862
    534.8551602363586
    534.9074227809906
    534.9482464790344
    534.9891948699951
    535.0419735908508
    535.1077115535736
    535.1541502475739
    535.1949381828308
    535.2306888103485
    535.2965979576111
    535.318422794342
    535.3590712547302
    535.4401421546936
    535.4925637245178
    535.5279412269592
    535.563154220581
    535.5889177322388
    535.6145436763763
    540.5118637084961
    540.5864689350128
    540.6386346817017
    540.7283084392548
    540.8013424873352
    540.8604836463928
    540.9067163467407
    540.9530317783356
    541.0117290019989
    541.0638294219971
    541.0891354084015
    541.1241900920868
    541.1836276054382
    541.2192559242249
    541.2545943260193
    541.2848241329193
    541.325159072876
    541.377598285675
    541.5331566333771
    541.5738310813904
    541.8319003582001
    541.8728451728821
    541.9032485485077
    542.0183975696564
    542.0489275455475
    542.1012287139893
    542.1366240978241
    542.177549123764
    542.3138496875763
    542.3401920795441
    542.3707990646362
    542.461179971695
    542.4871945381165
    542.5131850242615
    542.554450750351
    542.6281571388245
    542.6692006587982
    542.7098896503448
    542.7907471656799
    543.0471241474152
    543.0941004753113
    543.1462445259094
    543.1762976646423
    543.2064254283905
    543.2525207996368
    543.42968583107
    543.4512972831726
    543.472738981247
    543.5316474437714
    543.5718507766724
    543.6070656776428
    543.6482625007629
    543.6835329532623
    543.7355916500092
    543.7704720497131
    546.1241345405579
    546.1670818328857
    546.2403888702393
    546.2926149368286
    546.3227801322937
    546.3537104129791
    546.4192295074463
    546.4777729511261
    546.5184562206268
    546.5535504817963
    546.6423692703247
    546.6951780319214
    546.7411258220673
    546.7759613990784
    546.8161628246307
    546.8572990894318
    546.9092516899109
    546.9494524002075
    546.9947838783264
    547.0401413440704
    547.0860576629639
    547.2096934318542
    547.2351016998291
    547.3494126796722
    547.4212200641632
    547.4610059261322
    547.5067579746246
    547.5591909885406
    547.6838607788086
    547.7646021842957
    547.7864940166473
    547.8118686676025
    547.8417701721191
    547.8824877738953
    547.9417860507965
    547.9830343723297
    548.0360763072968
    548.061671257019
    548.1203064918518
    548.3383877277374
    548.3979704380035
    548.443932056427
    548.7446172237396
    548.7750351428986
    548.8054616451263
    548.8582139015198
    548.8988864421844
    551.1706736087799
    551.223183631897
    551.2483718395233
    551.2831609249115
    551.3289647102356
    551.3948831558228
    553.7086398601532
    553.7409150600433
    553.7822909355164
    553.8234515190125
    553.870941400528
    553.9019811153412
    553.9493761062622
    554.0029559135437
    554.029027223587
    554.0598976612091
    554.0907065868378
    554.126601934433
    554.2009303569794
    554.2269294261932
    554.3334333896637
    554.4004082679749
    554.4471673965454
    558.9806413650513
    559.0620641708374
    559.1085114479065
    559.1384470462799
    559.1734499931335
    559.2197856903076
    559.2497158050537
    559.3387832641602
    559.3790202140808
    586.9065194129944
    587.0767407417297
    587.1126861572266
    587.1486041545868
    587.2014701366425
    590.4692342281342
    590.5060563087463
    590.5712869167328
    590.6064145565033
    590.6517541408539
    590.717164516449
    590.7572689056396
    590.797073841095
    590.831701040268
    590.8663120269775
    590.8913791179657
    590.9796931743622
    591.019727230072
    591.0776560306549
    591.1503691673279
    591.1716477870941
    591.2071561813354
    591.2372241020203
    591.2775013446808
    591.5816504955292
    591.6172406673431
    591.676656961441
    591.7501900196075
    591.7973465919495
    591.8383059501648
    591.8737192153931
    591.9041874408722
    591.9504272937775
    592.2084658145905
    592.2488305568695
    592.2838079929352
    592.3495490550995
    592.3897061347961
    592.4363262653351
    592.5420455932617
    592.5949957370758
    593.4536426067352
    593.4797732830048
    593.505490064621
    593.5650234222412
    593.617437839508
    593.670996427536
    593.6971595287323
    593.7381939888
    593.8352017402649
    593.9522323608398
    593.9932606220245
    594.028913974762
    594.0880131721497
    594.159234046936
    594.2263934612274
    594.2728281021118
    594.3253273963928
    594.3609223365784
    594.4341223239899
    594.4696736335754
    594.5048003196716
    594.5350825786591
    594.6400890350342
    594.7205953598022
    594.7795703411102
    594.825510263443
    594.9698367118835
    595.1594743728638
    595.2332234382629
    596.2934353351593
    596.3245983123779
    596.384384393692
    596.4200229644775
    596.4504265785217
    596.59748005867
    596.6386206150055
    596.7050664424896
    596.8415756225586
    596.9588468074799
    596.9946801662445
    597.0301866531372
    597.0830237865448
    597.1425311565399
    597.1832947731018
    597.2049231529236
    597.2461330890656
    597.3359208106995
    597.4019508361816
    597.4547288417816
    597.5278747081757
    597.5692844390869
    597.6219737529755
    597.6679067611694
    597.7268011569977
    597.77281498909
    597.8792681694031
    597.9109101295471
    597.9533393383026
    597.9889934062958
    598.0657677650452
    598.11865401268
    598.1594128608704
    598.2485315799713
    598.2889606952667
    598.341992855072
    598.3773875236511
    598.4125955104828
    598.4779324531555
    598.5591804981232
    598.6179604530334
    598.6905248165131
    598.7259395122528
    598.7853224277496
    598.8374073505402
    598.8778920173645
    598.9237620830536
    598.9695632457733
    599.0101435184479
    599.0404326915741
    599.086540222168
    599.202196598053
    599.2240967750549
    599.2597637176514
    601.5668339729309
    601.6100783348083
    601.7179992198944
    601.7485394477844
    601.7792196273804
    601.8150458335876
    601.8501591682434
    601.8855106830597
    601.9324667453766
    601.9789850711823
    602.0670838356018
    602.1133997440338
    602.1483685970306
    602.2455608844757
    604.4267249107361
    604.472870349884
    604.4981114864349
    604.5328893661499
    604.5910704135895
    604.6257865428925
    604.6607415676117
    604.7414124011993
    604.7821061611176
    604.8078889846802
    604.8482444286346
    604.9139630794525
    604.9728760719299
    605.0078372955322
    605.0429213047028
    605.0890440940857
    605.1301958560944
    605.1826598644257
    605.2413985729218
    605.2821133136749
    606.396630525589
    606.5126831531525
    606.5433554649353
    606.5689990520477
    606.599086523056
    606.6514539718628
    606.6919391155243
    606.7224140167236
    606.7692029476166
    606.8157141208649
    606.8750078678131
    606.9281058311462
    606.9688940048218
    607.0215005874634
    607.0620918273926
    607.0929310321808
    607.1455428600311
    607.181170463562
    607.2278907299042
    607.2581293582916
    607.293539762497
    607.400808095932
    607.4365916252136
    607.4830996990204
    607.5186016559601
    607.5592277050018
    607.6126835346222
    607.6482775211334
    607.7290592193604
    607.7951159477234
    607.8543438911438
    607.9207921028137
    608.0273501873016
    608.0750436782837
    608.127692937851
    608.1629195213318
    608.2157108783722
    608.2817690372467
    608.3227345943451
    608.3637003898621
    610.0722856521606
    610.1132690906525
    610.1388702392578
    610.1855220794678
    610.2210428714752
    610.2736179828644
    610.3331756591797
    610.4065680503845
    610.4472134113312
    610.4774398803711
    610.5368874073029
    610.5727009773254
    610.6327848434448



```python

train_dataset_2 = [store_dict_and_data_FULLY_CONNECTED[i][2] for i in train_anchs]
test_dataset_2 = [store_dict_and_data_FULLY_CONNECTED[i][2] for i in test_anchs]

    
train_loader = DataLoader(train_dataset_2, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset_2, batch_size=64, shuffle=False)

```


```python
track_full = dict()
```


```python
torch.manual_seed(163)
model = GCN(hidden_channels=64)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track_full[64] = [train_accs, test_accs]
```

    Epoch: 001, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 002, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 003, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 004, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 005, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 006, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 007, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 008, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 009, Train Acc: 0.4025, Test Acc: 0.3000
    Epoch: 010, Train Acc: 0.4025, Test Acc: 0.3125
    Epoch: 011, Train Acc: 0.4066, Test Acc: 0.3125
    Epoch: 012, Train Acc: 0.4066, Test Acc: 0.3125
    Epoch: 013, Train Acc: 0.4523, Test Acc: 0.3250
    Epoch: 014, Train Acc: 0.5311, Test Acc: 0.4000
    Epoch: 015, Train Acc: 0.5436, Test Acc: 0.4000
    Epoch: 016, Train Acc: 0.5726, Test Acc: 0.4750
    Epoch: 017, Train Acc: 0.5809, Test Acc: 0.5250
    Epoch: 018, Train Acc: 0.5851, Test Acc: 0.5500
    Epoch: 019, Train Acc: 0.6017, Test Acc: 0.5375
    Epoch: 020, Train Acc: 0.5934, Test Acc: 0.5500
    Epoch: 021, Train Acc: 0.5809, Test Acc: 0.5500
    Epoch: 022, Train Acc: 0.5975, Test Acc: 0.5500
    Epoch: 023, Train Acc: 0.5892, Test Acc: 0.5750
    Epoch: 024, Train Acc: 0.5892, Test Acc: 0.5750
    Epoch: 025, Train Acc: 0.6058, Test Acc: 0.5875
    Epoch: 026, Train Acc: 0.6515, Test Acc: 0.6375
    Epoch: 027, Train Acc: 0.6846, Test Acc: 0.6375
    Epoch: 028, Train Acc: 0.7054, Test Acc: 0.6250
    Epoch: 029, Train Acc: 0.7386, Test Acc: 0.6250
    Epoch: 030, Train Acc: 0.7718, Test Acc: 0.6000
    Epoch: 031, Train Acc: 0.7925, Test Acc: 0.6125
    Epoch: 032, Train Acc: 0.8050, Test Acc: 0.6375
    Epoch: 033, Train Acc: 0.8091, Test Acc: 0.6250
    Epoch: 034, Train Acc: 0.8216, Test Acc: 0.5750
    Epoch: 035, Train Acc: 0.8465, Test Acc: 0.6500
    Epoch: 036, Train Acc: 0.8465, Test Acc: 0.6125
    Epoch: 037, Train Acc: 0.8589, Test Acc: 0.6125
    Epoch: 038, Train Acc: 0.8714, Test Acc: 0.6250
    Epoch: 039, Train Acc: 0.8797, Test Acc: 0.6125
    Epoch: 040, Train Acc: 0.8921, Test Acc: 0.6375
    Epoch: 041, Train Acc: 0.9004, Test Acc: 0.6250
    Epoch: 042, Train Acc: 0.9087, Test Acc: 0.6125
    Epoch: 043, Train Acc: 0.9253, Test Acc: 0.6250
    Epoch: 044, Train Acc: 0.9336, Test Acc: 0.6500
    Epoch: 045, Train Acc: 0.9336, Test Acc: 0.6000
    Epoch: 046, Train Acc: 0.9378, Test Acc: 0.6250
    Epoch: 047, Train Acc: 0.9461, Test Acc: 0.6000
    Epoch: 048, Train Acc: 0.9502, Test Acc: 0.5875
    Epoch: 049, Train Acc: 0.9502, Test Acc: 0.5750
    Epoch: 050, Train Acc: 0.9585, Test Acc: 0.5750
    Epoch: 051, Train Acc: 0.9585, Test Acc: 0.5875
    Epoch: 052, Train Acc: 0.9710, Test Acc: 0.6000
    Epoch: 053, Train Acc: 0.9710, Test Acc: 0.6000
    Epoch: 054, Train Acc: 0.9793, Test Acc: 0.6125
    Epoch: 055, Train Acc: 0.9834, Test Acc: 0.6000
    Epoch: 056, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 057, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 058, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 059, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 060, Train Acc: 0.9876, Test Acc: 0.6250
    Epoch: 061, Train Acc: 0.9876, Test Acc: 0.6000
    Epoch: 062, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 063, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 064, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 065, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 066, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 067, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 068, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 069, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 070, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 071, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 072, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 073, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 074, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 075, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 076, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 077, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 078, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 079, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 080, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 081, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 082, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 083, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 084, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 085, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 086, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 087, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 088, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 089, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 090, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 091, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 092, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 093, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 094, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 095, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 096, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 097, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 098, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 099, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 100, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 101, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 102, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 103, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 104, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 105, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 106, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 107, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 108, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 109, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 110, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 111, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 112, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 113, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 114, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 115, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 116, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 117, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 118, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 119, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 120, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 121, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 122, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 123, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 124, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 125, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 126, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 127, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 128, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 129, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 130, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 131, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 132, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 133, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 134, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 135, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 136, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 137, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 138, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 139, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 140, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 141, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 142, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 143, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 144, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 145, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 146, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 147, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 148, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 149, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 150, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 151, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 152, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 153, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 154, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 155, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 156, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 157, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 158, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 159, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 160, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 161, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 162, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 163, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 164, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 165, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 166, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 167, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 168, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 169, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 170, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 171, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 172, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 173, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 174, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 175, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 176, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 177, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 178, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 179, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 180, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 181, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 182, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 183, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 184, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 185, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 186, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 187, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 188, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 189, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 190, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 191, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 192, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 193, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 194, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 195, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 196, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 197, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 198, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 199, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 200, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 201, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 202, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 203, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 204, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 205, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 206, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 207, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 208, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 209, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 210, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 211, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 212, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 213, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 214, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 215, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 216, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 217, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 218, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 219, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 220, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 221, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 222, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 223, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 224, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 225, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 226, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 227, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 228, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 229, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 230, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 231, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 232, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 233, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 234, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 235, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 236, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 237, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 238, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 239, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 240, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 241, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 242, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 243, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 244, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 245, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 246, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 247, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 248, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 249, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 250, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 251, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 252, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 253, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 254, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 255, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 256, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 257, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 258, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 259, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 260, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 261, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 262, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 263, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 264, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 265, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 266, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 267, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 268, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 269, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 270, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 271, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 272, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 273, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 274, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 275, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 276, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 277, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 278, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 279, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 280, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 281, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 282, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 283, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 284, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 285, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 286, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 287, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 288, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 289, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 290, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 291, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 292, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 293, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 294, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 295, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 296, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 297, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 298, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 299, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 300, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 301, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 302, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 303, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 304, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 305, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 306, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 307, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 308, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 309, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 310, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 311, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 312, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 313, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 314, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 315, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 316, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 317, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 318, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 319, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 320, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 321, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 322, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 323, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 324, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 325, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 326, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 327, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 328, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 329, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 330, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 331, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 332, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 333, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 334, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 335, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 336, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 337, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 338, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 339, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 340, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 341, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 342, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 343, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 344, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 345, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 346, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 347, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 348, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 349, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 350, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 351, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 352, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 353, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 354, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 355, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 356, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 357, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 358, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 359, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 360, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 361, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 362, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 363, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 364, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 365, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 366, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 367, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 368, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 369, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 370, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 371, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 372, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 373, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 374, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 375, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 376, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 377, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 378, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 379, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 380, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 381, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 382, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 383, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 384, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 385, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 386, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 387, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 388, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 389, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 390, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 391, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 392, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 393, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 394, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 395, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 396, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 397, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 398, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 399, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 400, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 401, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 402, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 403, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 404, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 405, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 406, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 407, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 408, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 409, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 410, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 411, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 412, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 413, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 414, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 415, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 416, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 417, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 418, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 419, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 420, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 421, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 422, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 423, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 424, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 425, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 426, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 427, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 428, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 429, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 430, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 431, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 432, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 433, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 434, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 435, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 436, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 437, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 438, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 439, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 440, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 441, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 442, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 443, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 444, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 445, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 446, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 447, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 448, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 449, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 450, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 451, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 452, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 453, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 454, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 455, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 456, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 457, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 458, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 459, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 460, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 461, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 462, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 463, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 464, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 465, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 466, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 467, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 468, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 469, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 470, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 471, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 472, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 473, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 474, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 475, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 476, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 477, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 478, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 479, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 480, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 481, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 482, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 483, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 484, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 485, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 486, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 487, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 488, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 489, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 490, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 491, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 492, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 493, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 494, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 495, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 496, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 497, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 498, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 499, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 500, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 501, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 502, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 503, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 504, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 505, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 506, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 507, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 508, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 509, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 510, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 511, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 512, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 513, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 514, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 515, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 516, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 517, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 518, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 519, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 520, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 521, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 522, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 523, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 524, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 525, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 526, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 527, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 528, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 529, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 530, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 531, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 532, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 533, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 534, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 535, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 536, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 537, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 538, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 539, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 540, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 541, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 542, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 543, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 544, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 545, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 546, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 547, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 548, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 549, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 550, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 551, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 552, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 553, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 554, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 555, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 556, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 557, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 558, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 559, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 560, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 561, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 562, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 563, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 564, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 565, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 566, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 567, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 568, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 569, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 570, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6000



```python
torch.manual_seed(163)
model = GCN(hidden_channels=32)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track_full[32] = [train_accs, test_accs]
```

    Epoch: 001, Train Acc: 0.3859, Test Acc: 0.5125
    Epoch: 002, Train Acc: 0.3942, Test Acc: 0.5250
    Epoch: 003, Train Acc: 0.4025, Test Acc: 0.5250
    Epoch: 004, Train Acc: 0.3983, Test Acc: 0.5375
    Epoch: 005, Train Acc: 0.4149, Test Acc: 0.5500
    Epoch: 006, Train Acc: 0.4606, Test Acc: 0.5625
    Epoch: 007, Train Acc: 0.4896, Test Acc: 0.5375
    Epoch: 008, Train Acc: 0.4855, Test Acc: 0.5375
    Epoch: 009, Train Acc: 0.4647, Test Acc: 0.5125
    Epoch: 010, Train Acc: 0.4979, Test Acc: 0.5500
    Epoch: 011, Train Acc: 0.5062, Test Acc: 0.5625
    Epoch: 012, Train Acc: 0.5145, Test Acc: 0.5750
    Epoch: 013, Train Acc: 0.5311, Test Acc: 0.5875
    Epoch: 014, Train Acc: 0.5436, Test Acc: 0.5875
    Epoch: 015, Train Acc: 0.5394, Test Acc: 0.5625
    Epoch: 016, Train Acc: 0.5519, Test Acc: 0.6000
    Epoch: 017, Train Acc: 0.5394, Test Acc: 0.6125
    Epoch: 018, Train Acc: 0.5436, Test Acc: 0.5875
    Epoch: 019, Train Acc: 0.5519, Test Acc: 0.5875
    Epoch: 020, Train Acc: 0.5643, Test Acc: 0.6000
    Epoch: 021, Train Acc: 0.5726, Test Acc: 0.6000
    Epoch: 022, Train Acc: 0.5851, Test Acc: 0.6125
    Epoch: 023, Train Acc: 0.5851, Test Acc: 0.6125
    Epoch: 024, Train Acc: 0.5934, Test Acc: 0.6000
    Epoch: 025, Train Acc: 0.6100, Test Acc: 0.6375
    Epoch: 026, Train Acc: 0.6349, Test Acc: 0.6375
    Epoch: 027, Train Acc: 0.6349, Test Acc: 0.6375
    Epoch: 028, Train Acc: 0.6390, Test Acc: 0.6375
    Epoch: 029, Train Acc: 0.6307, Test Acc: 0.6125
    Epoch: 030, Train Acc: 0.6349, Test Acc: 0.6125
    Epoch: 031, Train Acc: 0.6515, Test Acc: 0.6000
    Epoch: 032, Train Acc: 0.6390, Test Acc: 0.6000
    Epoch: 033, Train Acc: 0.6515, Test Acc: 0.5875
    Epoch: 034, Train Acc: 0.6473, Test Acc: 0.6000
    Epoch: 035, Train Acc: 0.6680, Test Acc: 0.6000
    Epoch: 036, Train Acc: 0.6846, Test Acc: 0.6000
    Epoch: 037, Train Acc: 0.7054, Test Acc: 0.6125
    Epoch: 038, Train Acc: 0.7220, Test Acc: 0.6000
    Epoch: 039, Train Acc: 0.7552, Test Acc: 0.6125
    Epoch: 040, Train Acc: 0.7593, Test Acc: 0.5875
    Epoch: 041, Train Acc: 0.7676, Test Acc: 0.5875
    Epoch: 042, Train Acc: 0.7884, Test Acc: 0.6250
    Epoch: 043, Train Acc: 0.8133, Test Acc: 0.6250
    Epoch: 044, Train Acc: 0.8133, Test Acc: 0.6250
    Epoch: 045, Train Acc: 0.8216, Test Acc: 0.6125
    Epoch: 046, Train Acc: 0.8382, Test Acc: 0.6000
    Epoch: 047, Train Acc: 0.8631, Test Acc: 0.6125
    Epoch: 048, Train Acc: 0.8672, Test Acc: 0.6250
    Epoch: 049, Train Acc: 0.8797, Test Acc: 0.6125
    Epoch: 050, Train Acc: 0.8838, Test Acc: 0.6250
    Epoch: 051, Train Acc: 0.8880, Test Acc: 0.6375
    Epoch: 052, Train Acc: 0.9046, Test Acc: 0.6375
    Epoch: 053, Train Acc: 0.9004, Test Acc: 0.6250
    Epoch: 054, Train Acc: 0.9212, Test Acc: 0.6375
    Epoch: 055, Train Acc: 0.9253, Test Acc: 0.6125
    Epoch: 056, Train Acc: 0.9336, Test Acc: 0.6250
    Epoch: 057, Train Acc: 0.9419, Test Acc: 0.6250
    Epoch: 058, Train Acc: 0.9502, Test Acc: 0.6250
    Epoch: 059, Train Acc: 0.9502, Test Acc: 0.6125
    Epoch: 060, Train Acc: 0.9502, Test Acc: 0.6250
    Epoch: 061, Train Acc: 0.9544, Test Acc: 0.6250
    Epoch: 062, Train Acc: 0.9544, Test Acc: 0.6250
    Epoch: 063, Train Acc: 0.9585, Test Acc: 0.6375
    Epoch: 064, Train Acc: 0.9627, Test Acc: 0.6375
    Epoch: 065, Train Acc: 0.9544, Test Acc: 0.6375
    Epoch: 066, Train Acc: 0.9627, Test Acc: 0.6250
    Epoch: 067, Train Acc: 0.9627, Test Acc: 0.6250
    Epoch: 068, Train Acc: 0.9668, Test Acc: 0.6500
    Epoch: 069, Train Acc: 0.9627, Test Acc: 0.6375
    Epoch: 070, Train Acc: 0.9668, Test Acc: 0.6375
    Epoch: 071, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 072, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 073, Train Acc: 0.9710, Test Acc: 0.6375
    Epoch: 074, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 075, Train Acc: 0.9751, Test Acc: 0.6375
    Epoch: 076, Train Acc: 0.9751, Test Acc: 0.6500
    Epoch: 077, Train Acc: 0.9751, Test Acc: 0.6500
    Epoch: 078, Train Acc: 0.9751, Test Acc: 0.6250
    Epoch: 079, Train Acc: 0.9793, Test Acc: 0.6250
    Epoch: 080, Train Acc: 0.9793, Test Acc: 0.6375
    Epoch: 081, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 082, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 083, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 084, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 085, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 086, Train Acc: 0.9876, Test Acc: 0.6375
    Epoch: 087, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 088, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 089, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 090, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 091, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 092, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 093, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 094, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 095, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 096, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 097, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 098, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 099, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 100, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 101, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 102, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 103, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 104, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 105, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 106, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 107, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 108, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 109, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 110, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 111, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 112, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 113, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 114, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 115, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 116, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 117, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 118, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 119, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 120, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 121, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 122, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 123, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 124, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 125, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 126, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 127, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 128, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 129, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 130, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 131, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 132, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 133, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 134, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 135, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 136, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 137, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 138, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 139, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 140, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 141, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 142, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 143, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 144, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 145, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 146, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 147, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 148, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 149, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 150, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 151, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 152, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 153, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 154, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 155, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 156, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 157, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 158, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 159, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 160, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 161, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 162, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 163, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 164, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 165, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 166, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 167, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 168, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 169, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 170, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 171, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 172, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 173, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 174, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 175, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 176, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 177, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 178, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 179, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 180, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 181, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 182, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 183, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 184, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 185, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 186, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 187, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 188, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 189, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 190, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 191, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 192, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 193, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 194, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 195, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 196, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 197, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 198, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 199, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 200, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 201, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 202, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 203, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 204, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 205, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 206, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 207, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 208, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 209, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 210, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 211, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 212, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 213, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 214, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 215, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 216, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 217, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 218, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 219, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 220, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 221, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 222, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 223, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 224, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 225, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 226, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 227, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 228, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 229, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 230, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 231, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 232, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 233, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 234, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 235, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 236, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 237, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 238, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 239, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 240, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 241, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 242, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 243, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 244, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 245, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 246, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 247, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 248, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 249, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 250, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 251, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 252, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 253, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 254, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 255, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 256, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 257, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 258, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 259, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 260, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 261, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 262, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 263, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 264, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 265, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 266, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 267, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 268, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 269, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 270, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 271, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 272, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 273, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 274, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 275, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 276, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 277, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 278, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 279, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 280, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 281, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 282, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 283, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 284, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 285, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 286, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 287, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 288, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 289, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 290, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 291, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 292, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 293, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 294, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 295, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 296, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 297, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 298, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 299, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 300, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 301, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 302, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 303, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 304, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 305, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 306, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 307, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 308, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 309, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 310, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 311, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 312, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 313, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 314, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 315, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 316, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 317, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 318, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 319, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 320, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 321, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 322, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 323, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 324, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 325, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 326, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 327, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 328, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 329, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 330, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 331, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 332, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 333, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 334, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 335, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 336, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 337, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 338, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 339, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 340, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 341, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 342, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 343, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 344, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 345, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 346, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 347, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 348, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 349, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 350, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 351, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 352, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 353, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 354, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 355, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 356, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 357, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 358, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 359, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 360, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 361, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 362, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 363, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 364, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 365, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 366, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 367, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 368, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 369, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 370, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 371, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 372, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 373, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 374, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 375, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 376, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 377, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 378, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 379, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 380, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 381, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 382, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 383, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 384, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 385, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 386, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 387, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 388, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 389, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 390, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 391, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 392, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 393, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 394, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 395, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 396, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 397, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 398, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 399, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 400, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 401, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 402, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 403, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 404, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 405, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 406, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 407, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 408, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 409, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 410, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 411, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 412, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 413, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 414, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 415, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 416, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 417, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 418, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 419, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 420, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 421, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 422, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 423, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 424, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 425, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 426, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 427, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 428, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 429, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 430, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 431, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 432, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 433, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 434, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 435, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 436, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 437, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 438, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 439, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 440, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 441, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 442, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 443, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 444, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 445, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 446, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 447, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 448, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 449, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 450, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 451, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 452, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 453, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 454, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 455, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 456, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 457, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 458, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 459, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 460, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 461, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 462, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 463, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 464, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 465, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 466, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 467, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 468, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 469, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 470, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 471, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 472, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 473, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 474, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 475, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 476, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 477, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 478, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 479, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 480, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 481, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 482, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 483, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 484, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 485, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 486, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 487, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 488, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 489, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 490, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 491, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 492, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 493, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 494, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 495, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 496, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 497, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 498, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 499, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 500, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 501, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 502, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 503, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 504, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 505, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 506, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 507, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 508, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 509, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 510, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 511, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 512, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 513, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 514, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 515, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 516, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 517, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 518, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 519, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 520, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 521, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 522, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 523, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 524, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 525, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 526, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 527, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 528, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 529, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 530, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 531, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 532, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 533, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 534, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 535, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 536, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 537, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 538, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 539, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 540, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 541, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 542, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 543, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 544, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 545, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 546, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 547, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 548, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 549, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 550, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 551, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 552, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 553, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 554, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 555, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 556, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 557, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 558, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 559, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 560, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 561, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 562, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 563, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 564, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 565, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 566, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 567, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 568, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 569, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 570, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6500



```python
torch.manual_seed(163)
model = GCN(hidden_channels=16)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track_full[16] = [train_accs, test_accs]
```

    Epoch: 001, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 002, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 003, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 004, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 005, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 006, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 007, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 008, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 009, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 010, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 011, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 012, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 013, Train Acc: 0.3983, Test Acc: 0.3000
    Epoch: 014, Train Acc: 0.3983, Test Acc: 0.3000
    Epoch: 015, Train Acc: 0.4025, Test Acc: 0.3375
    Epoch: 016, Train Acc: 0.4274, Test Acc: 0.3375
    Epoch: 017, Train Acc: 0.4398, Test Acc: 0.3250
    Epoch: 018, Train Acc: 0.4606, Test Acc: 0.3500
    Epoch: 019, Train Acc: 0.4772, Test Acc: 0.3625
    Epoch: 020, Train Acc: 0.4772, Test Acc: 0.3750
    Epoch: 021, Train Acc: 0.5021, Test Acc: 0.3750
    Epoch: 022, Train Acc: 0.5228, Test Acc: 0.3875
    Epoch: 023, Train Acc: 0.5311, Test Acc: 0.4375
    Epoch: 024, Train Acc: 0.5519, Test Acc: 0.4625
    Epoch: 025, Train Acc: 0.5519, Test Acc: 0.4500
    Epoch: 026, Train Acc: 0.5685, Test Acc: 0.4500
    Epoch: 027, Train Acc: 0.5768, Test Acc: 0.4500
    Epoch: 028, Train Acc: 0.5768, Test Acc: 0.4750
    Epoch: 029, Train Acc: 0.5685, Test Acc: 0.5000
    Epoch: 030, Train Acc: 0.5726, Test Acc: 0.5125
    Epoch: 031, Train Acc: 0.5726, Test Acc: 0.5125
    Epoch: 032, Train Acc: 0.5809, Test Acc: 0.5125
    Epoch: 033, Train Acc: 0.5851, Test Acc: 0.4875
    Epoch: 034, Train Acc: 0.5809, Test Acc: 0.4875
    Epoch: 035, Train Acc: 0.5768, Test Acc: 0.5000
    Epoch: 036, Train Acc: 0.5726, Test Acc: 0.5250
    Epoch: 037, Train Acc: 0.5768, Test Acc: 0.5375
    Epoch: 038, Train Acc: 0.5851, Test Acc: 0.5500
    Epoch: 039, Train Acc: 0.5934, Test Acc: 0.5625
    Epoch: 040, Train Acc: 0.6017, Test Acc: 0.5625
    Epoch: 041, Train Acc: 0.5975, Test Acc: 0.5625
    Epoch: 042, Train Acc: 0.5851, Test Acc: 0.5500
    Epoch: 043, Train Acc: 0.5809, Test Acc: 0.5625
    Epoch: 044, Train Acc: 0.5975, Test Acc: 0.5625
    Epoch: 045, Train Acc: 0.6017, Test Acc: 0.5625
    Epoch: 046, Train Acc: 0.6100, Test Acc: 0.5625
    Epoch: 047, Train Acc: 0.6100, Test Acc: 0.5750
    Epoch: 048, Train Acc: 0.6349, Test Acc: 0.6000
    Epoch: 049, Train Acc: 0.6515, Test Acc: 0.5875
    Epoch: 050, Train Acc: 0.6556, Test Acc: 0.5875
    Epoch: 051, Train Acc: 0.6763, Test Acc: 0.5875
    Epoch: 052, Train Acc: 0.6888, Test Acc: 0.6125
    Epoch: 053, Train Acc: 0.7178, Test Acc: 0.6125
    Epoch: 054, Train Acc: 0.7261, Test Acc: 0.6125
    Epoch: 055, Train Acc: 0.7220, Test Acc: 0.6250
    Epoch: 056, Train Acc: 0.7261, Test Acc: 0.6250
    Epoch: 057, Train Acc: 0.7510, Test Acc: 0.6250
    Epoch: 058, Train Acc: 0.7593, Test Acc: 0.6250
    Epoch: 059, Train Acc: 0.7718, Test Acc: 0.6250
    Epoch: 060, Train Acc: 0.7842, Test Acc: 0.6000
    Epoch: 061, Train Acc: 0.7884, Test Acc: 0.6000
    Epoch: 062, Train Acc: 0.7925, Test Acc: 0.6250
    Epoch: 063, Train Acc: 0.8050, Test Acc: 0.6250
    Epoch: 064, Train Acc: 0.8050, Test Acc: 0.6125
    Epoch: 065, Train Acc: 0.8340, Test Acc: 0.6000
    Epoch: 066, Train Acc: 0.8382, Test Acc: 0.6125
    Epoch: 067, Train Acc: 0.8465, Test Acc: 0.6125
    Epoch: 068, Train Acc: 0.8382, Test Acc: 0.6125
    Epoch: 069, Train Acc: 0.8299, Test Acc: 0.6000
    Epoch: 070, Train Acc: 0.8340, Test Acc: 0.6000
    Epoch: 071, Train Acc: 0.8340, Test Acc: 0.6000
    Epoch: 072, Train Acc: 0.8465, Test Acc: 0.6000
    Epoch: 073, Train Acc: 0.8465, Test Acc: 0.6125
    Epoch: 074, Train Acc: 0.8589, Test Acc: 0.6125
    Epoch: 075, Train Acc: 0.8506, Test Acc: 0.6250
    Epoch: 076, Train Acc: 0.8548, Test Acc: 0.6250
    Epoch: 077, Train Acc: 0.8589, Test Acc: 0.6375
    Epoch: 078, Train Acc: 0.8631, Test Acc: 0.6125
    Epoch: 079, Train Acc: 0.8631, Test Acc: 0.5875
    Epoch: 080, Train Acc: 0.8714, Test Acc: 0.6125
    Epoch: 081, Train Acc: 0.8672, Test Acc: 0.6000
    Epoch: 082, Train Acc: 0.8797, Test Acc: 0.6000
    Epoch: 083, Train Acc: 0.8797, Test Acc: 0.6000
    Epoch: 084, Train Acc: 0.8921, Test Acc: 0.6000
    Epoch: 085, Train Acc: 0.8963, Test Acc: 0.6125
    Epoch: 086, Train Acc: 0.9004, Test Acc: 0.6000
    Epoch: 087, Train Acc: 0.9004, Test Acc: 0.6000
    Epoch: 088, Train Acc: 0.8963, Test Acc: 0.6000
    Epoch: 089, Train Acc: 0.9004, Test Acc: 0.6000
    Epoch: 090, Train Acc: 0.9046, Test Acc: 0.5875
    Epoch: 091, Train Acc: 0.9087, Test Acc: 0.5875
    Epoch: 092, Train Acc: 0.9046, Test Acc: 0.5875
    Epoch: 093, Train Acc: 0.9129, Test Acc: 0.6000
    Epoch: 094, Train Acc: 0.9129, Test Acc: 0.6000
    Epoch: 095, Train Acc: 0.9170, Test Acc: 0.5875
    Epoch: 096, Train Acc: 0.9253, Test Acc: 0.6000
    Epoch: 097, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 098, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 099, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 100, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 101, Train Acc: 0.9419, Test Acc: 0.5875
    Epoch: 102, Train Acc: 0.9419, Test Acc: 0.5875
    Epoch: 103, Train Acc: 0.9419, Test Acc: 0.5875
    Epoch: 104, Train Acc: 0.9378, Test Acc: 0.5875
    Epoch: 105, Train Acc: 0.9461, Test Acc: 0.5875
    Epoch: 106, Train Acc: 0.9461, Test Acc: 0.5875
    Epoch: 107, Train Acc: 0.9461, Test Acc: 0.5875
    Epoch: 108, Train Acc: 0.9502, Test Acc: 0.5875
    Epoch: 109, Train Acc: 0.9544, Test Acc: 0.5875
    Epoch: 110, Train Acc: 0.9627, Test Acc: 0.5875
    Epoch: 111, Train Acc: 0.9627, Test Acc: 0.6000
    Epoch: 112, Train Acc: 0.9585, Test Acc: 0.6000
    Epoch: 113, Train Acc: 0.9627, Test Acc: 0.6000
    Epoch: 114, Train Acc: 0.9668, Test Acc: 0.6250
    Epoch: 115, Train Acc: 0.9668, Test Acc: 0.6250
    Epoch: 116, Train Acc: 0.9710, Test Acc: 0.6125
    Epoch: 117, Train Acc: 0.9751, Test Acc: 0.6000
    Epoch: 118, Train Acc: 0.9710, Test Acc: 0.6250
    Epoch: 119, Train Acc: 0.9710, Test Acc: 0.6125
    Epoch: 120, Train Acc: 0.9793, Test Acc: 0.6250
    Epoch: 121, Train Acc: 0.9834, Test Acc: 0.6125
    Epoch: 122, Train Acc: 0.9793, Test Acc: 0.6375
    Epoch: 123, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 124, Train Acc: 0.9917, Test Acc: 0.6375
    Epoch: 125, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 126, Train Acc: 0.9917, Test Acc: 0.6500
    Epoch: 127, Train Acc: 0.9917, Test Acc: 0.6625
    Epoch: 128, Train Acc: 0.9917, Test Acc: 0.6375
    Epoch: 129, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 130, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 131, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 132, Train Acc: 0.9917, Test Acc: 0.6250
    Epoch: 133, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 134, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 135, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 136, Train Acc: 0.9959, Test Acc: 0.6500
    Epoch: 137, Train Acc: 0.9959, Test Acc: 0.6375
    Epoch: 138, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 139, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 140, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 141, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 142, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 143, Train Acc: 0.9959, Test Acc: 0.6250
    Epoch: 144, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 145, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 146, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 147, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 148, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 149, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 150, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 151, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 152, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 153, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 154, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 155, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 156, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 157, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 158, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 159, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 160, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 161, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 162, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 163, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 164, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 165, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 166, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 167, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 168, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 169, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 170, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 171, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 172, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 173, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 174, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 175, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 176, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 177, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 178, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 179, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 180, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 181, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 182, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 183, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 184, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 185, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 186, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 187, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 188, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 189, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 190, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 191, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 192, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 193, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 194, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 195, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 196, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 197, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 198, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 199, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 200, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 201, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 202, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 203, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 204, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 205, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 206, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 207, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 208, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 209, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 210, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 211, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 212, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 213, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 214, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 215, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 216, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 217, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 218, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 219, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 220, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 221, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 222, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 223, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 224, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 225, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 226, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 227, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 228, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 229, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 230, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 231, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 232, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 233, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 234, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 235, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 236, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 237, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 238, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 239, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 240, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 241, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 242, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 243, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 244, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 245, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 246, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 247, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 248, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 249, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 250, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 251, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 252, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 253, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 254, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 255, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 256, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 257, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 258, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 259, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 260, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 261, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 262, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 263, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 264, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 265, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 266, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 267, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 268, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 269, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 270, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 271, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 272, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 273, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 274, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 275, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 276, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 277, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 278, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 279, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 280, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 281, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 282, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 283, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 284, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 285, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 286, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 287, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 288, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 289, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 290, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 291, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 292, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 293, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 294, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 295, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 296, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 297, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 298, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 299, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 300, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 301, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 302, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 303, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 304, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 305, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 306, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 307, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 308, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 309, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 310, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 311, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 312, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 313, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 314, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 315, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 316, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 317, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 318, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 319, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 320, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 321, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 322, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 323, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 324, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 325, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 326, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 327, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 328, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 329, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 330, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 331, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 332, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 333, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 334, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 335, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 336, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 337, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 338, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 339, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 340, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 341, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 342, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 343, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 344, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 345, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 346, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 347, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 348, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 349, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 350, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 351, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 352, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 353, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 354, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 355, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 356, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 357, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 358, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 359, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 360, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 361, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 362, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 363, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 364, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 365, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 366, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 367, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 368, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 369, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 370, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 371, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 372, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 373, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 374, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 375, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 376, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 377, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 378, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 379, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 380, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 381, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 382, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 383, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 384, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 385, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 386, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 387, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 388, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 389, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 390, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 391, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 392, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 393, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 394, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 395, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 396, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 397, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 398, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 399, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 400, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 401, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 402, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 403, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 404, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 405, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 406, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 407, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 408, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 409, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 410, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 411, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 412, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 413, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 414, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 415, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 416, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 417, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 418, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 419, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 420, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 421, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 422, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 423, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 424, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 425, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 426, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 427, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 428, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 429, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 430, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 431, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 432, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 433, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 434, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 435, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 436, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 437, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 438, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 439, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 440, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 441, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 442, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 443, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 444, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 445, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 446, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 447, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 448, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 449, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 450, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 451, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 452, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 453, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 454, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 455, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 456, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 457, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 458, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 459, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 460, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 461, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 462, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 463, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 464, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 465, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 466, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 467, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 468, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 469, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 470, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 471, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 472, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 473, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 474, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 475, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 476, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 477, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 478, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 479, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 480, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 481, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 482, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 483, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 484, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 485, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 486, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 487, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 488, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 489, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 490, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 491, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 492, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 493, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 494, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 495, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 496, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 497, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 498, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 499, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 500, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 501, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 502, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 503, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 504, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 505, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 506, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 507, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 508, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 509, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 510, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 511, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 512, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 513, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 514, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 515, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 516, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 517, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 518, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 519, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 520, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 521, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 522, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 523, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 524, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 525, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 526, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 527, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 528, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 529, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 530, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 531, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 532, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 533, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 534, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 535, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 536, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 537, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 538, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 539, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 540, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 541, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 542, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 543, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 544, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 545, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 546, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 547, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 548, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 549, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 550, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 551, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 552, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 553, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 554, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 555, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 556, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 557, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 558, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 559, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 560, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 561, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 562, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 563, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 564, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 565, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 566, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 567, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 568, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 569, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 570, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.6625
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.6500
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.6750
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.6375
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.6250
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6125



```python
torch.manual_seed(163)
model = GCN(hidden_channels=8)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch)  # Perform a single forward pass.
        out = model(data, data.batch)
        
        predMap = dict({'CRISPR' : [1,0,0], 'MGE' : [0,1,0], 'unclassified' : [0,0,1]})
        ground = torch.tensor([predMap[i] for i in data.y]).float()
        
        #target_tensor = torch.tensor([1*(np.array(['CRISPR','MGE','unclassified']) == str(data.y))]).float()
        #target_tensor = target_tensor.expand_as(out)
        #print(target_tensor)
        #print('STOP!')
        loss = criterion(out, ground)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    predMap = dict({'CRISPR' : 0, 'MGE' : 1, 'unclassified' : 2})
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        ground = torch.tensor([predMap[i] for i in data.y])
        correct += int((pred == ground).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accs, test_accs = [], []
for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_accs.append(train_acc)
    test_accs.append(test_acc)
track_full[8] = [train_accs, test_accs]
```

    Epoch: 001, Train Acc: 0.2739, Test Acc: 0.1875
    Epoch: 002, Train Acc: 0.2905, Test Acc: 0.1875
    Epoch: 003, Train Acc: 0.3112, Test Acc: 0.2000
    Epoch: 004, Train Acc: 0.3195, Test Acc: 0.2000
    Epoch: 005, Train Acc: 0.3610, Test Acc: 0.2500
    Epoch: 006, Train Acc: 0.3942, Test Acc: 0.2875
    Epoch: 007, Train Acc: 0.3817, Test Acc: 0.2750
    Epoch: 008, Train Acc: 0.3859, Test Acc: 0.2875
    Epoch: 009, Train Acc: 0.3900, Test Acc: 0.3000
    Epoch: 010, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 011, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 012, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 013, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 014, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 015, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 016, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 017, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 018, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 019, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 020, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 021, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 022, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 023, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 024, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 025, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 026, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 027, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 028, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 029, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 030, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 031, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 032, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 033, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 034, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 035, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 036, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 037, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 038, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 039, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 040, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 041, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 042, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 043, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 044, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 045, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 046, Train Acc: 0.3942, Test Acc: 0.3000
    Epoch: 047, Train Acc: 0.3983, Test Acc: 0.3125
    Epoch: 048, Train Acc: 0.4066, Test Acc: 0.3250
    Epoch: 049, Train Acc: 0.4066, Test Acc: 0.3375
    Epoch: 050, Train Acc: 0.4149, Test Acc: 0.3375
    Epoch: 051, Train Acc: 0.4191, Test Acc: 0.3375
    Epoch: 052, Train Acc: 0.4191, Test Acc: 0.3375
    Epoch: 053, Train Acc: 0.4232, Test Acc: 0.3500
    Epoch: 054, Train Acc: 0.4232, Test Acc: 0.3625
    Epoch: 055, Train Acc: 0.4232, Test Acc: 0.3750
    Epoch: 056, Train Acc: 0.4274, Test Acc: 0.3875
    Epoch: 057, Train Acc: 0.4315, Test Acc: 0.4000
    Epoch: 058, Train Acc: 0.4315, Test Acc: 0.4125
    Epoch: 059, Train Acc: 0.4481, Test Acc: 0.4000
    Epoch: 060, Train Acc: 0.4481, Test Acc: 0.3875
    Epoch: 061, Train Acc: 0.4523, Test Acc: 0.4000
    Epoch: 062, Train Acc: 0.4606, Test Acc: 0.4000
    Epoch: 063, Train Acc: 0.4689, Test Acc: 0.4125
    Epoch: 064, Train Acc: 0.4855, Test Acc: 0.4000
    Epoch: 065, Train Acc: 0.4896, Test Acc: 0.4375
    Epoch: 066, Train Acc: 0.4938, Test Acc: 0.4250
    Epoch: 067, Train Acc: 0.5187, Test Acc: 0.4500
    Epoch: 068, Train Acc: 0.5353, Test Acc: 0.4625
    Epoch: 069, Train Acc: 0.5311, Test Acc: 0.4625
    Epoch: 070, Train Acc: 0.5560, Test Acc: 0.4625
    Epoch: 071, Train Acc: 0.5519, Test Acc: 0.4625
    Epoch: 072, Train Acc: 0.5519, Test Acc: 0.4625
    Epoch: 073, Train Acc: 0.5477, Test Acc: 0.4625
    Epoch: 074, Train Acc: 0.5436, Test Acc: 0.4625
    Epoch: 075, Train Acc: 0.5477, Test Acc: 0.4500
    Epoch: 076, Train Acc: 0.5477, Test Acc: 0.4500
    Epoch: 077, Train Acc: 0.5394, Test Acc: 0.4625
    Epoch: 078, Train Acc: 0.5602, Test Acc: 0.5000
    Epoch: 079, Train Acc: 0.5726, Test Acc: 0.5250
    Epoch: 080, Train Acc: 0.5851, Test Acc: 0.5250
    Epoch: 081, Train Acc: 0.5809, Test Acc: 0.5250
    Epoch: 082, Train Acc: 0.5851, Test Acc: 0.5125
    Epoch: 083, Train Acc: 0.5851, Test Acc: 0.5125
    Epoch: 084, Train Acc: 0.5892, Test Acc: 0.5125
    Epoch: 085, Train Acc: 0.5892, Test Acc: 0.5000
    Epoch: 086, Train Acc: 0.5851, Test Acc: 0.5250
    Epoch: 087, Train Acc: 0.5726, Test Acc: 0.5250
    Epoch: 088, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 089, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 090, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 091, Train Acc: 0.5768, Test Acc: 0.5500
    Epoch: 092, Train Acc: 0.5809, Test Acc: 0.5500
    Epoch: 093, Train Acc: 0.5768, Test Acc: 0.5625
    Epoch: 094, Train Acc: 0.5851, Test Acc: 0.5625
    Epoch: 095, Train Acc: 0.5851, Test Acc: 0.5625
    Epoch: 096, Train Acc: 0.5892, Test Acc: 0.5625
    Epoch: 097, Train Acc: 0.5892, Test Acc: 0.5375
    Epoch: 098, Train Acc: 0.5934, Test Acc: 0.5375
    Epoch: 099, Train Acc: 0.6141, Test Acc: 0.5375
    Epoch: 100, Train Acc: 0.6183, Test Acc: 0.5375
    Epoch: 101, Train Acc: 0.6017, Test Acc: 0.5375
    Epoch: 102, Train Acc: 0.6100, Test Acc: 0.5500
    Epoch: 103, Train Acc: 0.6100, Test Acc: 0.5625
    Epoch: 104, Train Acc: 0.6183, Test Acc: 0.5500
    Epoch: 105, Train Acc: 0.6141, Test Acc: 0.5625
    Epoch: 106, Train Acc: 0.6307, Test Acc: 0.5625
    Epoch: 107, Train Acc: 0.6349, Test Acc: 0.5875
    Epoch: 108, Train Acc: 0.6556, Test Acc: 0.6000
    Epoch: 109, Train Acc: 0.6598, Test Acc: 0.6000
    Epoch: 110, Train Acc: 0.6639, Test Acc: 0.5875
    Epoch: 111, Train Acc: 0.6763, Test Acc: 0.6000
    Epoch: 112, Train Acc: 0.6888, Test Acc: 0.6000
    Epoch: 113, Train Acc: 0.6846, Test Acc: 0.6000
    Epoch: 114, Train Acc: 0.6888, Test Acc: 0.5875
    Epoch: 115, Train Acc: 0.7012, Test Acc: 0.5750
    Epoch: 116, Train Acc: 0.7012, Test Acc: 0.5875
    Epoch: 117, Train Acc: 0.7054, Test Acc: 0.5875
    Epoch: 118, Train Acc: 0.7012, Test Acc: 0.5875
    Epoch: 119, Train Acc: 0.7137, Test Acc: 0.5875
    Epoch: 120, Train Acc: 0.7178, Test Acc: 0.5625
    Epoch: 121, Train Acc: 0.7220, Test Acc: 0.5500
    Epoch: 122, Train Acc: 0.7261, Test Acc: 0.5250
    Epoch: 123, Train Acc: 0.7427, Test Acc: 0.5500
    Epoch: 124, Train Acc: 0.7427, Test Acc: 0.5625
    Epoch: 125, Train Acc: 0.7469, Test Acc: 0.5375
    Epoch: 126, Train Acc: 0.7635, Test Acc: 0.5250
    Epoch: 127, Train Acc: 0.7676, Test Acc: 0.5250
    Epoch: 128, Train Acc: 0.7759, Test Acc: 0.5375
    Epoch: 129, Train Acc: 0.7759, Test Acc: 0.5375
    Epoch: 130, Train Acc: 0.7842, Test Acc: 0.5375
    Epoch: 131, Train Acc: 0.7801, Test Acc: 0.5375
    Epoch: 132, Train Acc: 0.7801, Test Acc: 0.5375
    Epoch: 133, Train Acc: 0.7801, Test Acc: 0.5375
    Epoch: 134, Train Acc: 0.7925, Test Acc: 0.5375
    Epoch: 135, Train Acc: 0.8008, Test Acc: 0.5375
    Epoch: 136, Train Acc: 0.7884, Test Acc: 0.5500
    Epoch: 137, Train Acc: 0.7967, Test Acc: 0.5500
    Epoch: 138, Train Acc: 0.8133, Test Acc: 0.5500
    Epoch: 139, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 140, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 141, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 142, Train Acc: 0.8091, Test Acc: 0.5500
    Epoch: 143, Train Acc: 0.8133, Test Acc: 0.5500
    Epoch: 144, Train Acc: 0.7967, Test Acc: 0.5500
    Epoch: 145, Train Acc: 0.8050, Test Acc: 0.5625
    Epoch: 146, Train Acc: 0.8133, Test Acc: 0.5500
    Epoch: 147, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 148, Train Acc: 0.8133, Test Acc: 0.5625
    Epoch: 149, Train Acc: 0.8216, Test Acc: 0.5625
    Epoch: 150, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 151, Train Acc: 0.8174, Test Acc: 0.5500
    Epoch: 152, Train Acc: 0.8216, Test Acc: 0.5500
    Epoch: 153, Train Acc: 0.8216, Test Acc: 0.5500
    Epoch: 154, Train Acc: 0.8257, Test Acc: 0.5625
    Epoch: 155, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 156, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 157, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 158, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 159, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 160, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 161, Train Acc: 0.8299, Test Acc: 0.5500
    Epoch: 162, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 163, Train Acc: 0.8257, Test Acc: 0.5375
    Epoch: 164, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 165, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 166, Train Acc: 0.8465, Test Acc: 0.5375
    Epoch: 167, Train Acc: 0.8382, Test Acc: 0.5500
    Epoch: 168, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 169, Train Acc: 0.8340, Test Acc: 0.5500
    Epoch: 170, Train Acc: 0.8423, Test Acc: 0.5500
    Epoch: 171, Train Acc: 0.8340, Test Acc: 0.5625
    Epoch: 172, Train Acc: 0.8506, Test Acc: 0.5375
    Epoch: 173, Train Acc: 0.8589, Test Acc: 0.5500
    Epoch: 174, Train Acc: 0.8548, Test Acc: 0.5375
    Epoch: 175, Train Acc: 0.8589, Test Acc: 0.5375
    Epoch: 176, Train Acc: 0.8589, Test Acc: 0.5500
    Epoch: 177, Train Acc: 0.8589, Test Acc: 0.5625
    Epoch: 178, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 179, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 180, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 181, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 182, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 183, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 184, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 185, Train Acc: 0.8714, Test Acc: 0.5500
    Epoch: 186, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 187, Train Acc: 0.8631, Test Acc: 0.5625
    Epoch: 188, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 189, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 190, Train Acc: 0.8631, Test Acc: 0.5500
    Epoch: 191, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 192, Train Acc: 0.8714, Test Acc: 0.5375
    Epoch: 193, Train Acc: 0.8672, Test Acc: 0.5375
    Epoch: 194, Train Acc: 0.8672, Test Acc: 0.5375
    Epoch: 195, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 196, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 197, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 198, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 199, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 200, Train Acc: 0.8672, Test Acc: 0.5500
    Epoch: 201, Train Acc: 0.8714, Test Acc: 0.5375
    Epoch: 202, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 203, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 204, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 205, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 206, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 207, Train Acc: 0.8755, Test Acc: 0.5375
    Epoch: 208, Train Acc: 0.8838, Test Acc: 0.5500
    Epoch: 209, Train Acc: 0.8797, Test Acc: 0.5375
    Epoch: 210, Train Acc: 0.8880, Test Acc: 0.5375
    Epoch: 211, Train Acc: 0.8921, Test Acc: 0.5375
    Epoch: 212, Train Acc: 0.8880, Test Acc: 0.5375
    Epoch: 213, Train Acc: 0.8880, Test Acc: 0.5375
    Epoch: 214, Train Acc: 0.8921, Test Acc: 0.5375
    Epoch: 215, Train Acc: 0.8921, Test Acc: 0.5125
    Epoch: 216, Train Acc: 0.9046, Test Acc: 0.5125
    Epoch: 217, Train Acc: 0.9087, Test Acc: 0.5250
    Epoch: 218, Train Acc: 0.9004, Test Acc: 0.5125
    Epoch: 219, Train Acc: 0.9004, Test Acc: 0.5125
    Epoch: 220, Train Acc: 0.9046, Test Acc: 0.5125
    Epoch: 221, Train Acc: 0.9004, Test Acc: 0.5250
    Epoch: 222, Train Acc: 0.9046, Test Acc: 0.5250
    Epoch: 223, Train Acc: 0.9046, Test Acc: 0.5250
    Epoch: 224, Train Acc: 0.9129, Test Acc: 0.5250
    Epoch: 225, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 226, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 227, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 228, Train Acc: 0.9087, Test Acc: 0.5375
    Epoch: 229, Train Acc: 0.9170, Test Acc: 0.5500
    Epoch: 230, Train Acc: 0.9129, Test Acc: 0.5375
    Epoch: 231, Train Acc: 0.9087, Test Acc: 0.5250
    Epoch: 232, Train Acc: 0.9170, Test Acc: 0.5250
    Epoch: 233, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 234, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 235, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 236, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 237, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 238, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 239, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 240, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 241, Train Acc: 0.9295, Test Acc: 0.5250
    Epoch: 242, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 243, Train Acc: 0.9212, Test Acc: 0.5250
    Epoch: 244, Train Acc: 0.9295, Test Acc: 0.5250
    Epoch: 245, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 246, Train Acc: 0.9253, Test Acc: 0.5250
    Epoch: 247, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 248, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 249, Train Acc: 0.9419, Test Acc: 0.5250
    Epoch: 250, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 251, Train Acc: 0.9336, Test Acc: 0.5125
    Epoch: 252, Train Acc: 0.9336, Test Acc: 0.5250
    Epoch: 253, Train Acc: 0.9378, Test Acc: 0.5250
    Epoch: 254, Train Acc: 0.9461, Test Acc: 0.5250
    Epoch: 255, Train Acc: 0.9419, Test Acc: 0.5250
    Epoch: 256, Train Acc: 0.9502, Test Acc: 0.5125
    Epoch: 257, Train Acc: 0.9502, Test Acc: 0.5250
    Epoch: 258, Train Acc: 0.9461, Test Acc: 0.5250
    Epoch: 259, Train Acc: 0.9461, Test Acc: 0.5250
    Epoch: 260, Train Acc: 0.9502, Test Acc: 0.5250
    Epoch: 261, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 262, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 263, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 264, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 265, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 266, Train Acc: 0.9585, Test Acc: 0.5250
    Epoch: 267, Train Acc: 0.9585, Test Acc: 0.5125
    Epoch: 268, Train Acc: 0.9585, Test Acc: 0.5125
    Epoch: 269, Train Acc: 0.9627, Test Acc: 0.5250
    Epoch: 270, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 271, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 272, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 273, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 274, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 275, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 276, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 277, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 278, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 279, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 280, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 281, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 282, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 283, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 284, Train Acc: 0.9627, Test Acc: 0.5625
    Epoch: 285, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 286, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 287, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 288, Train Acc: 0.9627, Test Acc: 0.5500
    Epoch: 289, Train Acc: 0.9585, Test Acc: 0.5500
    Epoch: 290, Train Acc: 0.9585, Test Acc: 0.5750
    Epoch: 291, Train Acc: 0.9627, Test Acc: 0.5625
    Epoch: 292, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 293, Train Acc: 0.9627, Test Acc: 0.5750
    Epoch: 294, Train Acc: 0.9627, Test Acc: 0.5750
    Epoch: 295, Train Acc: 0.9627, Test Acc: 0.5750
    Epoch: 296, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 297, Train Acc: 0.9710, Test Acc: 0.5750
    Epoch: 298, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 299, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 300, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 301, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 302, Train Acc: 0.9710, Test Acc: 0.5750
    Epoch: 303, Train Acc: 0.9751, Test Acc: 0.5500
    Epoch: 304, Train Acc: 0.9710, Test Acc: 0.5625
    Epoch: 305, Train Acc: 0.9668, Test Acc: 0.5750
    Epoch: 306, Train Acc: 0.9710, Test Acc: 0.5750
    Epoch: 307, Train Acc: 0.9751, Test Acc: 0.5750
    Epoch: 308, Train Acc: 0.9751, Test Acc: 0.5750
    Epoch: 309, Train Acc: 0.9793, Test Acc: 0.5500
    Epoch: 310, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 311, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 312, Train Acc: 0.9751, Test Acc: 0.5625
    Epoch: 313, Train Acc: 0.9710, Test Acc: 0.6000
    Epoch: 314, Train Acc: 0.9751, Test Acc: 0.5750
    Epoch: 315, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 316, Train Acc: 0.9793, Test Acc: 0.5750
    Epoch: 317, Train Acc: 0.9793, Test Acc: 0.5625
    Epoch: 318, Train Acc: 0.9793, Test Acc: 0.5750
    Epoch: 319, Train Acc: 0.9793, Test Acc: 0.6000
    Epoch: 320, Train Acc: 0.9793, Test Acc: 0.6000
    Epoch: 321, Train Acc: 0.9793, Test Acc: 0.6000
    Epoch: 322, Train Acc: 0.9793, Test Acc: 0.5500
    Epoch: 323, Train Acc: 0.9834, Test Acc: 0.5500
    Epoch: 324, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 325, Train Acc: 0.9793, Test Acc: 0.5875
    Epoch: 326, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 327, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 328, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 329, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 330, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 331, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 332, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 333, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 334, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 335, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 336, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 337, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 338, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 339, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 340, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 341, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 342, Train Acc: 0.9834, Test Acc: 0.5625
    Epoch: 343, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 344, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 345, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 346, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 347, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 348, Train Acc: 0.9834, Test Acc: 0.5875
    Epoch: 349, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 350, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 351, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 352, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 353, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 354, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 355, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 356, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 357, Train Acc: 0.9834, Test Acc: 0.5750
    Epoch: 358, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 359, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 360, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 361, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 362, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 363, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 364, Train Acc: 0.9876, Test Acc: 0.5875
    Epoch: 365, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 366, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 367, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 368, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 369, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 370, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 371, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 372, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 373, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 374, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 375, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 376, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 377, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 378, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 379, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 380, Train Acc: 0.9876, Test Acc: 0.5750
    Epoch: 381, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 382, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 383, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 384, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 385, Train Acc: 0.9876, Test Acc: 0.5625
    Epoch: 386, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 387, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 388, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 389, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 390, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 391, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 392, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 393, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 394, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 395, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 396, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 397, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 398, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 399, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 400, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 401, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 402, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 403, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 404, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 405, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 406, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 407, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 408, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 409, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 410, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 411, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 412, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 413, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 414, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 415, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 416, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 417, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 418, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 419, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 420, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 421, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 422, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 423, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 424, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 425, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 426, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 427, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 428, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 429, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 430, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 431, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 432, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 433, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 434, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 435, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 436, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 437, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 438, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 439, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 440, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 441, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 442, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 443, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 444, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 445, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 446, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 447, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 448, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 449, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 450, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 451, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 452, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 453, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 454, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 455, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 456, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 457, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 458, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 459, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 460, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 461, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 462, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 463, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 464, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 465, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 466, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 467, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 468, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 469, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 470, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 471, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 472, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 473, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 474, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 475, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 476, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 477, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 478, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 479, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 480, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 481, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 482, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 483, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 484, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 485, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 486, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 487, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 488, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 489, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 490, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 491, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 492, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 493, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 494, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 495, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 496, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 497, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 498, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 499, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 500, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 501, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 502, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 503, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 504, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 505, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 506, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 507, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 508, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 509, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 510, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 511, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 512, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 513, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 514, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 515, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 516, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 517, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 518, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 519, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 520, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 521, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 522, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 523, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 524, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 525, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 526, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 527, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 528, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 529, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 530, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 531, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 532, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 533, Train Acc: 0.9917, Test Acc: 0.6000
    Epoch: 534, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 535, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 536, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 537, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 538, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 539, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 540, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 541, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 542, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 543, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 544, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 545, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 546, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 547, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 548, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 549, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 550, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 551, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 552, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 553, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 554, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 555, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 556, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 557, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 558, Train Acc: 0.9917, Test Acc: 0.5500
    Epoch: 559, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 560, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 561, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 562, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 563, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 564, Train Acc: 0.9917, Test Acc: 0.5875
    Epoch: 565, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 566, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 567, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 568, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 569, Train Acc: 0.9917, Test Acc: 0.5625
    Epoch: 570, Train Acc: 0.9917, Test Acc: 0.5750
    Epoch: 571, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 572, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 573, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 574, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 575, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 576, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 577, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 578, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 579, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 580, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 581, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 582, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 583, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 584, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 585, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 586, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 587, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 588, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 589, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 590, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 591, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 592, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 593, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 594, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 595, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 596, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 597, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 598, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 599, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 600, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 601, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 602, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 603, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 604, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 605, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 606, Train Acc: 1.0000, Test Acc: 0.5625
    Epoch: 607, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 608, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 609, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 610, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 611, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 612, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 613, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 614, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 615, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 616, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 617, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 618, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 619, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 620, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 621, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 622, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 623, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 624, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 625, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 626, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 627, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 628, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 629, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 630, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 631, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 632, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 633, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 634, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 635, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 636, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 637, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 638, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 639, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 640, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 641, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 642, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 643, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 644, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 645, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 646, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 647, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 648, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 649, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 650, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 651, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 652, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 653, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 654, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 655, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 656, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 657, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 658, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 659, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 660, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 661, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 662, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 663, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 664, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 665, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 666, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 667, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 668, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 669, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 670, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 671, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 672, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 673, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 674, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 675, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 676, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 677, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 678, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 679, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 680, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 681, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 682, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 683, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 684, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 685, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 686, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 687, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 688, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 689, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 690, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 691, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 692, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 693, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 694, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 695, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 696, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 697, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 698, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 699, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 700, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 701, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 702, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 703, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 704, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 705, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 706, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 707, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 708, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 709, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 710, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 711, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 712, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 713, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 714, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 715, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 716, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 717, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 718, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 719, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 720, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 721, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 722, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 723, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 724, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 725, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 726, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 727, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 728, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 729, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 730, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 731, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 732, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 733, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 734, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 735, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 736, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 737, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 738, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 739, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 740, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 741, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 742, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 743, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 744, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 745, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 746, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 747, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 748, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 749, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 750, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 751, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 752, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 753, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 754, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 755, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 756, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 757, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 758, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 759, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 760, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 761, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 762, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 763, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 764, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 765, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 766, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 767, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 768, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 769, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 770, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 771, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 772, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 773, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 774, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 775, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 776, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 777, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 778, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 779, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 780, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 781, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 782, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 783, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 784, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 785, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 786, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 787, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 788, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 789, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 790, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 791, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 792, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 793, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 794, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 795, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 796, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 797, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 798, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 799, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 800, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 801, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 802, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 803, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 804, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 805, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 806, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 807, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 808, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 809, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 810, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 811, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 812, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 813, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 814, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 815, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 816, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 817, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 818, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 819, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 820, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 821, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 822, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 823, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 824, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 825, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 826, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 827, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 828, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 829, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 830, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 831, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 832, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 833, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 834, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 835, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 836, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 837, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 838, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 839, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 840, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 841, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 842, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 843, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 844, Train Acc: 1.0000, Test Acc: 0.6125
    Epoch: 845, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 846, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 847, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 848, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 849, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 850, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 851, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 852, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 853, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 854, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 855, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 856, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 857, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 858, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 859, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 860, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 861, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 862, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 863, Train Acc: 1.0000, Test Acc: 0.5750
    Epoch: 864, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 865, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 866, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 867, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 868, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 869, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 870, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 871, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 872, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 873, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 874, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 875, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 876, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 877, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 878, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 879, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 880, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 881, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 882, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 883, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 884, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 885, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 886, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 887, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 888, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 889, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 890, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 891, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 892, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 893, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 894, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 895, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 896, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 897, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 898, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 899, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 900, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 901, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 902, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 903, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 904, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 905, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 906, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 907, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 908, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 909, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 910, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 911, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 912, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 913, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 914, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 915, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 916, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 917, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 918, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 919, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 920, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 921, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 922, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 923, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 924, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 925, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 926, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 927, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 928, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 929, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 930, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 931, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 932, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 933, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 934, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 935, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 936, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 937, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 938, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 939, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 940, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 941, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 942, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 943, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 944, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 945, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 946, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 947, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 948, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 949, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 950, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 951, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 952, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 953, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 954, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 955, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 956, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 957, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 958, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 959, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 960, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 961, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 962, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 963, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 964, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 965, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 966, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 967, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 968, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 969, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 970, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 971, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 972, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 973, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 974, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 975, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 976, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 977, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 978, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 979, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 980, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 981, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 982, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 983, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 984, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 985, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 986, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 987, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 988, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 989, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 990, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 991, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 992, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 993, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 994, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 995, Train Acc: 1.0000, Test Acc: 0.5875
    Epoch: 996, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 997, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 998, Train Acc: 1.0000, Test Acc: 0.6000
    Epoch: 999, Train Acc: 1.0000, Test Acc: 0.6000



```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
len(track_full[8][1])
```




    999




```python

plt.figure(figsize=(12, 15))

plt.xlim(-1, 1001)  # X range [-1, 1001]
plt.ylim(0, 1.1)   

x = list(range(2,1001))

sns.lineplot(x=x, y=track_full[8][0],color='red')
sns.lineplot(x=x, y=track_full[8][1],color='red')

sns.lineplot(x=x, y=track_full[16][0],color='green')
sns.lineplot(x=x, y=track_full[16][1],color='green')

sns.lineplot(x=x, y=track_full[32][0],color='blue')
sns.lineplot(x=x, y=track_full[32][1],color='blue')

sns.lineplot(x=x, y=track_full[64][0],color='purple')
sns.lineplot(x=x, y=track_full[64][1],color='purple')

#plt.figure(figsize=(40, 80))

```




    <AxesSubplot:>




    
![png](output_50_1.png)
    



```python
plt.figure(figsize=(12, 15))

plt.xlim(-1, 1001)  # X range [-1, 1001]
plt.ylim(0, 1.1)   

x = list(range(2,1001))

sns.lineplot(x=x, y=track[8][0],color='red')
sns.lineplot(x=x, y=track[8][1],color='red')

sns.lineplot(x=x, y=track[16][0],color='green')
sns.lineplot(x=x, y=track[16][1],color='green')

sns.lineplot(x=x, y=track[32][0],color='blue')
sns.lineplot(x=x, y=track[32][1],color='blue')

sns.lineplot(x=x, y=track[64][0],color='purple')
sns.lineplot(x=x, y=track[64][1],color='purple')

```




    <AxesSubplot:>




    
![png](output_51_1.png)
    


<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

#### Take data from microbial. 


```python
ann = glob.glob('/oak/stanford/groups/horence/punit/bioinformatics/results/*George_rerun_10*/bowtie*/Rfam_merge/merged_final.tsv')
fir = pd.read_csv(ann[0],sep='\t')
fir['dataset'] = ann[0]
for i in ann[1:]:
    nex = pd.read_csv(i,sep='\t')
    nex['dataset'] = i
    fir = pd.concat([fir,nex])
fir['GH_CRISPR'] = (fir['sequence_hits_spacers'] != '*') | (fir['sequence_hits_direct_repeats'] != '*')
fir['GH_MGE'] = (fir['sequence_hits_dfam_te_eukaryota'] != '*') | (fir['sequence_hits_ice_iceberg'] != '*') | (fir['sequence_hits_mge_aclame_genes_all_0'] != '*') | (fir['sequence_hits_tncentral_te_prokaryotes_final'] != '*')
fir['GH_positive'] = (fir['GH_CRISPR'] * 1) + (fir['GH_MGE'] * 1)
l = fir.groupby(['dataset','GH_CRISPR'])['anchor'].nunique().reset_index().rename(columns={'anchor':'CRISPR_anchs'})
r = fir.groupby(['dataset','GH_MGE'])['anchor'].nunique().reset_index().rename(columns={'anchor':'MGE_anchs'})
l[['GH_MGE','MGE_anchs']] = r[['GH_MGE','MGE_anchs']] 
l.loc[16]['dataset']
```

#### Ok, we're picking V. cholerae.


```python
os.mkdir('vibrio_satc')
v = fir[fir['dataset']==l.loc[16]['dataset']].reset_index(drop=True)
cat = []
for i in range(len(v)):
    if v['GH_CRISPR'][i]: 
        cat.append("CRISPR")
    elif v['GH_MGE'][i]:
        cat.append("MGE")
    else:
        cat.append("unclassified")
v['class'] = cat
v[['anchor']].to_csv('vibrio_satc/anchor_list.txt',sep='\t',index=None,header=None)
satcs = glob.glob('/oak/stanford/groups/horence/punit/bioinformatics/results/vibrio_cholerae_v2.0.3_10_targets_George_rerun_10302023/kmc-nomad/result_satc/*')

for i in satcs:
    
    command = '/oak/stanford/groups/horence/george/splash_2.3.0/satc_dump --anchor_list vibrio_satc/anchor_list.txt '+str(i)+' vibrio_satc/'+i.split('/')[-1]
    os.system(command)
satcs = glob.glob('vibrio_satc/*bin*')
satc = pd.read_csv(satcs[0],sep='\t',header=None)
for i in satcs[1:]:
    satc = pd.concat([satc,pd.read_csv(i,sep='\t',header=None)])
satc = satc.rename(columns={0:'sample',1:'anchor',2:'target',3:'count'})
select = satc[satc['anchor']=='TAAGCTTCAACACTTTACATTTGAACC']
```


```python
construct_graph(data, graphName, nodeType, nodeFeatures, edgeFeatures, connectedness, ordering):

```
