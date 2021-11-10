<h1 align="center">
  RACE2T
</h1>

### Requirements
The codebase is implemented in Python 3.8 or 3.6. Required packages is:
    pytorch    1.8(gpu) or pytorch 1.9(gpu), torch_scatter
    
### Dataset:
- We use FB15K and YAGO43K dataset for FRGAT. 
- We use FB15KET and YAGO43KET dataset for knowledge graph entity type prediction. 


### Entity Type prediction Results
Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15KET | 0.6456 | 0.5607 | 0.6884 | 0.8172
YAGO43KET | 0.3437 | 0.2482 | 0.3762 | 0.5230


