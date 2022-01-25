<h1 align="center">
  RACE2T
</h1>

### Entity Type prediction Results
Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15KET | 0.6456 | 0.5607 | 0.6884 | 0.8172
YAGO43KET | 0.3437 | 0.2482 | 0.3762 | 0.5230

Parameter Setup

### Dependencies
- Dependencies can be installed using `requirements.txt`.
### Datasets 
- FB15K,FB15KET,YAGO43K and YAGO43KET are stored in folder 'data'(this folder contains compressed files, which need to be decompressed before use).
- We use FB15K and YAGO43K dataset for FRGAT. 
- We use FB15KET and YAGO43KET dataset for knowledge graph entity type prediction. 
### Source file
- config.py: Parameter configuration
- load_data.py: Load dataset file (equivalent to data_loader).
- logger_init.py: log file.
- layers.py: this file defines the operation process of FRGAT.
- model.py: this file defines CE2T model, RACE2T model, etc.
- main.py: this file is used for model training and testing.
### How to run
