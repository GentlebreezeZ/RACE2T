<h1 align="center">
  RACE2T
</h1>
<h4 align="center">Knowledge Graph Entity Type Prediction with Relational Aggregation Graph Attention Network.</h4>

<h2 align="center">
  The framework of RACE2T
  <img align="center"  src="https://github.com/GentlebreezeZ/RACE2T/blob/main/Framework.jpg" alt="...">
</h2>


### Requirements
* Python 3.8
* torch==1.8.0
* torch_scatter==2.0.4

### Entity Type prediction Results
Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15KET | 0.6456 | 0.5607 | 0.6884 | 0.8172
YAGO43KET | 0.3437 | 0.2482 | 0.3762 | 0.5230

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
RACE2T (for FB15K and FB15KET):
- python main.py -num_layers 2 -nb_heads 2 -embsize_entity_type 200 -hidden_embedding_size 300 -output_embedding_size 600 -frgat_computing_model TransE -frgat_composition_operator sub -learningrate 0.0001
- python main.py -num_layers 2 -nb_heads 2 -embsize_entity_type 200 -hidden_embedding_size 300 -output_embedding_size 600 -frgat_computing_model DisMult -frgat_composition_operator mult -learningrate 0.0001
- python main.py -num_layers 2 -nb_heads 2 -embsize_entity_type 200 -hidden_embedding_size 300 -output_embedding_size 600 -frgat_computing_model DisMult -frgat_composition_operator sub -learningrate 0.0001

RACE2T (for YAGO43K and YAGO43KET):
- python main.py -num_layers 1 -nb_heads 1 -embsize_entity_type 200 -hidden_embedding_size 200 -output_embedding_size 600 -frgat_computing_model TransE -frgat_composition_operator sub -learningrate 0.00055
- python main.py -num_layers 1 -nb_heads 1 -embsize_entity_type 200 -hidden_embedding_size 200 -output_embedding_size 600 -frgat_computing_model DisMult -frgat_composition_operator mult -learningrate 0.00055
### Detailed optimal parameters
- YAGO43K(more detailed parameters are set in the config.py):
epochs:500
batchsize:256
num_filters:64
filt_h:1
filt_w:2
conv_stride:2
learningrate:0.00055
droupt_output_decoder:0.4
label_smoothing_decoder:0.1
embsize_entity:100
embsize_relation:100
embsize_entity_type:200
num_layers:1
nb_heads:1
hidden_embedding_size:200
output_embedding_size:600
alpha:0.2
frgat_drop:0.3
frgat_output_dropout:0.2
frgat_initial_dropout:0.0
frgat_computing_model:'DisMult'
frgat_composition_operator:'mult'

- FB15K(more detailed parameters are set in the config.py):

epochs:500
batchsize:128
num_filters:64
filt_h:1
filt_w:2
conv_stride:2
learningrate:0.0001
droupt_output_decoder:0.4
label_smoothing_decoder:0.1
embsize_entity:100
embsize_relation:100
embsize_entity_type:200
num_layers:2
nb_heads:2
hidden_embedding_size:300
output_embedding_size:600
alpha:0.2
frgat_drop:0.3
frgat_output_dropout:0.2
frgat_initial_dropout:0.0
frgat_computing_model:'TransE'
frgat_composition_operator:'sub'
### How to test
- The relevant code of the test is in main.py. When running 400 epochs, it is tested every 5 epochs.

#### When using this codebase please cite:
```
@inproceedings{RACE2T,
  author    = {Changlong Zou and
               Jingmin An and
               Guanyu Li},
  title     = {Knowledge Graph Entity Type Prediction with Relational Aggregation
               Graph Attention Network},
  booktitle = {The Semantic Web - 19th International Conference, {ESWC} 2022, Hersonissos,
               Crete, Greece, May 29 - June 2, 2022, Proceedings},
  volume    = {13261},
  pages     = {39--55},
  year      = {2022},
  doi       = {10.1007/978-3-031-06981-9\_3}
}
```
