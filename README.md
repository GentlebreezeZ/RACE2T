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
# RACE2T (for FB15K and FB15KET)
- python run.py -num_layers 2 -nb_heads 2 -hidden_embedding_size 300 -output_embedding_size 600 -frgat_computing_model TransE -frgat_composition_operator sub -learningrate 0.0001
- python run.py -num_layers 2 -nb_heads 2 -hidden_embedding_size 300 -output_embedding_size 600 -frgat_computing_model DisMult -frgat_composition_operator mult -learningrate 0.0001
- python run.py -num_layers 2 -nb_heads 2 -hidden_embedding_size 300 -output_embedding_size 600 -frgat_computing_model DisMult -frgat_composition_operator sub -learningrate 0.0001
# RACE2T (for YAGO43K and YAGO43KET)
- python run.py -num_layers 1 -nb_heads 1 -hidden_embedding_size 200 -output_embedding_size 600 -frgat_computing_model TransE -frgat_composition_operator sub -learningrate 0.00055
- python run.py -num_layers 2 -nb_heads 2 -hidden_embedding_size 200 -output_embedding_size 600 -frgat_computing_model DisMult -frgat_composition_operator mult -learningrate 0.00055
### Detailed optimal parameters
#YAGO
# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', default=500, help='Number of epochs (default: 200)')
# parser.add_argument('--batchsize', type=int, default=256, help='Batch size (default: 128)')
# parser.add_argument('--num_filters', type=int, default=64, help='number of filters CNN')
# parser.add_argument('--filt_h', type=int, default=1, help='filt_h of  CNN')
# parser.add_argument('--filt_w', type=int, default=2, help='filt_w of CNN')
# parser.add_argument('--conv_stride', type=int, default=2, help='stride of CNN')
# parser.add_argument('--pool_h', type=int, default=1, help='pool_h')
# parser.add_argument('--pool_w', type=int, default=2, help='pool_w')
# parser.add_argument('--pool_stride', type=int, default=2, help='pool_w')
# parser.add_argument('--learningrate', default=0.00055, help='Learning rate (default: 0.00005)')
# parser.add_argument("--droupt_output_decoder", default=0.4, type=float, help="droupt regularization item 0.2")
# parser.add_argument("--label_smoothing_decoder", default=0.1, type=float, help="label smoothing")
# parser.add_argument("--CUDA", default=True, type=bool, help="True:GPU False:CPU")
# parser.add_argument("--decoder_model_name", default='CE2T', type=str, help="CE2T ConnectE ETE")
# parser.add_argument("--margin", default=2.0, type=float, help="ConnectE ETE’s margine ")
# parser.add_argument('--embsize_entity', default=100, help='Entity Embedding size (default: 200)')
# parser.add_argument('--embsize_relation', default=100, help='Entity Embedding size (default: 200)')
# parser.add_argument('--embsize_entity_type', default=200, help='Entity Type Embedding size (default: 100)')
# parser.add_argument('--num_layers', type=int, default=1, help='Number of FRGAT layers')
# parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
# parser.add_argument('--hidden_embedding_size', type=int, default=200, help='Number of hidden units.')
# parser.add_argument('--output_embedding_size', type=int, default=600, help='Number of input units.')
# parser.add_argument("--alpha", "--alpha", type=float, default=0.2, help="LeakyRelu alphs for FRGAT layer")
# parser.add_argument('--frgat_drop', default=0.3, type=float, help='FRGAT dropout')
# parser.add_argument('--frgat_output_dropout', default=0.2, type=float, help='FRGAT output dropout')
# parser.add_argument('--frgat_initial_dropout', default=0.0, type=float, help='FRGAT initial dropout')
# parser.add_argument('--frgat_computing_model', default='DisMult', type=str,
#                     help='Attention computing model:TransE TransH TransD DisMult RotatE QuatE dot')
# parser.add_argument('--frgat_composition_operator', default='mult', type=str, help='sub mult corr void')

#FB15K
# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', default=500, help='Number of epochs (default: 200)')
# parser.add_argument('--batchsize', type=int, default=128, help='Batch size (default: 128)')
# parser.add_argument('--num_filters', type=int, default=64, help='number of filters CNN')
# parser.add_argument('--filt_h', type=int, default=1, help='filt_h of  CNN')
# parser.add_argument('--filt_w', type=int, default=2, help='filt_w of CNN')
# parser.add_argument('--conv_stride', type=int, default=2, help='stride of CNN')
# parser.add_argument('--pool_h', type=int, default=1, help='pool_h')
# parser.add_argument('--pool_w', type=int, default=2, help='pool_w')
# parser.add_argument('--pool_stride', type=int, default=2, help='pool_w')
# parser.add_argument('--learningrate', default=0.0001, help='Learning rate (default: 0.00005)')
# parser.add_argument("--droupt_output_decoder", default=0.4, type=float, help="droupt regularization item 0.2")
# parser.add_argument("--label_smoothing_decoder", default=0.1, type=float, help="label smoothing")
# parser.add_argument("--CUDA", default=True, type=bool, help="True:GPU False:CPU")
# parser.add_argument("--decoder_model_name", default='CE2T', type=str, help="CE2T ConnectE ETE")
# parser.add_argument("--margin", default=2.0, type=float, help="ConnectE ETE’s margine ")
# parser.add_argument('--embsize_entity', default=100, help='Entity Embedding size (default: 200)')
# parser.add_argument('--embsize_relation', default=100, help='Entity Embedding size (default: 200)')
# parser.add_argument('--embsize_entity_type', default=200, help='Entity Type Embedding size (default: 100)')
# parser.add_argument('--num_layers', type=int, default=2, help='Number of FRGAT layers')
# parser.add_argument('--nb_heads', type=int, default=2, help='Number of head attentions.')
# parser.add_argument('--hidden_embedding_size', type=int, default=300, help='Number of hidden units.')
# parser.add_argument('--output_embedding_size', type=int, default=600, help='Number of input units.')
# parser.add_argument("--alpha", "--alpha", type=float, default=0.2, help="LeakyRelu alphs for FRGAT layer")
# parser.add_argument('--frgat_drop', default=0.3, type=float, help='FRGAT dropout')
# parser.add_argument('--frgat_output_dropout', default=0.2, type=float, help='FRGAT output dropout')
# parser.add_argument('--frgat_initial_dropout', default=0.0, type=float, help='FRGAT initial dropout')
# parser.add_argument('--frgat_computing_model', default='DisMult', type=str,
#                     help='Attention computing model:TransE TransH TransD DisMult RotatE QuatE dot')
# parser.add_argument('--frgat_composition_operator', default='sub', type=str, help='sub mult corr void')
