from copy import deepcopy
import torch
import numpy as np
from model import CE2T, ConnectE, ETE, RACE2T
from datetime import datetime
import random
from collections import defaultdict
import config
from torch.optim.lr_scheduler import ExponentialLR
from logger_init import get_logger

logger = get_logger('train', True, file_log=True)
logger.info('START TIME : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


# set gpu if you need
# torch.cuda.set_device(0)


class Experiment:
    def __init__(self, decay_rate=0.99, batch_size=128, learning_rate=0.001, entity_embedding_dim=200,
                 relation_embedding_dim=200, entity_type_embedding_dim=100,
                 epochs=50000, num_filters=200, droupt_output=0.2,
                 label_smoothing=0.1, cuda=True, filt_h=1, filt_w=9, conv_stride=1, pool_h=2, pool_w=2,
                 pool_stride=2, decoder_model_name='CE2T', margin=2.0, frgat_heads=2, frgat_alpha=0.2,
                 frgat_dropout=0.3, frgat_output_dropout=0.2, frgat_initial_dropout=0.2, frgat_layers=2,
                 frgat_computing_model='TransE', frgat_composition_operator='sub', hidden_size=300, output_size=600):
        # decoder
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_entity_dim = entity_embedding_dim
        self.embedding_relation_dim = relation_embedding_dim
        self.embedding_entity_type_dim = entity_type_embedding_dim
        self.epochs = epochs
        self.num_filters = num_filters
        self.num_entity = len(config.d.entity_idxs)
        self.num_entity_type = len(config.d.entity_types_idxs)
        self.droupt_output = droupt_output
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.filt_h = filt_h
        self.filt_w = filt_w
        self.conv_stride = conv_stride
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pool_stride = pool_stride
        self.decay_rate = decay_rate
        self.decoder_model_name = decoder_model_name
        self.margin = margin
        # result
        self.MRR = list()
        self.HITS1 = list()
        self.HITS3 = list()
        self.HITS10 = list()
        # 1-1
        self.MRR_1 = list()
        self.HITS1_1 = list()
        self.HITS3_1 = list()
        self.HITS10_1 = list()
        # 1-N
        self.MRR_N = list()
        self.HITS1_N = list()
        self.HITS3_N = list()
        self.HITS10_N = list()
        # gat
        self.frgat_heads = frgat_heads
        self.frgat_layers = frgat_layers
        self.frgat_alpha = frgat_alpha
        self.frgat_dropout = frgat_dropout
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.frgat_output_dropout = frgat_output_dropout
        self.frgat_initial_dropout = frgat_initial_dropout
        self.frgat_computing_model = frgat_computing_model
        self.frgat_composition_operator = frgat_composition_operator
        logger.info('-------------frgat_layers-------------: {} '.format(frgat_layers))
        logger.info('-------------frgat_heads-------------: {} '.format(frgat_heads))
        logger.info('-------------frgat_alpha-------------: {} '.format(frgat_alpha))
        logger.info('-------------hidden_size-------------: {} '.format(hidden_size))
        logger.info('-------------frgat_output_size-------------: {} '.format(output_size))
        logger.info('-------------frgat_dropout-------------: {} '.format(frgat_dropout))
        logger.info('-------------frgat_initial_dropout-------------: {} '.format(frgat_initial_dropout))
        logger.info('-------------frgat_output_dropout-------------: {} '.format(frgat_output_dropout))
        logger.info('-------------frgat_computing_model-------------: {} '.format(frgat_computing_model))
        logger.info('-------------frgat_composition_operator-------------: {} '.format(frgat_composition_operator))
        logger.info('-------------decoder-------------')
        logger.info('-------------model_name-------------: {} '.format(decoder_model_name))
        logger.info('embedding_entity_dim: {} '.format(self.embedding_entity_dim))
        logger.info('embedding_entity_type_dim: {} '.format(self.embedding_entity_type_dim))
        logger.info('batch_size: {} '.format(batch_size))
        logger.info('learning_rate: {} '.format(learning_rate))
        logger.info('num_filters: {} '.format(num_filters))
        logger.info('droupt_output: {} '.format(droupt_output))
        logger.info('label_smoothing: {} '.format(label_smoothing))
        logger.info('filt_h: {} '.format(filt_h))
        logger.info('filt_w: {} '.format(filt_w))
        logger.info('pool_h: {} '.format(pool_h))
        logger.info('pool_w: {} '.format(pool_w))
        logger.info('pool_stride: {} '.format(pool_stride))

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for et in data:
            er_vocab[(et[0])].append(et[1])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:min(idx + self.batch_size, len(er_vocab_pairs))]
        targets = np.zeros((len(batch), len(config.d.types)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data, entity_embedding):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = data
        er_vocab = self.get_er_vocab(config.d.over_et_data)

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e_idx = torch.tensor(data_batch[:, 0], dtype=torch.long)
            t_idx = torch.tensor(data_batch[:, 1], dtype=torch.long)
            if self.cuda:
                e_idx = e_idx.cuda()
                t_idx = t_idx.cuda()
            predictions = model.predict(e_idx, entity_embedding)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0])]
                target_value = predictions[j, t_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, t_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == t_idx[j])[0][0].cpu().item()
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info('Hits @10: {0}'.format(np.mean(hits[9])))
        self.HITS10.append(np.mean(hits[9]))
        logger.info('Hits @3: {0}'.format(np.mean(hits[2])))
        self.HITS3.append(np.mean(hits[2]))
        logger.info('Hits @1: {0}'.format(np.mean(hits[0])))
        self.HITS1.append(np.mean(hits[0]))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        self.MRR.append(np.mean(1. / np.array(ranks)))

    def evaluate_1_1(self, model, data, entity_embedding):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = data
        er_vocab = self.get_er_vocab(config.d.over_et_data)

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e_idx = torch.tensor(data_batch[:, 0], dtype=torch.long)
            t_idx = torch.tensor(data_batch[:, 1], dtype=torch.long)
            if self.cuda:
                e_idx = e_idx.cuda()
                t_idx = t_idx.cuda()
            predictions = model.predict(e_idx, entity_embedding)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0])]
                target_value = predictions[j, t_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, t_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == t_idx[j])[0][0].cpu().item()
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info('1-1 Hits @10: {0}'.format(np.mean(hits[9])))
        self.HITS10_1.append(np.mean(hits[9]))
        logger.info('1-1 Hits @3: {0}'.format(np.mean(hits[2])))
        self.HITS3_1.append(np.mean(hits[2]))
        logger.info('1-1 Hits @1: {0}'.format(np.mean(hits[0])))
        self.HITS1_1.append(np.mean(hits[0]))
        logger.info('1-1 Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        self.MRR_1.append(np.mean(1. / np.array(ranks)))

    def evaluate_1_N(self, model, data, entity_embedding):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = data
        er_vocab = self.get_er_vocab(config.d.over_et_data)

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e_idx = torch.tensor(data_batch[:, 0], dtype=torch.long)
            t_idx = torch.tensor(data_batch[:, 1], dtype=torch.long)
            if self.cuda:
                e_idx = e_idx.cuda()
                t_idx = t_idx.cuda()
            predictions = model.predict(e_idx, entity_embedding)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0])]
                target_value = predictions[j, t_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, t_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == t_idx[j])[0][0].cpu().item()
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info('1-N Hits @10: {0}'.format(np.mean(hits[9])))
        self.HITS10_N.append(np.mean(hits[9]))
        logger.info('1-N Hits @3: {0}'.format(np.mean(hits[2])))
        self.HITS3_N.append(np.mean(hits[2]))
        logger.info('1-N Hits @1: {0}'.format(np.mean(hits[0])))
        self.HITS1_N.append(np.mean(hits[0]))
        logger.info('1-N Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        self.MRR_N.append(np.mean(1. / np.array(ranks)))

    def get_max_mrr_hits(self):
        logger.info('MAX Hits @10: {0}'.format(max(self.HITS10)))
        logger.info('MAX Hits @3: {0}'.format(max(self.HITS3)))
        logger.info('MAX Hits @1: {0}'.format(max(self.HITS1)))
        logger.info('MAX Mean reciprocal rank: {0}'.format(max(self.MRR)))

    def get_max_mrr_hits_1_1_and_1_N(self):
        logger.info('1-1 MAX Hits @10: {0}'.format(max(self.HITS10_1)))
        logger.info('1-1 MAX Hits @3: {0}'.format(max(self.HITS3_1)))
        logger.info('1-1 MAX Hits @1: {0}'.format(max(self.HITS1_1)))
        logger.info('1-1 MAX Mean reciprocal rank: {0}'.format(max(self.MRR_1)))
        logger.info('----------------------------------------------')
        logger.info('1-N MAX Hits @10: {0}'.format(max(self.HITS10_N)))
        logger.info('1-N MAX Hits @3: {0}'.format(max(self.HITS3_N)))
        logger.info('1-N MAX Hits @1: {0}'.format(max(self.HITS1_N)))
        logger.info('1-N MAX Mean reciprocal rank: {0}'.format(max(self.MRR_N)))

    def train_and_eval(self):
        er_vocab = self.get_er_vocab(config.d.train_et_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        model = RACE2T(entity_num=len(config.d.entity_idxs), relation_num=len(config.d.relation_idxs) + 1,
                       entity_type_num=len(config.d.entity_types_idxs), de=self.embedding_entity_dim,
                       dr=self.embedding_relation_dim, dt=self.embedding_entity_type_dim,
                       nfeat_dim=self.embedding_entity_dim, nhidden_dim=self.hidden_size,
                       nout_dim=self.output_size, frgat_droupt=self.frgat_dropout,
                       frgat_output_dropout=self.frgat_output_dropout, frgat_initial_dropout=self.frgat_initial_dropout,
                       alpha=self.frgat_alpha, frgat_computing_model=self.frgat_computing_model,
                       frgat_composition_operator=self.frgat_composition_operator, frgat_layers=self.frgat_layers,
                       nheads=self.frgat_heads, droupt_output_ce2T=self.droupt_output, filt_h=self.filt_h,
                       filt_w=self.filt_w, num_filters=self.num_filters, conv_stride=self.conv_stride,
                       pool_h=self.pool_h, pool_w=self.pool_w, pool_stride=self.pool_stride)
        print("----------------------------------------updatable parameters----------------------------------------")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        print("----------------------------------------------------------------------------------------------------")

        if self.cuda:
            model.cuda()

        graph_edge_index = torch.LongTensor(config.d.graph_edge_index).cuda().t()
        graph_edge_type = torch.LongTensor(config.d.graph_edge_type).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        for it in range(1, self.epochs + 1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e_idx = torch.tensor(data_batch, dtype=torch.long)
                if self.cuda:
                    e_idx = e_idx.cuda()
                predictions = model.forward(e_idx, graph_edge_index, graph_edge_type)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())

            logger.info('epoch: {} train_loss: {}'.format(it, np.mean(losses)))
            model.eval()
            with torch.no_grad():
                if it >= 500 and it % 10 == 0:
                    entity_embedding = model.forward_base_spgat(graph_edge_index, graph_edge_type)
                    logger.info('------------------------------------it:{}'.format(it))
                    self.evaluate(model, config.d.test_et_idxs, entity_embedding)
                    self.evaluate_1_1(model, config.d.test_data_1_1, entity_embedding)
                    self.evaluate_1_N(model, config.d.test_data_1_N, entity_embedding)
        self.get_max_mrr_hits()
        self.get_max_mrr_hits_1_1_and_1_N()


if __name__ == '__main__':
    experiment = Experiment(batch_size=config.args.batchsize, learning_rate=config.args.learningrate,
                            entity_embedding_dim=config.args.embsize_entity,
                            relation_embedding_dim=config.args.embsize_relation,
                            entity_type_embedding_dim=config.args.embsize_entity_type,
                            epochs=config.args.epochs, num_filters=config.args.num_filters,
                            droupt_output=config.args.droupt_output_decoder,
                            label_smoothing=config.args.label_smoothing_decoder,
                            cuda=config.args.CUDA, filt_h=config.args.filt_h, filt_w=config.args.filt_w,
                            pool_h=config.args.pool_h, pool_w=config.args.pool_w,
                            pool_stride=config.args.pool_stride, conv_stride=config.args.conv_stride,
                            decoder_model_name=config.args.decoder_model_name, margin=config.args.margin,
                            frgat_heads=config.args.nb_heads, frgat_layers=config.args.num_layers,
                            frgat_alpha=config.args.alpha,
                            frgat_computing_model=config.args.frgat_computing_model,
                            frgat_composition_operator=config.args.frgat_composition_operator,
                            frgat_dropout=config.args.frgat_drop, frgat_output_dropout=config.args.frgat_output_dropout,
                            hidden_size=config.args.hidden_embedding_size,
                            output_size=config.args.output_embedding_size,
                            frgat_initial_dropout=config.args.frgat_initial_dropout)
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    experiment.train_and_eval()
