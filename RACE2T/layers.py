import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

pi = 3.14159265358979323846


def ccorr(a, b):
    # return torch.fft.irfft(com_mult(torch.conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
    return torch.fft.ifft(torch.conj(torch.fft.fft(a) * torch.fft.fft(b))).real


def scatter_(name, src, index, dim_size=None):
    if name == 'add': name = 'sum'
    assert name in ['sum', 'mean', 'max']
    out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
    return out[0] if isinstance(out, tuple) else out


class FRGAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, computing_model, composition_operator, concat=True,
                 kg_entity_num=0,
                 kg_relation_num=0):
        super(FRGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.computing_model = computing_model
        self.composition_operator = composition_operator

        if self.computing_model == "TransE":
            self.pfun = self.TransE_model_function
        elif self.computing_model == "TransH":
            self.Wr = torch.nn.Embedding(kg_relation_num, out_features, padding_idx=0)  # def project Wr
            torch.nn.init.xavier_uniform_(self.Wr.weight.data)
            self.pfun = self.TransH_model_function
        elif self.computing_model == "TransD":
            self.entity_mappings = torch.nn.Embedding(kg_entity_num, out_features, padding_idx=0)
            self.relation_mappings = torch.nn.Embedding(kg_relation_num, out_features, padding_idx=0)
            torch.nn.init.xavier_uniform_(self.entity_mappings.weight.data)
            torch.nn.init.xavier_uniform_(self.relation_mappings.weight.data)
            self.pfun = self.TransD_model_function
        elif self.computing_model == "DisMult":
            self.pfun = self.DisMult_model_function
        elif self.computing_model == "RotatE":
            self.entity_embeddings_imag = torch.nn.Embedding(kg_entity_num, out_features, padding_idx=0)
            torch.nn.init.xavier_uniform_(self.entity_embeddings_imag.weight.data)
            self.embedding_range = (6.0 + 2.0) / out_features
            self.pfun = self.RotatE_model_function
        elif self.computing_model == "QuatE":
            self.emb_x_a = nn.Embedding(kg_entity_num, out_features, padding_idx=0)
            self.emb_y_a = nn.Embedding(kg_entity_num, out_features, padding_idx=0)
            self.emb_z_a = nn.Embedding(kg_entity_num, out_features, padding_idx=0)
            self.rel_x_b = nn.Embedding(kg_relation_num, out_features, padding_idx=0)
            self.rel_y_b = nn.Embedding(kg_relation_num, out_features, padding_idx=0)
            self.rel_z_b = nn.Embedding(kg_relation_num, out_features, padding_idx=0)
            torch.nn.init.xavier_uniform_(self.emb_x_a.weight.data)
            torch.nn.init.xavier_uniform_(self.emb_y_a.weight.data)
            torch.nn.init.xavier_uniform_(self.emb_z_a.weight.data)
            torch.nn.init.xavier_uniform_(self.rel_x_b.weight.data)
            torch.nn.init.xavier_uniform_(self.rel_y_b.weight.data)
            torch.nn.init.xavier_uniform_(self.rel_z_b.weight.data)
            self.pfun = self.QuatE_model_function
        elif self.computing_model == "dot":
            self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
            nn.init.xavier_normal_(self.a.data)
            self.pfun = self.initial_attention_function
        else:
            raise NotImplementedError

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def TransE_model_function(self, entity_embedding, relation_embedding, h, r, t):
        head_embedding = entity_embedding[h, :]
        rel_embedding = relation_embedding[r]
        tail_embedding = entity_embedding[t, :]
        assert not torch.isnan(head_embedding).any()
        assert not torch.isnan(rel_embedding).any()
        assert not torch.isnan(tail_embedding).any()
        return torch.norm(head_embedding + rel_embedding - tail_embedding, p=2, dim=1)

    def TransH_model_function(self, entity_embedding, relation_embedding, h, r, t):
        head_embedding = entity_embedding[h, :]
        rel_embedding = relation_embedding[r]
        tail_embedding = entity_embedding[t, :]
        Wr = self.Wr(r)
        assert not torch.isnan(head_embedding).any()
        assert not torch.isnan(rel_embedding).any()
        assert not torch.isnan(tail_embedding).any()
        assert not torch.isnan(Wr).any()
        proj_vec = F.normalize(Wr, p=2, dim=-1)
        head_embedding = head_embedding - torch.sum(head_embedding * proj_vec, dim=-1, keepdims=True) * proj_vec
        tail_embedding = tail_embedding - torch.sum(tail_embedding * proj_vec, dim=-1, keepdims=True) * proj_vec
        return torch.norm(head_embedding + rel_embedding - tail_embedding, p=2, dim=1)

    def TransD_model_function(self, entity_embedding, relation_embedding, h, r, t):
        head_embedding = entity_embedding[h, :]
        rel_embedding = relation_embedding[r]
        tail_embedding = entity_embedding[t, :]
        head_mapping = self.entity_mappings(h)
        rel_mapping = self.entity_mappings(r)
        tail_mapping = self.entity_mappings(t)
        assert not torch.isnan(head_embedding).any()
        assert not torch.isnan(rel_embedding).any()
        assert not torch.isnan(tail_embedding).any()
        assert not torch.isnan(head_mapping).any()
        assert not torch.isnan(rel_mapping).any()
        assert not torch.isnan(tail_mapping).any()
        head_embedding = head_embedding + torch.sum(head_embedding * head_mapping, axis=-1, keepdims=True) * rel_mapping
        tail_embedding = tail_embedding + torch.sum(tail_embedding * tail_mapping, axis=-1, keepdims=True) * rel_mapping
        return torch.norm(head_embedding + rel_embedding - tail_embedding, p=2, dim=1)

    def DisMult_model_function(self, entity_embedding, relation_embedding, h, r, t):
        head_embedding = entity_embedding[h, :]
        rel_embedding = relation_embedding[r]
        tail_embedding = entity_embedding[t, :]
        assert not torch.isnan(head_embedding).any()
        assert not torch.isnan(rel_embedding).any()
        assert not torch.isnan(tail_embedding).any()
        return torch.sum(head_embedding * rel_embedding * tail_embedding, -1)

    def RotatE_model_function(self, entity_embedding, relation_embedding, h, r, t):
        h_e_r = entity_embedding[h, :]
        h_e_i = self.entity_embeddings_imag(h)
        r_e_r = relation_embedding[r]
        t_e_r = entity_embedding[t, :]
        t_e_i = self.entity_embeddings_imag(t)
        assert not torch.isnan(h_e_r).any()
        assert not torch.isnan(r_e_r).any()
        assert not torch.isnan(t_e_r).any()
        assert not torch.isnan(h_e_i).any()
        assert not torch.isnan(t_e_i).any()
        r_e_r = r_e_r / (self.embedding_range / pi)
        r_e_i = torch.sin(r_e_r)
        r_e_r = torch.cos(r_e_r)
        score_r = h_e_r * r_e_r - h_e_i * r_e_i - t_e_r
        score_i = h_e_r * r_e_i + h_e_i * r_e_r - t_e_i
        return (6.0 - torch.sum(score_r ** 2 + score_i ** 2, axis=-1))

    def quate_calc(self, s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b):
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        score_r = (A * s_c + B * x_c + C * y_c + D * z_c)

        return torch.sum(score_r, -1)

    def QuatE_model_function(self, entity_embedding, relation_embedding, h, r, t):
        s_a = entity_embedding[h, :]
        x_a = self.emb_x_a(h)
        y_a = self.emb_y_a(h)
        z_a = self.emb_z_a(h)

        s_c = entity_embedding[t, :]
        x_c = self.emb_x_a(t)
        y_c = self.emb_y_a(t)
        z_c = self.emb_z_a(t)

        s_b = relation_embedding[r]
        x_b = self.rel_x_b(r)
        y_b = self.rel_y_b(r)
        z_b = self.rel_z_b(r)
        return self.quate_calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)

    def initial_attention_function(self, entity_embedding, relation_embedding, h, r, t):
        head_embedding = entity_embedding[h, :]
        rel_embedding = relation_embedding[r]
        tail_embedding = entity_embedding[t, :]
        assert not torch.isnan(head_embedding).any()
        assert not torch.isnan(rel_embedding).any()
        assert not torch.isnan(tail_embedding).any()
        edge_h = torch.cat((head_embedding, tail_embedding), dim=1).t()
        return self.a.mm(edge_h).squeeze()

    def rel_composition_operator(self, ent_embed, rel_embed, mode="void"):
        if mode == "sub":
            com_embed = ent_embed - rel_embed
        elif mode == "mult":
            com_embed = ent_embed * rel_embed
        elif mode == "corr":
            com_embed = ccorr(ent_embed, rel_embed)
        elif mode == "void":
            com_embed = ent_embed
        else:
            raise NotImplementedError
        return com_embed

    def message(self, x_j, edge_type, rel_embedd, edge_attention):
        rel_emb = torch.index_select(rel_embedd, 0, edge_type)
        out = self.rel_composition_operator(x_j, rel_emb, self.composition_operator)
        return out if edge_attention is None else out * edge_attention.view(-1, 1)

    def propagate(self, edge_index, edge_type, entity_embed, rel_embed, edge_attentrion):
        tail_embedding = entity_embed[edge_index[1, :]]
        out = self.message(tail_embedding, edge_type, rel_embed, edge_attentrion)
        out = scatter_("add", out, edge_index[0, :], dim_size=entity_embed.shape[0])
        return out

    def calc_total_edge_num(self, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape).to_dense()
        b = b.cuda()
        return torch.matmul(a, b)

    def forward(self, entity_embedding, relation_embedding, edge_index, edge_type):
        N = entity_embedding.size()[0]

        entity_embedding = torch.mm(entity_embedding, self.W)
        relation_embedding = torch.mm(relation_embedding, self.W)

        edge_e = torch.exp(-self.leakyrelu(
            self.pfun(entity_embedding, relation_embedding, edge_index[0, :], edge_type, edge_index[1, :])))

        assert not torch.isnan(edge_e).any()

        edge_e = self.dropout(edge_e)

        e_rowsum = self.calc_total_edge_num(edge_index, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)))
        e_rowsum[e_rowsum == 0.0] = 1e-12

        h_prime = self.propagate(edge_index, edge_type, entity_embedding, relation_embedding, edge_e)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)

        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
