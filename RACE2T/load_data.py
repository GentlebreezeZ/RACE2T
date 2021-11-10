import random


class Data:
    def __init__(self, data_dir="data/FB15k/"):
        ###############################################entity type#################################################################
        self.train_type_data = self.load_type_data(data_dir, "Entity_Type_train")
        self.valid_type_data = self.load_type_data(data_dir, "Entity_Type_valid")
        self.test_type_data = self.load_type_data(data_dir, "Entity_Type_test")
        self.type_data = self.train_type_data + self.valid_type_data + self.test_type_data
        self.types = self.get_types(self.type_data)

        #entity type id
        self.entity_types_idxs = {self.types[i]: i for i in range(len(self.types))}
        #self.entity_types_idxs = self.encode_type_to_id(data_dir)
        self.types_entity_idxs = {v: k for k, v in self.entity_types_idxs.items()}

        ###########################################################################################################################
        self.train_triplet_data = self.load_triplet_data(data_dir, "train", reverse=False)
        self.valid_triplet_data = self.load_triplet_data(data_dir, "valid", reverse=False)
        self.test_triplet_data = self.load_triplet_data(data_dir, "test", reverse=False)
        self.triplet_data = self.train_triplet_data + self.valid_triplet_data + self.test_triplet_data

        self.entities = self.get_triplet_entities(self.triplet_data)

        self.train_relations = self.get_triplet_relations(self.train_triplet_data)
        self.valid_relations = self.get_triplet_relations(self.valid_triplet_data)
        self.test_relations = self.get_triplet_relations(self.test_triplet_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                                                 if i not in self.train_relations] + [i for i in self.test_relations \
                                                                                      if i not in self.train_relations]
        # entity id
        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        #self.entity_idxs = self.encode_entity_to_id(data_dir)
        #relation id
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}
        #self.relation_idxs = self.encode_relation_to_id(data_dir)
        ###########################################################################################################################
        self.train_et_idxs = self.get_type_data_idxs(self.train_type_data)
        random.shuffle(self.train_et_idxs)
        self.valid_et_idxs = self.get_type_data_idxs(self.valid_type_data)
        self.test_et_idxs = self.get_type_data_idxs(self.test_type_data)

        self.over_et_data = self.train_et_idxs + self.valid_et_idxs + self.test_et_idxs
        #create 1-1 and 1-N test data
        self.type_to_entity_dict = self.get_type_to_entity(self.train_et_idxs)
        self.type_to_entity_dict2 = self.get_type_to_entity(self.valid_et_idxs)
        self.type_to_entity_dict3 = self.get_type_to_entity(self.test_et_idxs)
        self.entity_to_type_dict = self.get_type_data_idxs_dict(self.train_et_idxs)
        self.entity_to_type_dict2 = self.get_type_data_idxs_dict(self.valid_et_idxs)
        self.entity_to_type_dict3 = self.get_type_data_idxs_dict(self.test_et_idxs)
        self.test_data_1_1, self.test_data_1_N = self.get_1_1_OR_1_N_test_data(self.entity_to_type_dict3)
        random.shuffle(self.test_data_1_1)
        random.shuffle(self.test_data_1_N)
        ###########################################################################################################################
        self.train_triplet_data_idxs = self.get_triplet_data_idxs(self.train_triplet_data)
        self.valid_triplet_data_idxs = self.get_triplet_data_idxs(self.valid_triplet_data)
        self.test_triplet_data_idxs = self.get_triplet_data_idxs(self.test_triplet_data)

        self.graph_edge_index, self.graph_edge_type = self.construct_adj(self.train_triplet_data_idxs)  #No self ring

        print("Create data complete!")
        ###########################################################################################################################

    def encode_entity_to_id(self, data_dir):
        entity2id = {}
        total_entity_num = 0
        with open(data_dir + 'entity2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ent, ent2id = line.strip().split("\t")
                entity2id[ent] = int(ent2id)
        return entity2id

    def encode_relation_to_id(self, data_dir):
        relation2id = {}
        with open(data_dir + 'relation2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                rel, rel2id = line.strip().split("\t")
                relation2id[rel] = int(rel2id)
        return relation2id

    def encode_type_to_id(self, data_dir):
        type2id = {}
        with open(data_dir + 'type2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                t, t2id = line.strip().split("\t")
                type2id[t] = int(t2id)
        return type2id

    def load_type_data(self, data_dir, data_type):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding='utf-8') as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data


    def get_types(self, data):
        types = sorted(list(set([d[1] for d in data])))
        return types

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data])))
        return entities

    def get_type_data_idxs(self, data):
        entity_type_data_idxs = [(self.entity_idxs[data[i][0]], self.entity_types_idxs[data[i][1]]) for i in
                                 range(len(data))]
        return entity_type_data_idxs

    def get_type_data_idxs_dict(self, data):
        entity_type = {}
        for temp in data:
            entity_type.setdefault(temp[0], set()).add(temp[1])
        return entity_type

    def get_type_to_entity(self, data):
        type2entity = {}
        for temp in data:
            type2entity.setdefault(temp[1], set()).add(temp[0])
        return type2entity

    def get_1_1_OR_1_N_test_data(self, data):
        test_data_1_1 = []
        test_data_1_N = []
        for k in data.keys():
            if (len(data[k]) == 1):
                temp = (k, list(data[k])[0])
                test_data_1_1.append(temp)
            else:
                for t in data[k]:
                    temp = (k, t)
                    test_data_1_N.append(temp)
        return test_data_1_1, test_data_1_N

    ######################################################################################################################
    def load_triplet_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                e1, r, e2 = line.strip().split("\t")
                data.append([e1,r,e2])
            return data

    def get_triplet_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def get_triplet_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_triplet_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    def construct_adj(self,data):
        edge_index, edge_type = [], []
        for sub, rel, obj in data:
            edge_index.append((sub, obj))
            edge_type.append(rel)
        return edge_index, edge_type


# d = Data(data_dir='data/FB15k/')