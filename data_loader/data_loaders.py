import os
import torch
import collections
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import torch.utils.data as Data
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate

from base import BaseDataLoader

class DataLoader(BaseDataLoader):
    def __init__(self, 
                 data_dir, 
                 batch_size, 
                 score='synergy 0', # name of field for synergy score and threshold for being synergistic
                 n_hop=2, # max number of hops in protein graph
                 n_memory=32, # limits num of element in neighbor set
                 shuffle=True, 
                 validation_split=0.1,
                 test_split=0.2, 
                 num_workers=1):
        self.data_dir = data_dir
        self.score, self.threshold = score.split(' ')
        self.n_hop = n_hop
        self.n_memory = n_memory
        
        # load data
        self.drug_combination_df, self.ppi_df, self.cpi_df, self.dpi_df = self.load_data()
        # get node map
        self.node_map_dict, self.node_num_dict = self.get_node_map_dict()
        # remap the node in the data frame
        self.df_node_remap()
        # drug combinations data remapping
        self.feature_index = self.drug_combination_process()

        # create dataset
        self.dataset = self.create_dataset()
        # create dataloader
        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)
        
        # build the graph
        self.graph = self.build_graph()
        # get target dict
        self.cell_protein_dict, self.drug_protein_dict = self.get_target_dict()
        # some indexs
        self.cells = list(self.cell_protein_dict.keys())
        self.drugs = list(self.drug_protein_dict.keys())
        # get neighbor set
        self.cell_neighbor_set = self.get_neighbor_set(items=self.cells,
                                                       item_target_dict=self.cell_protein_dict)
        self.drug_neighbor_set = self.get_neighbor_set(items=self.drugs,
                                                       item_target_dict=self.drug_protein_dict)
        # save data
        self._save()
        
    def get_cell_neighbor_set(self):
        return self.cell_neighbor_set

    def get_drug_neighbor_set(self):
        return self.drug_neighbor_set

    def get_feature_index(self):
        return self.feature_index

    def get_node_num_dict(self):
        return self.node_num_dict

    def load_data(self):
        drug_combination_df = pd.read_csv(os.path.join(self.data_dir, 'drug_combinations.csv'))
        ppi_df = pd.read_excel(os.path.join(self.data_dir, 'protein-protein_network.xlsx'))
        cpi_df = pd.read_csv(os.path.join(self.data_dir, 'cell_protein.csv'))
        dpi_df = pd.read_csv(os.path.join(self.data_dir, 'drug_protein.csv'))

        return drug_combination_df, ppi_df, cpi_df, dpi_df
    
    def get_node_map_dict(self):
        protein_node = list(set(self.ppi_df['protein_a']) | set(self.ppi_df['protein_b']))
        cell_node = list(set(self.cpi_df['cell']))
        drug_node = list(set(self.dpi_df['drug']))

        node_num_dict = {'protein': len(protein_node), 'cell': len(cell_node), 'drug': len(drug_node)}
        
        mapping = {protein_node[idx]:idx for idx in range(len(protein_node))}
        mapping.update({cell_node[idx]:idx for idx in range(len(cell_node))})
        mapping.update({drug_node[idx]:idx for idx in range(len(drug_node))})

        # display data info
        print('undirected graph')
        print('# proteins: {0}, # drugs: {1}, # cells: {2}'.format(
                len(protein_node), len(drug_node), len(cell_node)))
        print('# protein-protein interactions: {0}, # drug-protein associations: {1}, # cell-protein associations: {2}'.format(
            len(self.ppi_df), len(self.dpi_df), len(self.cpi_df)))

        return mapping, node_num_dict

    def df_node_remap(self):
        self.ppi_df['protein_a'] = self.ppi_df['protein_a'].map(self.node_map_dict)
        self.ppi_df['protein_b'] = self.ppi_df['protein_b'].map(self.node_map_dict)
        self.ppi_df = self.ppi_df[['protein_a', 'protein_b']]

        self.cpi_df['cell'] = self.cpi_df['cell'].map(self.node_map_dict)
        self.cpi_df['protein'] = self.cpi_df['protein'].map(self.node_map_dict)
        self.cpi_df = self.cpi_df[['cell', 'protein']]

        self.dpi_df['drug'] = self.dpi_df['drug'].map(self.node_map_dict)
        self.dpi_df['protein'] = self.dpi_df['protein'].map(self.node_map_dict)
        self.dpi_df = self.dpi_df[['drug', 'protein']]

        self.drug_combination_df['drug1_db'] = self.drug_combination_df['drug1_db'].map(self.node_map_dict)
        self.drug_combination_df['drug2_db'] = self.drug_combination_df['drug2_db'].map(self.node_map_dict)
        self.drug_combination_df['cell'] = self.drug_combination_df['cell'].map(self.node_map_dict)

    def drug_combination_process(self):
        self.drug_combination_df['synergistic'] = [0] * len(self.drug_combination_df)
        self.drug_combination_df.loc[self.drug_combination_df[self.score] > eval(self.threshold), 'synergistic'] = 1
        self.drug_combination_df.to_csv(os.path.join(self.data_dir, 'drug_combination_processed.csv'), index=False)
        
        self.drug_combination_df = self.drug_combination_df[['cell', 'drug1_db', 'drug2_db', 'synergistic']]

        return {'cell': 0, 'drug1': 1, 'drug2': 2}

    def build_graph(self):
        tuples = [tuple(x) for x in self.ppi_df.values]
        graph = nx.Graph()
        graph.add_edges_from(tuples)
        return graph

    def get_target_dict(self):
        cp_dict = collections.defaultdict(list)
        cell_list = list(set(self.cpi_df['cell']))
        for cell in cell_list:
            cell_df = self.cpi_df[self.cpi_df['cell']==cell]
            target = list(set(cell_df['protein']))
            cp_dict[cell] = target
        
        dp_dict = collections.defaultdict(list)
        drug_list = list(set(self.dpi_df['drug']))
        for drug in drug_list:
            drug_df = self.dpi_df[self.dpi_df['drug']==drug]
            target = list(set(drug_df['protein']))
            dp_dict[drug] = target
        
        return cp_dict, dp_dict

    def create_dataset(self):
        # shuffle data
        self.drug_combination_df = self.drug_combination_df.sample(frac=1, random_state=1)
        # shape [n_data, 3]
        feature = torch.from_numpy(self.drug_combination_df.to_numpy())
        # shape [n_data, 1]
        label = torch.from_numpy(self.drug_combination_df[['synergistic']].to_numpy())
        # change tensor type
        feature = feature.type(torch.LongTensor)
        label = label.type(torch.FloatTensor)
        # create dataset
        dataset = Data.TensorDataset(feature, label)
        return dataset

    def get_neighbor_set(self, items, item_target_dict):
        print('constructing neighbor set ...')

        neighbor_set = collections.defaultdict(list)
        for item in items:
            for hop in range(self.n_hop):
                # use the target directly
                if hop == 0:
                    replace = len(item_target_dict[item]) < self.n_memory
                    target_list = list(np.random.choice(item_target_dict[item], size=self.n_memory, replace=replace))
                else:
                    # use the last one to find k+1 hop neighbors
                    origin_nodes = neighbor_set[item][-1]
                    neighbors = []
                    for node in origin_nodes:
                        neighbors += self.graph.neighbors(node)
                    # sample
                    replace = len(neighbors) < self.n_memory
                    target_list = list(np.random.choice(neighbors, size=self.n_memory, replace=replace))
                
                neighbor_set[item].append(target_list)

        return neighbor_set

    def _save(self):
        with open(os.path.join(self.data_dir, 'node_map_dict.pickle'), 'wb') as f:
            pickle.dump(self.node_map_dict, f)
        with open(os.path.join(self.data_dir, 'cell_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.cell_neighbor_set, f)
        with open(os.path.join(self.data_dir, 'drug_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.drug_neighbor_set, f)



class CVDataLoader:
    def __init__(self, 
                 data_dir, 
                 batch_size, 
                 score='synergy 0', # name of field for synergy score and threshold for being synergistic
                 n_hop=2, # max number of hops in protein graph
                 n_memory=32, # limits num of element in neighbor set
                 shuffle=True, 
                 folds=5,
                 validation_split=0.1, 
                 num_workers=1):
        self.data_dir = data_dir
        self.score, self.threshold = score.split(' ')
        self.n_hop = n_hop
        self.n_memory = n_memory
        self.folds = folds
        self.validation_split = validation_split
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            # 'collate_fn': default_collate,
            'num_workers': num_workers
        }

        # Load data frames
        drug_combination_df = pd.read_csv(os.path.join(self.data_dir, 'drug_combinations.csv'))
        ppi_df = pd.read_excel(os.path.join(self.data_dir, 'protein-protein_network.xlsx'))
        cpi_df = pd.read_csv(os.path.join(self.data_dir, 'cell_protein.csv'))
        dpi_df = pd.read_csv(os.path.join(self.data_dir, 'drug_protein.csv'))

        # get node map
        protein_node = list(set(ppi_df['protein_a']) | set(ppi_df['protein_b']))
        cell_node = list(set(cpi_df['cell']))
        drug_node = list(set(dpi_df['drug']))

        self.node_num_dict = {'protein': len(protein_node), 'cell': len(cell_node), 'drug': len(drug_node)}
        
        self.node_map_dict = {protein_node[idx]:idx for idx in range(len(protein_node))}
        self.node_map_dict.update({cell_node[idx]:idx for idx in range(len(cell_node))})
        self.node_map_dict.update({drug_node[idx]:idx for idx in range(len(drug_node))})

        ## display data info
        print('undirected graph')
        print('# proteins: {0}, # drugs: {1}, # cells: {2}'.format(
                len(protein_node), len(drug_node), len(cell_node)))
        print('# protein-protein interactions: {0}, # drug-protein associations: {1}, # cell-protein associations: {2}'.format(
            len(ppi_df), len(dpi_df), len(cpi_df)))

        # remap the node in the data frame
        ppi_df['protein_a'] = ppi_df['protein_a'].map(self.node_map_dict)
        ppi_df['protein_b'] = ppi_df['protein_b'].map(self.node_map_dict)
        ppi_df = ppi_df[['protein_a', 'protein_b']]

        cpi_df['cell'] = cpi_df['cell'].map(self.node_map_dict)
        cpi_df['protein'] = cpi_df['protein'].map(self.node_map_dict)
        cpi_df = cpi_df[['cell', 'protein']]

        dpi_df['drug'] = dpi_df['drug'].map(self.node_map_dict)
        dpi_df['protein'] = dpi_df['protein'].map(self.node_map_dict)
        dpi_df = dpi_df[['drug', 'protein']]

        drug_combination_df['drug1_db'] = drug_combination_df['drug1_db'].map(self.node_map_dict)
        drug_combination_df['drug2_db'] = drug_combination_df['drug2_db'].map(self.node_map_dict)
        drug_combination_df['cell'] = drug_combination_df['cell'].map(self.node_map_dict)

        # drug combinations data remapping
        drug_combination_df['synergistic'] = [0] * len(drug_combination_df)
        drug_combination_df.loc[drug_combination_df[self.score] > eval(self.threshold), 'synergistic'] = 1
        drug_combination_df.to_csv(os.path.join(self.data_dir, 'drug_combination_processed.csv'), index=False)
        
        drug_combination_df = drug_combination_df[['cell', 'drug1_db', 'drug2_db', 'synergistic']]

        self.feature_index = {'cell': 0, 'drug1': 1, 'drug2': 2}

        # create dataset
        ## shuffle data
        drug_combination_df = drug_combination_df.sample(frac=1, random_state=1)
        ## shape [n_data, 3]
        feature = torch.from_numpy(drug_combination_df.to_numpy())
        ## shape [n_data, 1]
        label = torch.from_numpy(drug_combination_df[['synergistic']].to_numpy())
        ## change tensor type
        self.features = feature.type(torch.LongTensor)
        self.labels = label.type(torch.FloatTensor)

        # split data into leakage free groups
        # TODO: Leakage free splitting in groups
        self._groups = np.arange(0, len(label), 1, dtype=int)
        self._group_kfold = GroupKFold(n_splits=folds)
        self._group_kfold.get_n_splits(self.features, self.labels)

        # build the graph
        tuples = [tuple(x) for x in ppi_df.values]
        self.graph = nx.Graph()
        self.graph.add_edges_from(tuples)

        # get target dict
        self.cell_protein_dict = collections.defaultdict(list)
        cell_list = list(set(cpi_df['cell']))
        for cell in cell_list:
            cell_df = cpi_df[cpi_df['cell']==cell]
            target = list(set(cell_df['protein']))
            self.cell_protein_dict[cell] = target
        
        self.drug_protein_dict = collections.defaultdict(list)
        drug_list = list(set(dpi_df['drug']))
        for drug in drug_list:
            drug_df = dpi_df[dpi_df['drug']==drug]
            target = list(set(drug_df['protein']))
            self.drug_protein_dict[drug] = target
        

        # some indexes
        self.cells = list(self.cell_protein_dict.keys())
        self.drugs = list(self.drug_protein_dict.keys())

        self.cell_neighbor_set = self.get_neighbor_set(items=self.cells,
                                                       item_target_dict=self.cell_protein_dict)
        self.drug_neighbor_set = self.get_neighbor_set(items=self.drugs,
                                                       item_target_dict=self.drug_protein_dict)

        # save data
        with open(os.path.join(self.data_dir, 'node_map_dict.pickle'), 'wb') as f:
            pickle.dump(self.node_map_dict, f)
        with open(os.path.join(self.data_dir, 'cell_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.cell_neighbor_set, f)
        with open(os.path.join(self.data_dir, 'drug_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.drug_neighbor_set, f)


    def get_neighbor_set(self, items, item_target_dict):
        print('constructing neighbor set ...')

        neighbor_set = collections.defaultdict(list)
        for item in items:
            for hop in range(self.n_hop):
                # use the target directly
                if hop == 0:
                    replace = len(item_target_dict[item]) < self.n_memory
                    target_list = list(np.random.choice(item_target_dict[item], size=self.n_memory, replace=replace))
                else:
                    # use the last one to find k+1 hop neighbors
                    origin_nodes = neighbor_set[item][-1]
                    neighbors = []
                    for node in origin_nodes:
                        neighbors += self.graph.neighbors(node)
                    # sample
                    replace = len(neighbors) < self.n_memory
                    target_list = list(np.random.choice(neighbors, size=self.n_memory, replace=replace))
                
                neighbor_set[item].append(target_list)

        return neighbor_set

    def __iter__(self):
        self.fold = 0
        self.iterator = iter(self._group_kfold.split(self.features, self.labels, self._groups))
        return self

    def __next__(self):
        # pass
        # for i, (train_index, test_index) in enumerate(self._group_kfold.split(self.features, self.labels, self._groups)):
        #     self.fold = i

        #     # train and validation split
        #     # TODO: leakage free validation split
        #     X_train, X_val, y_train, y_val = train_test_split(self.features[train_index], self.labels[train_index],
        #                                                          test_size=self.validation_split, random_state=42,
        #                                                          shuffle=True, stratify=None)
            
        #     # get test data
        #     X_test, y_test = self.features[test_index], self.features[test_index]
            
        #     # convert to Dataset for DataLoader
        #     training_data = Data.TensorDataset(X_train, y_train)
        #     validation_data = Data.TensorDataset(X_train, y_train)
        #     test_data = Data.TensorDataset(X_train, y_train)

        #     # DataLoader for each set
        #     train_dataloader = torch.utils.data.DataLoader(training_data, **self.init_kwargs)
        #     val_dataloader = torch.utils.data.DataLoader(validation_data, **self.init_kwargs)
        #     test_dataloader = torch.utils.data.DataLoader(test_data, **self.init_kwargs)
            
        #     yield self.fold, train_dataloader, val_dataloader, test_dataloader
            
        # raise StopIteration

        if self.fold < self.folds:
            # train and validation split
            # TODO: leakage free validation split
            train_index, test_index = next(self.iterator)
            X_train, X_val, y_train, y_val = train_test_split(self.features[train_index], self.labels[train_index],
                                                                 test_size=self.validation_split, random_state=42,
                                                                 shuffle=True, stratify=None)
            
            # get test data
            X_test, y_test = self.features[test_index], self.features[test_index]
            
            # convert to Dataset for DataLoader
            training_data = Data.TensorDataset(X_train, y_train)
            validation_data = Data.TensorDataset(X_train, y_train)
            test_data = Data.TensorDataset(X_train, y_train)

            # DataLoader for each set
            train_dataloader = torch.utils.data.DataLoader(training_data, **self.init_kwargs)
            val_dataloader = torch.utils.data.DataLoader(validation_data, **self.init_kwargs)
            test_dataloader = torch.utils.data.DataLoader(test_data, **self.init_kwargs)

            fold = self.fold
            self.fold += 1
            return fold, train_dataloader, val_dataloader, test_dataloader
        else:
            raise StopIteration