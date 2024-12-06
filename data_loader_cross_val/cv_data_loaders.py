import os
import torch
import collections
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import torch.utils.data as Data

import random

from data_loader_cross_val.cv_base_data_loader import CrossValidationBaseDataLoader


class CrossValidationDataLoader(CrossValidationBaseDataLoader):
    def __init__(self,
                 data_dir,
                 batch_size,
                 score='synergy 0',
                 n_hop=2,
                 n_memory=32,
                 num_folds=5,
                 shuffle=True,
                 num_workers=1,
                 cleanup_neighbors = True):
        self.data_dir = data_dir
        self.score, self.threshold = score.split(' ')
        self.n_hop = n_hop
        self.n_memory = n_memory
        self.cleanup_neighbors = cleanup_neighbors
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

        # THIS IS THE STEP WHERE YOU NEED TO CREATE THE FOLDS
        # creates a list of lists containing dataframe row indices for k-folds cross validation
        self.fold_indices = self.create_fold_indices(num_folds)
        # create dataloader
        super().__init__(self.dataset, batch_size, self.fold_indices, shuffle, num_workers)

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

    def get_fold_indices(self):
        return self.fold_indices

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

        mapping = {protein_node[idx]: idx for idx in range(len(protein_node))}
        mapping.update({cell_node[idx]: idx for idx in range(len(cell_node))})
        mapping.update({drug_node[idx]: idx for idx in range(len(drug_node))})

        # display data info
        print('undirected graph')
        print('# proteins: {0}, # drugs: {1}, # cells: {2}'.format(
            len(protein_node), len(drug_node), len(cell_node)))
        print(
            '# protein-protein interactions: {0}, # drug-protein associations: {1}, # cell-protein associations: {2}'.format(
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

    '''
    GRAPH MANIPULATION
    '''

    def build_randomized_graph(self, drop_ratio):
        tuples = [tuple(x) for x in self.ppi_df.values]
        remove_edges = random.sample(tuples, int(len(tuples) * drop_ratio))
        if drop_ratio == 0:
            assert remove_edges == [] or None, "The number of edges to be removed is not 0 despite drop_ratio being 0"
        graph = nx.Graph()
        graph.add_edges_from(tuples)
        graph.remove_edges_from(remove_edges)
        return graph

    def get_target_dict(self):
        cp_dict = collections.defaultdict(list)
        cell_list = list(set(self.cpi_df['cell']))
        for cell in cell_list:
            cell_df = self.cpi_df[self.cpi_df['cell'] == cell]
            target = list(set(cell_df['protein']))
            cp_dict[cell] = target

        dp_dict = collections.defaultdict(list)
        drug_list = list(set(self.dpi_df['drug']))
        for drug in drug_list:
            drug_df = self.dpi_df[self.dpi_df['drug'] == drug]
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

    '''
    Get indices of rows to be added to each fold
    '''

    def create_fold_indices(self, n_folds=5):
        # copy drug combination dataframe to preserve original
        df_copy = self.drug_combination_df.copy()
        # add new column with tuples that are sorted containing each drug combination (ChatGPT)
        df_copy['drug_combination'] = df_copy.apply(lambda row: tuple(sorted([row['drug1_db'], row['drug2_db']])),
                                                    axis=1)
        # library that stores folds as keys and row indices as values
        fold_indices = collections.defaultdict(list)
        # iterate through each unique drug combination
        for idx, combo in enumerate(df_copy['drug_combination'].unique()):
            # identify rows with drug combination
            row_idx = self.drug_combination_df.loc[df_copy['drug_combination'] == combo].index
            # rotate through the folds throughout the loop
            fold = idx % n_folds
            # add rows with drug combination to fold
            fold_indices[fold].extend(row_idx)

        # total number of indices from the folds must be equal to the dataset  size
        assert sum([len(x) for x in fold_indices.values()]) == len(self.drug_combination_df), "Folds do not have the same number of items as the dataset"

        return fold_indices

    def get_neighbor_set(self, items, item_target_dict):
        print('constructing neighbor set ...', end=' ')

        if self.cleanup_neighbors:
            neighbor_set = collections.defaultdict(list)
            for item in items:
                for hop in range(self.n_hop):
                    # use the target directly
                    if hop == 0:
                        replace = len(item_target_dict[item]) < self.n_memory # to fill up missing spots for target_list
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
        else:
            neighbor_set = collections.defaultdict(list)
            for item in items:
                neighbors_old = set()
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
                        neighbors = set(neighbors).difference(neighbors_old)
                        neighbors_old = neighbors_old.union(neighbors)
                        neighbors = list(neighbors)
                        # sample
                        replace = len(neighbors) < self.n_memory
                        target_list = list(np.random.choice(neighbors, size=self.n_memory, replace=replace))

                    neighbor_set[item].append(target_list)

        print('done')
        return neighbor_set

    def _save(self):
        with open(os.path.join(self.data_dir, 'node_map_dict.pickle'), 'wb') as f:
            pickle.dump(self.node_map_dict, f)
        with open(os.path.join(self.data_dir, 'cell_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.cell_neighbor_set, f)
        with open(os.path.join(self.data_dir, 'drug_neighbor_set.pickle'), 'wb') as f:
            pickle.dump(self.drug_neighbor_set, f)
