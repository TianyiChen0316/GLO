import os
import sys
import numpy as np
import math
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from .core.sql_featurizer import database

# Stack
TABLES = {
    'stack': [
        "badge",
        "comment",
        "account",
        "post_link",
        "so_user",
        "question",
        "tag",
        "site",
        "tag_question",
        "answer",
    ],
    'tpcds': [
        'catalog_page',
        'promotion',
        'store_returns',
        'catalog_returns',
        'catalog_sales',
        'customer_address',
        'customer',
        'customer_demographics',
        'date_dim',
        'dbgen_version',
        'household_demographics',
        'income_band',
        'inventory',
        'item',
        'reason',
        'ship_mode',
        'store',
        'store_sales',
        'time_dim',
        'warehouse',
        'web_page',
        'web_returns',
        'web_sales',
        'web_site',
        'call_center',
    ],
    'job': [
        "aka_name",
        "cast_info",
        "complete_cast",
        "char_name",
        "comp_cast_type",
        "aka_title",
        "company_type",
        "company_name",
        "keyword",
        "movie_companies",
        "kind_type",
        "link_type",
        "info_type",
        "movie_info_idx",
        "movie_info",
        "title",
        "role_type",
        "name",
        "movie_keyword",
        "movie_link",
        "person_info",
    ],
}

def table_column_distribution_by_count(table_name, column_name):
    sql = f"""select {column_name}, count(*) as _c from {table_name} group by {column_name} order by _c"""
    results = database.execute(sql, cache=True)
    database._cache_backup()
    return tuple(results)

_dist_points = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
def _column_dist_features(dist):
    total = len(dist)
    _, value = zip(*dist)
    value = np.array(value)
    total_value = value.sum()
    res = []
    for point in _dist_points:
        index = min(total - 1, max(0, round(total * point)))
        index_value = value[:index + 1].sum()
        res.append(index_value / total_value)
    return res

def column_features(table_name, column_name):
    # 2 + 13 + 7
    dist = table_column_distribution_by_count(table_name, column_name)
    property_feature = database.schema.column_features(table_name, column_name, dtype=None)
    dist_feature = _column_dist_features(dist)

    table_obj = database.schema.name_to_table[table_name]
    is_pk = column_name in table_obj.primary_keys
    is_fk = False
    for column_names, pk_table_name, pk_column_names in table_obj.foreign_keys:
        if column_name in column_names:
            is_fk = True
    fk_feature = [1 if is_pk else 0, 1 if is_fk else 0]
    return (*fk_feature, *property_feature, *dist_feature)

def table_features(table_name):
    # 5
    table_obj = database.schema.name_to_table[table_name]
    table_out_degree = len(table_obj.foreign_keys)
    table_in_degree = len(table_obj.foreign_keys_rev)
    table_pk_num = len(table_obj.primary_keys)
    table_row_count = math.log10(max(table_obj.row_count, 1))
    table_col_count = table_obj.row_count
    res = (table_out_degree, table_in_degree, table_pk_num, table_row_count, table_col_count)
    return res

def dataset_gen():
    res = {}
    for table_obj in tqdm(database.schema.name_to_table.values(), total=database.schema.size, desc='Generating dataset'):
        table_name = table_obj.name
        _table_features = table_features(table_name)
        for column_name in table_obj.columns.values():
            _column_features = column_features(table_name, column_name)
            res[table_name] = (table_name, column_name, _table_features, _column_features)
    return res

def _data_to_tensor(value):
    _table_features, _column_features = value[2], value[3]
    table_row_count = _table_features[3]
    return torch.Tensor([*_column_features, *_table_features, math.log(table_row_count + 1), 1 if table_row_count < 4 else 0])

def dataset_preprocess(dataset, batch_size=16):
    tensors = list(map(_data_to_tensor, dataset))
    tensors = torch.stack(tensors, dim=0)
    dataset = TensorDataset(tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    return dataloader

def cluster_wss(k_means: KMeans, tensors: np.ndarray, cluster_labels = None):
    centroids = k_means.cluster_centers_
    if cluster_labels is None:
        cluster_labels = k_means.predict(tensors)
    tensor_centroids = centroids[cluster_labels]
    wss = ((tensors - tensor_centroids) ** 2).mean(axis=0).sum()
    return wss

from sklearn.metrics import silhouette_score

def visualize_table_cluster_metrics(features : dict, test_tables):
    train_tables = set(features.keys()).difference(test_tables)

    train_features = [(k, features[k]) for k in train_tables]
    train_table_names, train_tensors = zip(*train_features)
    train_tensors = torch.stack(train_tensors, dim=0).detach().numpy()

    ks = list(range(2, 11))
    wss = []
    sil = []
    for _k in ks:
        _cluster_model = KMeans(n_clusters=_k)
        _train_cluster = _cluster_model.fit_predict(train_tensors)
        _wss = cluster_wss(_cluster_model, train_tensors, _train_cluster)
        _sil = silhouette_score(train_tensors, _train_cluster, metric='euclidean')
        print(f'K: {_k}, WSS: {_wss:.04f}, Sil: {_sil:.04f}')
        wss.append(_wss)
        sil.append(_sil)

    f = plt.figure()
    ax = plt.axes()
    plt.ylabel('WSS')
    plt.xlabel('K')
    plt.plot(ks, wss, linewidth=1)
    plt.show()

    f = plt.figure()
    ax = plt.axes()
    plt.ylabel('Silhouette')
    plt.xlabel('K')
    plt.plot(ks, sil, linewidth=1)
    plt.show()


def assign_table_cluster(features : dict, test_tables, k):
    train_tables = set(features.keys()).difference(test_tables)

    train_features = [(k, features[k]) for k in train_tables]
    train_table_names, train_tensors = zip(*train_features)
    train_tensors = torch.stack(train_tensors, dim=0).detach().numpy()

    cluster_model = KMeans(n_clusters=k)
    train_cluster : np.ndarray = cluster_model.fit_predict(train_tensors)
    train_cluster = train_cluster.tolist()

    test_features = [(k, features[k]) for k in test_tables]
    test_table_names, test_tensors = zip(*test_features)
    test_tensors = torch.stack(test_tensors, dim=0).detach().numpy()

    test_cluster = cluster_model.predict(test_tensors)
    test_cluster = test_cluster.tolist()

    res = {**{k : v for k, v in zip(train_table_names, train_cluster)}, **{k : v for k, v in zip(test_table_names, test_cluster)}}
    return cluster_model, res

def visualize_3d(table_embeddings):
    names, tensors = zip(*table_embeddings.items())
    tensors_numpy = torch.stack(tensors, dim=0).detach().numpy()
    pca = PCA(n_components=3)
    coords = pca.fit_transform(tensors_numpy)
    f = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    sc = ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2])
    plt.show()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    import argparse

    from lib.torch.seed import seed

    class AutoEncoder(torch.nn.Module):
        def __init__(self, input_size, output_size, hidden_size=64):
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(input_size)
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size, output_size),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(output_size, hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size, input_size),
            )

        def forward(self, input):
            norm_out = self.norm(input)
            encoded = self.encoder(norm_out)
            decoded = self.decoder(encoded)
            return encoded, decoded, norm_out

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='database')
    parser.add_argument('--user', type=str, default='postgres')
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--password', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('--test-dataset', type=str, choices=('job', 'stack', 'tpcds'), default='tpcds')
    parser.add_argument('--model-checkpoint', type=str, default='data_driven/model')
    parser.add_argument('--feature-checkpoint', type=str, default='data_driven/table')
    parser.add_argument('--dataset-cache', type=str, default='data_driven/dataset')
    parser.add_argument('-k', '--clusters', type=int, default=6)

    args = parser.parse_args()

    database.setup(dbname=args.database, user=args.user, port=args.port, password=args.password)

    test_tables = TABLES[args.test_dataset]

    def train():
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': table_model.parameters()},
        ], lr=args.learning_rate, weight_decay=1e-4)
        postfix = {'loss': '/'}

        for epoch in range(args.epochs):
            gen = tqdm(dataloader, desc=f'Epoch {epoch:03d}', postfix=postfix)

            model.train()
            table_model.train()
            for x, *_ in gen:
                encoded, decoded, norm_out = model(x)
                loss = torch.nn.functional.mse_loss(decoded, norm_out)

                table_encoded, table_decoded, table_norm_out = table_model(x[..., -7:])
                table_loss = torch.nn.functional.mse_loss(table_decoded, table_norm_out)

                all_encoded = torch.cat([encoded, table_encoded], dim=-1)
                reg_loss = torch.nn.functional.l1_loss(all_encoded,
                                                       torch.zeros_like(all_encoded, device=encoded.device))

                loss = loss * 0.45 + table_loss * 0.45 + reg_loss * 0.1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                postfix.update(
                    loss=loss.item()
                )
                gen.set_postfix(postfix)
        torch.save({
            'model': model.state_dict(),
            'table_model': table_model.state_dict(),
        }, f'{args.model_checkpoint}_{args.test_dataset}.pkl')

    def assign_table_embeddings():
        model.eval()
        table_model.eval()

        features = {}
        dataset = train_set + test_set
        table_dataset = {}
        table_embs = {}

        _visualize = {}
        for data in dataset:
            table_name, column_name, table_feature, column_feature = data
            _table_dataset = table_dataset.setdefault(table_name, [])
            data_tensor = _data_to_tensor(data).unsqueeze(0)
            res, _, _ = model(data_tensor)
            res = res.squeeze(0)
            _table_dataset.append((column_name, res))
            _visualize[f'{table_name}.{column_name}'] = res
            if not table_name in table_embs:
                res, _, _ = table_model(data_tensor[..., -7:])
                res = res.squeeze(0)
                table_embs[table_name] = res

        for table_name, _table_dataset in tqdm(table_dataset.items(), total=len(table_dataset),
                                               desc='Calculating table embeddings'):
            _, tensors = zip(*_table_dataset)
            embs = torch.stack(tensors, dim=0)
            embs_pooling = embs.max(dim=0).values  # torch.cat([embs.mean(dim=0), embs.max(dim=0).values], dim=-1)
            table_emb = table_embs[table_name]
            features[table_name] = torch.cat([embs_pooling, table_emb], dim=-1)
        return features

    _input_size = 29
    model = AutoEncoder(_input_size, 6, hidden_size=64)
    _table_input_size = 7
    table_model = AutoEncoder(_table_input_size, 3, hidden_size=16)

    dataset_cache = f'{args.dataset_cache}.pkl'
    if os.path.isfile(args.dataset_cache):
        with open(dataset_cache, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = dataset_gen()
        with open(dataset_cache, 'wb') as f:
            pickle.dump(dataset, f)

    train_set, test_set = [], []
    for table, data in dataset.items():
        if table in test_tables:
            test_set.append(data)
        else:
            train_set.append(data)

    seed(0)

    dataloader = dataset_preprocess(train_set, batch_size=args.batch_size)

    model_checkpoint = f'{args.model_checkpoint}_{args.test_dataset}.pkl'
    if os.path.isfile(model_checkpoint):
        state_dict = torch.load(model_checkpoint)
        model.load_state_dict(state_dict['model'])
        table_model.load_state_dict(state_dict['table_model'])
    else:
        train()

    table_embeddings = assign_table_embeddings()

    visualize_table_cluster_metrics(table_embeddings, test_tables)
    cluster_model, cluster_res = assign_table_cluster(table_embeddings, test_tables, args.clusters)

    cluster_res_rev = {}
    for k, v in cluster_res.items():
        _dict = cluster_res_rev.setdefault(v, [])
        _dict.append(k)
    print('Clusters:', file=sys.stderr)
    for cluster_index, tables in cluster_res_rev.items():
        print(f'- {cluster_index}', file=sys.stderr)
        for table in sorted(set(tables).difference(test_tables)):
            table_obj = database.schema.name_to_table[table]
            print(f'    {table} : {table_obj.row_count}, {len(table_obj.foreign_keys)}, {len(table_obj.foreign_keys_rev)}', file=sys.stderr)

    print('Test table clusters:', file=sys.stderr)
    for cluster_index, tables in cluster_res_rev.items():
        print(f'- {cluster_index}', file=sys.stderr)
        for table in sorted(set(tables).intersection(test_tables)):
            table_obj = database.schema.name_to_table[table]
            print(f'    {table} : {table_obj.row_count}, {len(table_obj.foreign_keys)}, {len(table_obj.foreign_keys_rev)}', file=sys.stderr)

    torch.save({
        'model': model.state_dict(),
        'embeddings': table_embeddings,
        'cluster_model': cluster_model.get_params(deep=True),
        'cluster': cluster_res,
    }, f'{args.feature_checkpoint}_{args.test_dataset}.pkl')
