import urllib.request as ur
import zipfile
import os
import os.path as osp
import errno
from tqdm import tqdm
import pandas as pd
import numpy as np

GBFACTOR = float(1 << 30)

def decide_download(url):
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"])/GBFACTOR

    ### confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N)\n" % (size)).lower() == "y"
    else:
        return True

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    makedirs(folder)
    data = ur.urlopen(url)

    size = int(data.info()["Content-Length"])

    chunk_size = 1024*1024
    num_iter = int(size/chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                f.write(chunk)
    except:
        if os.path.exists(path):
             os.remove(path)
        raise RuntimeError('Stopped downloading due to interruption.')


    return path

def maybe_log(path, log=True):
    if log:
        print('Extracting', path)

def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)



### reading raw files from a directory.
### for homogeneous graph
def read_csv_graph_raw(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = []):
    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of graphs.
    Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
    edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

    additional_node_files and additional_edge_files must be in the raw directory.
    - The name should be {additional_node_file, additional_edge_file}.csv.gz
    - The length should be num_nodes or num_edges

    additional_node_files must start from 'node_'
    additional_edge_files must start from 'edge_'

    
    '''

    print('Loading necessary files...')
    print('This might take a while.')
    # loading necessary files
    try:
        edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
        num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list

    except FileNotFoundError:
        raise RuntimeError('No necessary file')

    try:
        node_feat = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), compression='gzip', header = None).values
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
    except FileNotFoundError:
        node_feat = None

    try:
        edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), compression='gzip', header = None).values
        if 'int' in str(edge_feat.dtype):
            edge_feat = edge_feat.astype(np.int64)
        else:
            #float
            edge_feat = edge_feat.astype(np.float32)

    except FileNotFoundError:
        edge_feat = None


    additional_node_info = {}   
    for additional_file in additional_node_files:
        assert(additional_file[:5] == 'node_')

        # hack for ogbn-proteins
        if additional_file == 'node_species' and osp.exists(osp.join(raw_dir, 'species.csv.gz')):
            os.rename(osp.join(raw_dir, 'species.csv.gz'), osp.join(raw_dir, 'node_species.csv.gz'))

        temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header = None).values

        if 'int' in str(temp.dtype):
            additional_node_info[additional_file] = temp.astype(np.int64)
        else:
            # float
            additional_node_info[additional_file] = temp.astype(np.float32)

    additional_edge_info = {}   
    for additional_file in additional_edge_files:
        assert(additional_file[:5] == 'edge_')
        temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header = None).values

        if 'int' in str(temp.dtype):
            additional_edge_info[additional_file] = temp.astype(np.int64)
        else:
            # float
            additional_edge_info[additional_file] = temp.astype(np.float32)


    graph_list = []
    num_node_accum = 0
    num_edge_accum = 0

    print('Processing graphs...')
    for num_node, num_edge in tqdm(zip(num_node_list, num_edge_list), total=len(num_node_list)):

        graph = dict()

        ### handling edge
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum+num_edge], 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]

            graph['edge_index'] = duplicated_edge

            if edge_feat is not None:
                graph['edge_feat'] = np.repeat(edge_feat[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)
            else:
                graph['edge_feat'] = None

            for key, value in additional_edge_info.items():
                graph[key] = np.repeat(value[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)

        else:
            graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum+num_edge]

            if edge_feat is not None:
                graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
            else:
                graph['edge_feat'] = None

            for key, value in additional_edge_info.items():
                graph[key] = value[num_edge_accum:num_edge_accum+num_edge]

        num_edge_accum += num_edge

        ### handling node
        if node_feat is not None:
            graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
        else:
            graph['node_feat'] = None

        for key, value in additional_node_info.items():
            graph[key] = value[num_node_accum:num_node_accum+num_node]


        graph['num_nodes'] = num_node
        num_node_accum += num_node

        graph_list.append(graph)

    return graph_list


### reading raw files from a directory.
### npz ver
### for homogeneous graph
def read_binary_graph_raw(raw_dir, add_inverse_edge = False):
    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of graphs.
    Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
    edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

    raw_dir must contain data.npz
    - edge_index
    - num_nodes_list
    - num_edges_list
    - node_** (optional, node_feat is the default node features)
    - edge_** (optional, edge_feat is the default edge features)
    '''

    if add_inverse_edge:
        raise RuntimeError('add_inverse_edge is depreciated in read_binary')

    print('Loading necessary files...')
    print('This might take a while.')
    data_dict = np.load(osp.join(raw_dir, 'data.npz'))

    edge_index = data_dict['edge_index']
    num_nodes_list = data_dict['num_nodes_list']
    num_edges_list = data_dict['num_edges_list']

    # storing node and edge features
    node_dict = {}
    edge_dict = {}

    for key in list(data_dict.keys()):
        if key == 'edge_index' or key == 'num_nodes_list' or key == 'num_edges_list':
            continue

        if key[:5] == 'node_':
            node_dict[key] = data_dict[key]
        elif key[:5] == 'edge_':
            edge_dict[key] = data_dict[key]
        else:
            raise RuntimeError(f"Keys in graph object should start from either \'node_\' or \'edge_\', but found \'{key}\'.")

    graph_list = []
    num_nodes_accum = 0
    num_edges_accum = 0

    print('Processing graphs...')
    for num_nodes, num_edges in tqdm(zip(num_nodes_list, num_edges_list), total=len(num_nodes_list)):

        graph = dict()

        graph['edge_index'] = edge_index[:, num_edges_accum:num_edges_accum+num_edges]

        for key, feat in edge_dict.items():
            graph[key] = feat[num_edges_accum:num_edges_accum+num_edges]

        if 'edge_feat' not in graph:
            graph['edge_feat'] =  None

        for key, feat in node_dict.items():
            graph[key] = feat[num_nodes_accum:num_nodes_accum+num_nodes]

        if 'node_feat' not in graph:
            graph['node_feat'] = None

        graph['num_nodes'] = num_nodes

        num_edges_accum += num_edges
        num_nodes_accum += num_nodes

        graph_list.append(graph)

    return graph_list


def read_npz_dict(path):
    tmp = np.load(path)
    dict = {}
    for key in tmp.keys():
        dict[key] = tmp[key]
    del tmp
    return dict

def read_node_label_hetero(raw_dir):
    df = pd.read_csv(osp.join(raw_dir, 'nodetype-has-label.csv.gz'))
    label_dict = {}
    for nodetype in df.keys():
        has_label = df[nodetype].values[0]
        if has_label:
            label_dict[nodetype] = pd.read_csv(osp.join(raw_dir, 'node-label', nodetype, 'node-label.csv.gz'), compression='gzip', header = None).values

    if len(label_dict) == 0:
        raise RuntimeError('No node label file found.')

    return label_dict


def read_nodesplitidx_split_hetero(split_dir):
    df = pd.read_csv(osp.join(split_dir, 'nodetype-has-split.csv.gz'))
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    for nodetype in df.keys():
        has_label = df[nodetype].values[0]
        if has_label:
            train_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
            valid_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
            test_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

    if len(train_dict) == 0:
        raise RuntimeError('No split file found.')

    return train_dict, valid_dict, test_dict
