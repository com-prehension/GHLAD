import os
import torch
from NODE_TRANSFORMER.functions.构建树 import  build_tree_from_txt,construct_tree_to_nx,\
     process_trees_to_construct_tree_chains,construct_tree_to_nx_with_filemap, \
    construct_tree_to_nx_with_filemap_and_high_subgraph\
    ,construct_layered_graph,construct_tree_to_nx_with_filemap_position,construct_tree_to_nx_with_filemap_layerfeatures, \
    construct_tree_to_nx_with_filemap_layerfeatures_new,construct_tree_to_nx_with_filemap_layerfeatures_new_edges_new

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
from NODE_TRANSFORMER.functions.构件图_链条 import deal_tree_to_chain

from functions.precision_index import show_metrics_AUPRC_new
import random

# 设置随机数种子以确保结果可重复
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#将文件转换为tree
def process_files_to_construct_trees(directory,event_map):
    tree_dataset = []
    for root1, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root1, file_name)
                root = build_tree_from_txt(file_path)
                g = construct_tree_to_nx(root, event_map)
                g.remove_node("root")
                tree_dataset.append((root,g,file_path))
    return tree_dataset

def process_files_to_construct_trees_chains(directory,event_map):
    tree_dataset = []
    all_chains=[]
    for root1, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root1, file_name)
                root = build_tree_from_txt(file_path)
                g = construct_tree_to_nx(root, event_map)
                g.remove_node("root")
                tree_dataset.append((root,g,file_path))

    return tree_dataset

def node2vec_to_tensor(trees, node2vec_model):
    file_name_map = {}
    for _,tree in trees:

        for node in tree.nodes():
            feat = []
            try:
                file_embedding = list(node2vec_model.wv[node.split(':')[0]])
            except KeyError:
                file_embedding = [0.0] * 256
            feat.extend(file_embedding)
            file_name_map[node]=feat
    return file_name_map

# 处理数据集
def parse_graph_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []
    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap(root, event_map, file_name_map,exception_map)
        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)
        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]
        edge_index = np.transpose(edges).tolist()

        egde_index_new = torch.tensor(edge_index)

        if (egde_index_new.size(0) != 2):
            continue


        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)

        bro_data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels ,z=tree)
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset

def parse_graph_layer_feats_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []
    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap_layerfeatures(root, event_map, file_name_map,exception_map)
        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        layer_feats = torch.tensor([tree.nodes[node]["layer_feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)
        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]

        edge_index = np.transpose(edges).tolist()

        egde_index_new = torch.tensor(edge_index)

        if (egde_index_new.size(0) != 2):
            continue
        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, layer_feats=layer_feats, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)

        bro_data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels ,z=tree)

        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))

    return dataset

# 处理数据集
def parse_graph_layered_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []
    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap(root, event_map, file_name_map,exception_map)
        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)
        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)

        bro_graph = construct_layered_graph(tree)
        # bro_graph.remove_node("root")
        bro_edges = [[list(bro_graph.nodes).index(u), list(bro_graph.nodes).index(v)] for u, v in
                     bro_graph.edges]
        bro_edge_index = np.transpose(bro_edges).tolist()
        bro_data = Data(x=feats,edge_index=torch.tensor(bro_edge_index).long())
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset

def construct_layered_graph_new(G, root_label='root'):
    # 缓存文件名
    graph_hash = get_graph_hash(G)

# 处理数据集
def parse_graph_layered_adaptive_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []
    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap_position(root, event_map, file_name_map,exception_map)
        bro_graph = construct_layered_graph_new(tree)

        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)
        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)

        bro_graph.remove_node("root")
        bro_edges = [[list(bro_graph.nodes).index(u), list(bro_graph.nodes).index(v)] for u, v in
                     bro_graph.edges]

        bro_edge_index = np.transpose(bro_edges).tolist()
        bro_data = Data(x=feats,edge_index=torch.tensor(bro_edge_index).long())
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset

def parse_graph_layered_adaptive_random_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []
    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap_position(root, event_map, file_name_map,exception_map)
        bro_graph = construct_random_layered_graph(tree)

        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)
        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]

        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)

        bro_graph.remove_node("root")
        bro_edges = [[list(bro_graph.nodes).index(u), list(bro_graph.nodes).index(v)] for u, v in
                     bro_graph.edges]

        bro_edge_index = np.transpose(bro_edges).tolist()
        bro_data = Data(x=feats,edge_index=torch.tensor(bro_edge_index).long())
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset


def parse_graph_layered_adaptive_layerfeatures_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []

    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap_layerfeatures(root, event_map, file_name_map,exception_map)
        # tree = construct_tree_to_nx_with_filemap_layerfeatures_new(root, event_map, file_name_map,exception_map)
        # bro_graph = construct_random_layered_graph(tree)

        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        layer_feats = torch.tensor([tree.nodes[node]["layer_feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)

        # print(weights)

        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]



        edge_index = np.transpose(edges).tolist()
        egde_index_new = torch.tensor(edge_index)

        if(egde_index_new.size(0)!=2):
            continue
            # print(path)


        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])

        chains = deal_tree_to_chain(root, tree)
        chains = torch.tensor(chains)
        chains_number = chains.size(0)

        # data = Data(x=feats, edge_index=torch.tensor(edge_index).long(),layer_feats=layer_feats, y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), layer_feats=layer_feats, y=labels,
                    c_n=chains_number, w=weights, z=tree, c=chains, t=root, f=path, s=sort, n=nodes)

        # data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), layer_feats=layer_feats, y=labels
        #         , w=weights, z=tree, t=root, f=path, s=sort, n=nodes)

        bro_data = Data(x=feats)
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset
def parse_graph_layered_new_graphs_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []

    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap_layerfeatures_new_edges_new(root, event_map, file_name_map,exception_map)

        # bro_graph = construct_random_layered_graph(tree)

        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        layer_feats = torch.tensor([tree.nodes[node]["layer_feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)

        # print(weights)

        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]



        edge_index = np.transpose(edges).tolist()
        egde_index_new = torch.tensor(edge_index)

        if(egde_index_new.size(0)!=2):
            continue
            # print(path)


        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])

        chains = deal_tree_to_chain(root, tree)
        chains = torch.tensor(chains)
        chains_number = chains.size(0)

        # data = Data(x=feats, edge_index=torch.tensor(edge_index).long(),layer_feats=layer_feats, y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), layer_feats=layer_feats, y=labels,
                    c_n=chains_number, w=weights, z=tree, c=chains, t=root, f=path, s=sort, n=nodes)

        # data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), layer_feats=layer_feats, y=labels
        #         , w=weights, z=tree, t=root, f=path, s=sort, n=nodes)

        bro_data = Data(x=feats)
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset

def parse_graph_layerfeatures_dataset_new(trees, event_map, file_name_map,exception_map):
    dataset = []

    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap_layerfeatures_new(root, event_map, file_name_map,exception_map)
        # bro_graph = construct_random_layered_graph(tree)

        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        layer_feats = torch.tensor([tree.nodes[node]["layer_feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)

        # print(weights)

        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]



        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])

        chains = deal_tree_to_chain(root, tree)
        chains = torch.tensor(chains)
        chains_number = chains.size(0)

        # data = Data(x=feats, edge_index=torch.tensor(edge_index).long(),layer_feats=layer_feats, y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), layer_feats=layer_feats, y=labels,
                    c_n=chains_number, w=weights, z=tree, c=chains, t=root, f=path, s=sort, n=nodes)

        # data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), layer_feats=layer_feats, y=labels
        #         , w=weights, z=tree, t=root, f=path, s=sort, n=nodes)

        bro_data = Data(x=feats)
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset
#
# def parse_graph_HimNet_dataset(trees, event_map, file_name_map,exception_map):
#     dataset = []
#
#
#     graph_labels = []
#     for root, graph, path in trees:
#         node_labels = []
#         for node in graph.nodes:
#             if node != "root":
#                 node_labels.append(graph.nodes[node]["label"])
#         if 1 not in node_labels:
#             graph_labels.append(0)
#         else:
#             graph_labels.append(1)
#
#     num_unique_node_labels=2
#
#     for root,_,path in trees:
#         trace_id = root.trace_id
#         sort = sort_label(path)
#         # node feature: event embedding + 文件背景信息 + cost
#         tree = construct_tree_to_nx_with_HimNet(root, event_map, file_name_map,exception_map)
#
#         tree.remove_node("root")
#
#         labels = [tree.nodes[node]["label"] for node in tree.nodes()]
#         graph_label=0
#         if 1 in labels:
#             graph_label=1
#
#         new_graph = convert_tree_to_graph_format(tree, graph_label, num_unique_node_labels)
#
#         dataset.append(new_graph)
#
#     return dataset

# 处理数据集
def parse_graph_layered_position_dataset(trees, event_map, file_name_map,exception_map):
    dataset = []
    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree = construct_tree_to_nx_with_filemap_position(root, event_map, file_name_map,exception_map)

        tree.remove_node("root")

        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        weights = torch.tensor([tree.nodes[node]["weight"] for node in tree.nodes()], dtype=torch.float)
        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, w=weights ,z=tree ,t=root ,f=path,s=sort,n=nodes)

        bro_graph = construct_layered_graph(tree)

        # bro_graph.remove_node("root")
        bro_edges = [[list(bro_graph.nodes).index(u), list(bro_graph.nodes).index(v)] for u, v in
                     bro_graph.edges]

        bro_edge_index = np.transpose(bro_edges).tolist()
        bro_data = Data(x=feats,edge_index=torch.tensor(bro_edge_index).long())
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset


# 处理数据集
def parse_graph_dataset_high_subgraph(trees, event_map, file_name_map,exception_map):
    dataset = []
    for root,_,path in trees:
        trace_id = root.trace_id
        sort = sort_label(path)
        # node feature: event embedding + 文件背景信息 + cost
        tree,high_graph = construct_tree_to_nx_with_filemap_and_high_subgraph(root, event_map, file_name_map,exception_map)
        tree.remove_node("root")
        feats = torch.tensor([tree.nodes[node]["feature"] for node in tree.nodes()], dtype=torch.float)
        nodes = [node for node in tree.nodes()]
        edges = [[list(tree.nodes).index(u), list(tree.nodes).index(v)] for u, v in tree.edges]
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([tree.nodes[node]["label"] for node in tree.nodes()])
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels ,z=tree ,t=root ,f=path,s=sort,n=nodes)

        high_graph.remove_node("root")
        bro_edges = [[list(high_graph.nodes).index(u), list(high_graph.nodes).index(v)] for u, v in
                     high_graph.edges]
        bro_edge_index = np.transpose(bro_edges).tolist()
        bro_data = Data(edge_index=torch.tensor(bro_edge_index).long())
        # add tree and only brother edge graph
        dataset.append((data, bro_data, [trace_id] * len(tree.nodes()), list(tree.nodes())))
    print(len(dataset))
    # graph_dataset = torch_geometric.data.Batch.from_data_list(graph_data_list)
    return dataset

# 将label进行分类
def sort_label(path):
    error_list = ['perfect','middle','noperfect']
    ## 为0代表noexception
    sort=0
    new_sort="noexception"
    with open(path,"r") as f:
        lines = f.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("exception="):
                new_sort = line.strip().split("=")[1]
    for key in error_list:
        if key == new_sort:
            sort=error_list.index(key)+1
            break
    return torch.tensor(sort,dtype=torch.long).unsqueeze(0)

# 将字典类型的转换为二维集合再转换成二维向量
def file_name_map_to_nodes(filename_map,trees):
    file_map=[]
    for graph in trees:
        for node in graph.nodes():
            file_map.append(filename_map[node].tolist())
    return file_map

def event_map_to_nodes(primeval_map,trees):
    event_map=[]
    for graph in trees:
        for node in graph.nodes():
            event_map.append(primeval_map[graph.nodes[node]['event']].tolist())
    return event_map

# 按比例划分数据集
def random_split_data(dataset, train_ratio):
    train_num = int(len(dataset) * train_ratio)
    train_set = dataset[0: train_num]
    test_set = dataset[train_num:]
    return train_set, test_set

# 只添加正确图或链条
def get_normal_chains_in_train_trees(train_trees, event_map):
    all_chains = process_trees_to_construct_tree_chains(train_trees, event_map)
    normal_chains = []
    for chains in all_chains:
        for chain in chains:
            if 1 not in chain.labels:
                normal_chains.append([name.split(":")[0] for name in chain.names])
    return normal_chains

def get_normal_train_graphs(graphs):
    normal = []
    for _, graph in graphs:
        labels = [graph.nodes[node]["label"] for node in graph.nodes()]
        if 1 not in labels:  # 只添加正确的图
            normal.append(graph)
    return normal

def deal_labels(data):
    new_data=[]
    for tree,graph,path in data:
        labels=[graph.nodes[node]["label"] for node in graph.nodes()]

        if 1 not in labels:
            new_data.append((tree,graph,path))
    return new_data

# 读取event_one_hot
def read_one_hot(event_embedding_path):
    reflection={}
    with open(event_embedding_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            event_one_hot = [float(x) for x in line_new.split(",") if x]
            if line_old in reflection:
                continue
            else:
                reflection[line_old] = event_one_hot
    f.close()
    return reflection

def read_file_one_hot(file_embedding_path):
    file_name_map = {}
    with open(file_embedding_path,"r") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            file_one_hot = [float(x) for x in line_new.split(",") if x] + [float(0) for i in range(25)]
            if line_old in file_name_map:
                continue
            else:
                file_name_map[line_old]=file_one_hot
    file.close()
    return file_name_map

def read_exception_one_hot(exception_embedding_path):
    file_name_map = {}
    with open(exception_embedding_path,"r") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            file_one_hot = [float(x) for x in line_new.split(",") if x] + [float(0) for i in range(67)]
            if line_old in file_name_map:
                continue
            else:
                file_name_map[line_old]=file_one_hot
    file.close()
    return file_name_map


def read_file_one_hot_new(file_embedding_path,feature_size):
    file_name_map = {}
    with open(file_embedding_path,"r") as file:
        lines = file.readlines()
        padding_size = feature_size - len(lines)
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")


            file_one_hot = [float(x) for x in line_new.split(",") if x] + [float(0) for i in range(padding_size)]
            if line_old in file_name_map:
                continue
            else:
                file_name_map[line_old]=file_one_hot
    file.close()
    return file_name_map

def read_exception_one_hot_new(exception_embedding_path,feature_size):
    file_name_map = {}
    with open(exception_embedding_path,"r") as file:
        lines = file.readlines()
        padding_size = feature_size - len(lines)
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            file_one_hot = [float(x) for x in line_new.split(",") if x] + [float(0) for i in range(padding_size)]
            if line_old in file_name_map:
                continue
            else:
                file_name_map[line_old]=file_one_hot
    file.close()
    return file_name_map

# 记录每张图的长度和节点
def deal_Graph(graphs):
    graphs_nodes = []
    graph_len = []
    for graph in graphs:
        graph_nodes = {}
        graph_len.append(len(graph.nodes))
        for index, node in enumerate(graph.nodes):
            node = node.split(":")[0]
            if node not in graph_nodes:
                graph_nodes[node] = [index]
            else:
                graph_nodes[node] += [index]
        graphs_nodes.append(graph_nodes)
    return graph_len,graphs_nodes


# 记录每张图的长度和节点(DGL图)
def deal_DGL_Graph(graph):
    graphs_nodes = {}
    for index, node in enumerate(graph.ndata):
        node = node.split(":")[0]
        if node not in graphs_nodes:
            graphs_nodes[node] = [index]
        else:
            graphs_nodes[node] += [index]

    return graphs_nodes

def deal_Graph_nodes(graphs):
    graphs_nodes = []
    graphs_all_nodes = []
    graph_len = []
    for graph in graphs:
        graph_nodes = []
        graph_len.append(len(graph.nodes))
        for index, node in enumerate(graph.nodes):
            graphs_all_nodes.append(node)
            # node = node.split(":")[0]
            graph_nodes.append(node)
        graphs_nodes.append(graph_nodes)
    return graph_len,graphs_nodes,graphs_all_nodes
# 从数据集中获取树
def get_trees(data):
    trees=[]
    for tree,graph,_ in data:

        trees.append(tree)

    return trees

# 获取数据中图的类型标签
def add_label(file_path_list):
    error_list = ['call_change','argument_change','chain_change','condition_change']

    error_label=""
    error_labels=[]
    for path in file_path_list:
        for key in error_list:
            if key in path:
                error_label=key.replace("_"," ")
                error_labels.append(error_label)
                break
    return error_labels


def write_result_to_txt(storage_path,train_traceid_list,test_traceid_list):
    file_path = storage_path
    with open(file_path, 'w') as file:
        file.write("训练集:" + '\n')
        file.write("==========================" + '\n')
        for item in train_traceid_list:
            file.write(str(item[0]) + '\n')

        file.write("测试集:" + '\n')
        file.write("==========================" + '\n')
        for item in test_traceid_list:
            file.write(str(item[0]) + '\n')


        file.close()

class CallChain:
    def __init__(self, attributes1, labels1, node_names1, timestamps1):
        self.attributes = [x for x in attributes1]
        self.attributes.reverse()
        self.costs = []
        self.events = []
        self.exceptions = []
        for attr in self.attributes:
            _, cost, event, exception = attr.split(",")
            cost_str = cost.split("=")[1].replace("ms", "")
            self.costs.append(float(cost_str))
            # print("debug",event.split("=")[1])
            exc = exception.split("=")[1]
            if exc == "null":
                self.exceptions.append(float(0))
            else:
                self.exceptions.append(float(exc))

        self.labels = [x for x in labels1]
        self.labels.reverse()
        self.names = [x for x in node_names1]
        self.names.reverse()
        self.timestamps = [x for x in timestamps1]
        self.timestamps.reverse()


def process_trees_to_construct_tree_chains(trees):
    all_chains = []
    for tree1 in trees:
        attributes = []
        labels = []
        node_names = []
        timestamps = []
        chains = []
        traverse_tree(tree1, attributes, labels, node_names, timestamps, chains)
        all_chains.append(chains)

    return all_chains

def traverse_tree(node, attributes, labels, node_names, timestamps, chains):
    if not node.children:
        chains.append(CallChain(attributes, labels, node_names, timestamps))
    for child, weight, props, label in node.children:
        attributes.append(props)
        labels.append(label)
        node_names.append(child.name)
        timestamps.append(float(weight))
        traverse_tree(child, attributes, labels, node_names, timestamps, chains)
        labels.pop()
        node_names.pop()
        attributes.pop()
        timestamps.pop()



def validation_onehot(model, t_val_dataloader, g_val_dataloader, val_result_list, storage_path, rewrite,device):
    model.eval()

    criterion = torch.nn.BCELoss()
    total_loss = 0.0
    val_true_labels = []
    val_predicted_prob = []
    with torch.no_grad():
        for tree, graph in zip(t_val_dataloader, g_val_dataloader):
            tree = tree.to(device)
            graph = graph.to(device)

            output = model(tree.x, tree.edge_index, graph.edge_index)
            loss = criterion(output.squeeze().float(), tree.y.float())
            total_loss += loss.item()
            # total_node_num += batch.num_nodes

            # y_pred = torch.argmax(output, dim=1)

            pred_prob = output.squeeze().cpu().numpy()
            # 存储真实标签和预测标签
            val_true_labels.extend(tree.y.cpu().numpy())
            val_predicted_prob.extend(pred_prob)

    val_predicted_labels = (np.array(val_predicted_prob) >= 0.5).astype(int)
    show_metrics_AUPRC_new(val_true_labels, val_predicted_labels, val_predicted_prob, storage_path, rewrite)
    # 计算 F1 分数、召回率、精确度、auc
    avg_loss = total_loss / len(t_val_dataloader)
    print("validation avg loss:", avg_loss)
    val_result_list += ["validation avg loss:" + str(avg_loss)]

def validation_node_gru(model,t_val_dataloader,g_val_dataloader,val_result_list,storage_path, rewrite,device):
    model.eval()

    criterion = torch.nn.BCELoss()
    total_loss = 0.0
    val_true_labels = []
    val_predicted_prob = []
    with torch.no_grad():
        for tree, graph in zip(t_val_dataloader, g_val_dataloader):

            tree = tree.to(device)
            graph = graph.to(device)


            # output = model(tree.x , tree.edge_index, graph.edge_index)
            output = model(tree)
            loss = criterion(output.squeeze().float(), tree.y.float())
            total_loss += loss.item()
            # total_node_num += batch.num_nodes

            # y_pred = torch.argmax(output, dim=1)

            pred_prob = output.squeeze().cpu().numpy()
            # 存储真实标签和预测标签
            val_true_labels.extend(tree.y.cpu().numpy())
            val_predicted_prob.extend(pred_prob)

    val_predicted_labels = (np.array(val_predicted_prob) >= 0.5).astype(int)
    show_metrics_AUPRC_new(val_true_labels, val_predicted_labels, val_predicted_prob, storage_path, rewrite)
    # 计算 F1 分数、召回率、精确度、auc
    avg_loss = total_loss / len(t_val_dataloader)
    print("validation avg loss:", avg_loss)
    val_result_list += ["validation avg loss:" + str(avg_loss)]

def write_txt(file_path,call_all_mean,argument_all_mean,chain_all_mean,condition_all_mean,train_traceid_list,
              val_traceid_list,test_one_traceid_list,test_traceid_list,incorrect_trace_ids_call_change,
              incorrect_trace_ids_argument_change,incorrect_trace_ids_chain_change,incorrect_trace_ids_condition_change):
    with open(file_path, 'a') as file:
        file.write("精度:" + '\n')
        file.write("==========================" + '\n')
        file.write(str(call_all_mean) + '\n')
        file.write(str(argument_all_mean) + '\n')
        file.write(str(chain_all_mean) + '\n')
        file.write(str(condition_all_mean) + '\n')

        file.write("训练集:" + '\n')
        file.write("==========================" + '\n')
        for item in train_traceid_list:
            file.write(str(item[0]) + '\n')

        file.write("验证集:" + '\n')
        file.write("==========================" + '\n')
        for item in val_traceid_list:
            file.write(str(item[0]) + '\n')

        file.write("测试集一种一个:" + '\n')
        file.write("==========================" + '\n')
        for item in test_one_traceid_list:
            file.write(str(item[0]) + '\n')

        file.write("测试集:" + '\n')
        file.write("==========================" + '\n')
        for item in test_traceid_list:
            file.write(str(item[0]) + '\n')

        file.write("错误图:" + '\n')
        file.write("==========================" + '\n')
        for item in incorrect_trace_ids_call_change:
            file.write(str(item) + '\n')
        for item in incorrect_trace_ids_argument_change:
            file.write(str(item) + '\n')
        for item in incorrect_trace_ids_chain_change:
            file.write(str(item) + '\n')
        for item in incorrect_trace_ids_condition_change:
            file.write(str(item) + '\n')

        file.close()


def load_dataloader(train_tree_set, val_tree_set, test_data_list, test_one_list):
    # event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot.txt"
    # file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot.txt"
    # exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list.txt"

    feature_size = 166
    # feature_size=82

    event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot2.txt"
    file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot1.txt"
    exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list1.txt"

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    # tree_data_list = process_files_to_construct_trees(data_path, event_map)
    # random.shuffle(tree_data_list)
    # train_tree_set,test_data_list = random_split_data(tree_data_list, train_ratio=0.4)

    # 下面是使用固定分层
    combined_train_data_list = parse_graph_dataset(train_tree_set, event_map, file_name_map,exception_map)  # load graph data

    # np.random.shuffle(combined_train_data_list)

    combined_val_data_list = parse_graph_dataset(val_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_test_one_data_list = parse_graph_dataset(test_one_list, event_map, file_name_map,exception_map)  # load graph data

    combined_test_data_list = parse_graph_dataset(test_data_list, event_map, file_name_map,exception_map)

    t_train_data_list, g_train_data_list, train_traceid_list, train_node_name_list = zip(*combined_train_data_list)
    t_val_data_list, g_val_data_list, val_traceid_list, val_node_name_list = zip(*combined_val_data_list)
    t_test_one_data_list, g_test_one_data_list, test_one_traceid_list, test_one_node_name_list = zip(
        *combined_test_one_data_list)
    t_test_data_list, g_test_data_list, test_traceid_list, test_node_name_list = zip(*combined_test_data_list)
    t_train_dataset = torch_geometric.data.Batch.from_data_list(list(t_train_data_list))
    g_train_dataset = torch_geometric.data.Batch.from_data_list(list(g_train_data_list))
    t_val_dataset = torch_geometric.data.Batch.from_data_list(list(t_val_data_list))
    g_val_dataset = torch_geometric.data.Batch.from_data_list(list(g_val_data_list))
    t_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_one_data_list))
    g_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_one_data_list))

    t_test_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_data_list))
    g_test_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_data_list))

    # 创建数据加载器，用于批处理图数据
    # batch_size = 32
    batch_size = 128
    t_train_dataloader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_train_dataloader = DataLoader(g_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    t_val_dataloader = DataLoader(t_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_val_dataloader = DataLoader(g_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    t_test_one_dataloader = DataLoader(t_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_test_one_dataloader = DataLoader(g_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    t_test_dataloader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_test_dataloader = DataLoader(g_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    return t_train_dataloader,g_train_dataloader,t_val_dataloader,g_val_dataloader,t_test_one_dataloader,g_test_one_dataloader,t_test_dataloader,g_test_dataloader


def load_dataloader_layer_feats(train_tree_set, val_tree_set, test_data_list, test_one_list):
    # event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot.txt"
    # file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot.txt"
    # exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list.txt"

    feature_size = 166
    # feature_size=82

    event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot2.txt"
    file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot1.txt"
    exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list1.txt"

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    # tree_data_list = process_files_to_construct_trees(data_path, event_map)
    # random.shuffle(tree_data_list)
    # train_tree_set,test_data_list = random_split_data(tree_data_list, train_ratio=0.4)

    # 下面是使用固定分层
    combined_train_data_list = parse_graph_layer_feats_dataset(train_tree_set, event_map, file_name_map,exception_map)  # load graph data

    # np.random.shuffle(combined_train_data_list)

    combined_val_data_list = parse_graph_layer_feats_dataset(val_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_test_one_data_list = parse_graph_layer_feats_dataset(test_one_list, event_map, file_name_map,exception_map)  # load graph data

    combined_test_data_list = parse_graph_layer_feats_dataset(test_data_list, event_map, file_name_map,exception_map)

    t_train_data_list, g_train_data_list, train_traceid_list, train_node_name_list = zip(*combined_train_data_list)
    t_val_data_list, g_val_data_list, val_traceid_list, val_node_name_list = zip(*combined_val_data_list)
    t_test_one_data_list, g_test_one_data_list, test_one_traceid_list, test_one_node_name_list = zip(
        *combined_test_one_data_list)
    t_test_data_list, g_test_data_list, test_traceid_list, test_node_name_list = zip(*combined_test_data_list)
    t_train_dataset = torch_geometric.data.Batch.from_data_list(list(t_train_data_list))
    g_train_dataset = torch_geometric.data.Batch.from_data_list(list(g_train_data_list))
    t_val_dataset = torch_geometric.data.Batch.from_data_list(list(t_val_data_list))
    g_val_dataset = torch_geometric.data.Batch.from_data_list(list(g_val_data_list))
    t_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_one_data_list))
    g_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_one_data_list))

    t_test_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_data_list))
    g_test_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_data_list))

    # 创建数据加载器，用于批处理图数据
    batch_size = 32
    t_train_dataloader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_train_dataloader = DataLoader(g_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    t_val_dataloader = DataLoader(t_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_val_dataloader = DataLoader(g_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    t_test_one_dataloader = DataLoader(t_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_test_one_dataloader = DataLoader(g_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    t_test_dataloader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    g_test_dataloader = DataLoader(g_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    return t_train_dataloader,g_train_dataloader,t_val_dataloader,g_val_dataloader,t_test_one_dataloader,g_test_one_dataloader,t_test_dataloader,g_test_dataloader



def load_dataloader_layerfeatures(train_tree_set, val_tree_set, test_data_list, test_one_list):

    event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot.txt"
    file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot.txt"
    exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list.txt"

    # feature_size=166
    feature_size=82

    # event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot2.txt"
    # file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot1.txt"
    # exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list1.txt"

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path,feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path,feature_size)



    # 下面是使用自适应分层
    combined_train_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(train_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_val_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(val_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_test_one_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(test_one_list, event_map, file_name_map,exception_map)  # load graph data

    combined_test_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(test_data_list, event_map, file_name_map,exception_map)

    t_train_data_list, g_train_data_list, train_traceid_list, train_node_name_list = zip(*combined_train_data_list)
    t_val_data_list, g_val_data_list, val_traceid_list, val_node_name_list = zip(*combined_val_data_list)
    t_test_one_data_list, g_test_one_data_list, test_one_traceid_list, test_one_node_name_list = zip(
        *combined_test_one_data_list)
    t_test_data_list, g_test_data_list, test_traceid_list, test_node_name_list = zip(*combined_test_data_list)
    t_train_dataset = torch_geometric.data.Batch.from_data_list(list(t_train_data_list))
    g_train_dataset = torch_geometric.data.Batch.from_data_list(list(g_train_data_list))
    t_val_dataset = torch_geometric.data.Batch.from_data_list(list(t_val_data_list))
    g_val_dataset = torch_geometric.data.Batch.from_data_list(list(g_val_data_list))
    t_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_one_data_list))
    g_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_one_data_list))

    t_test_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_data_list))
    g_test_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_data_list))

    # 创建数据加载器，用于批处理图数据
    # batch_size = 128
    batch_size = 32
    # num_workers=os.cpu_count() // 2
    num_workers=0
    t_train_dataloader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_train_dataloader = DataLoader(g_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_val_dataloader = DataLoader(t_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_val_dataloader = DataLoader(g_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_test_one_dataloader = DataLoader(t_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_test_one_dataloader = DataLoader(g_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    t_test_dataloader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_test_dataloader = DataLoader(g_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return t_train_dataloader,g_train_dataloader,t_val_dataloader,g_val_dataloader,t_test_one_dataloader,g_test_one_dataloader,t_test_dataloader,g_test_dataloader


def load_dataloader_layerfeatures_from_dataname(train_tree_set, val_tree_set, test_data_list, test_one_list, dataname):
    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname=="forum":
        feature_size = 82
    elif dataname=="novel":
        feature_size = 166
    elif dataname=="halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    # 下面是使用自适应分层
    combined_train_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(train_tree_set, event_map,
                                                                                  file_name_map,
                                                                                  exception_map)  # load graph data

    combined_val_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(val_tree_set, event_map, file_name_map,
                                                                                exception_map)  # load graph data

    combined_test_one_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(test_one_list, event_map,
                                                                                     file_name_map,
                                                                                     exception_map)  # load graph data

    combined_test_data_list = parse_graph_layered_adaptive_layerfeatures_dataset(test_data_list, event_map,
                                                                                 file_name_map, exception_map)

    t_train_data_list, g_train_data_list, train_traceid_list, train_node_name_list = zip(*combined_train_data_list)
    t_val_data_list, g_val_data_list, val_traceid_list, val_node_name_list = zip(*combined_val_data_list)
    t_test_one_data_list, g_test_one_data_list, test_one_traceid_list, test_one_node_name_list = zip(
        *combined_test_one_data_list)
    t_test_data_list, g_test_data_list, test_traceid_list, test_node_name_list = zip(*combined_test_data_list)
    t_train_dataset = torch_geometric.data.Batch.from_data_list(list(t_train_data_list))
    g_train_dataset = torch_geometric.data.Batch.from_data_list(list(g_train_data_list))
    t_val_dataset = torch_geometric.data.Batch.from_data_list(list(t_val_data_list))
    g_val_dataset = torch_geometric.data.Batch.from_data_list(list(g_val_data_list))
    t_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_one_data_list))
    g_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_one_data_list))

    t_test_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_data_list))
    g_test_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_data_list))

    # 创建数据加载器，用于批处理图数据
    # batch_size = 128
    batch_size = 32
    # num_workers=os.cpu_count() // 2
    num_workers = 0
    t_train_dataloader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_train_dataloader = DataLoader(g_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_val_dataloader = DataLoader(t_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_val_dataloader = DataLoader(g_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_test_one_dataloader = DataLoader(t_test_one_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)
    g_test_one_dataloader = DataLoader(g_test_one_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)

    t_test_dataloader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_test_dataloader = DataLoader(g_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return t_train_dataloader, g_train_dataloader, t_val_dataloader, g_val_dataloader, t_test_one_dataloader, g_test_one_dataloader, t_test_dataloader, g_test_dataloader


def load_dataloader_layerfeatures_new_graphs(train_tree_set, val_tree_set, test_data_list, test_one_list):
    # event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot.txt"
    # file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot.txt"
    # exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list.txt"

    event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot2.txt"
    file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot1.txt"
    exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list1.txt"

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot(file_embedding_path)
    exception_map = read_exception_one_hot(exception_embedding_path)



    # 下面是使用自适应分层
    combined_train_data_list = parse_graph_layered_new_graphs_dataset(train_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_val_data_list = parse_graph_layered_new_graphs_dataset(val_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_test_one_data_list = parse_graph_layered_new_graphs_dataset(test_one_list, event_map, file_name_map,exception_map)  # load graph data

    combined_test_data_list = parse_graph_layered_new_graphs_dataset(test_data_list, event_map, file_name_map,exception_map)

    t_train_data_list, g_train_data_list, train_traceid_list, train_node_name_list = zip(*combined_train_data_list)
    t_val_data_list, g_val_data_list, val_traceid_list, val_node_name_list = zip(*combined_val_data_list)
    t_test_one_data_list, g_test_one_data_list, test_one_traceid_list, test_one_node_name_list = zip(
        *combined_test_one_data_list)
    t_test_data_list, g_test_data_list, test_traceid_list, test_node_name_list = zip(*combined_test_data_list)
    t_train_dataset = torch_geometric.data.Batch.from_data_list(list(t_train_data_list))
    g_train_dataset = torch_geometric.data.Batch.from_data_list(list(g_train_data_list))
    t_val_dataset = torch_geometric.data.Batch.from_data_list(list(t_val_data_list))
    g_val_dataset = torch_geometric.data.Batch.from_data_list(list(g_val_data_list))
    t_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_one_data_list))
    g_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_one_data_list))

    t_test_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_data_list))
    g_test_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_data_list))

    # 创建数据加载器，用于批处理图数据
    batch_size = 32
    # num_workers=os.cpu_count() // 2
    num_workers=0
    t_train_dataloader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_train_dataloader = DataLoader(g_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_val_dataloader = DataLoader(t_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_val_dataloader = DataLoader(g_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_test_one_dataloader = DataLoader(t_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_test_one_dataloader = DataLoader(g_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    t_test_dataloader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_test_dataloader = DataLoader(g_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return t_train_dataloader,g_train_dataloader,t_val_dataloader,g_val_dataloader,t_test_one_dataloader,g_test_one_dataloader,t_test_dataloader,g_test_dataloader


def load_dataloader_layerfeatures_new(train_tree_set, val_tree_set, test_data_list, test_one_list):
    event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot.txt"
    file_embedding_path = r"/root/autodl-tmp/project/data/file_name_one_hot.txt"
    exception_embedding_path = r"/root/autodl-tmp/project/data/exception_list.txt"

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot(file_embedding_path)
    exception_map = read_exception_one_hot(exception_embedding_path)



    # 下面是使用自适应分层
    combined_train_data_list = parse_graph_layerfeatures_dataset_new(train_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_val_data_list = parse_graph_layerfeatures_dataset_new(val_tree_set, event_map, file_name_map,exception_map)  # load graph data

    combined_test_one_data_list = parse_graph_layerfeatures_dataset_new(test_one_list, event_map, file_name_map,exception_map)  # load graph data

    combined_test_data_list = parse_graph_layerfeatures_dataset_new(test_data_list, event_map, file_name_map,exception_map)

    t_train_data_list, g_train_data_list, train_traceid_list, train_node_name_list = zip(*combined_train_data_list)
    t_val_data_list, g_val_data_list, val_traceid_list, val_node_name_list = zip(*combined_val_data_list)
    t_test_one_data_list, g_test_one_data_list, test_one_traceid_list, test_one_node_name_list = zip(
        *combined_test_one_data_list)
    t_test_data_list, g_test_data_list, test_traceid_list, test_node_name_list = zip(*combined_test_data_list)
    t_train_dataset = torch_geometric.data.Batch.from_data_list(list(t_train_data_list))
    g_train_dataset = torch_geometric.data.Batch.from_data_list(list(g_train_data_list))
    t_val_dataset = torch_geometric.data.Batch.from_data_list(list(t_val_data_list))
    g_val_dataset = torch_geometric.data.Batch.from_data_list(list(g_val_data_list))
    t_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_one_data_list))
    g_test_one_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_one_data_list))

    t_test_dataset = torch_geometric.data.Batch.from_data_list(list(t_test_data_list))
    g_test_dataset = torch_geometric.data.Batch.from_data_list(list(g_test_data_list))

    # 创建数据加载器，用于批处理图数据
    batch_size = 32
    # num_workers=os.cpu_count() // 2
    num_workers=0
    t_train_dataloader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_train_dataloader = DataLoader(g_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_val_dataloader = DataLoader(t_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_val_dataloader = DataLoader(g_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    t_test_one_dataloader = DataLoader(t_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_test_one_dataloader = DataLoader(g_test_one_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    t_test_dataloader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g_test_dataloader = DataLoader(g_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return t_train_dataloader,g_train_dataloader,t_val_dataloader,g_val_dataloader,t_test_one_dataloader,g_test_one_dataloader,t_test_dataloader,g_test_dataloader




def classify_nodes_from_multiple_graphs(graphs, root_label='root'):
    top_level_count = defaultdict(int)
    middle_level_count = defaultdict(int)
    bottom_level_count = defaultdict(int)

    for G in graphs:
        for node in G.nodes():
            if node == root_label:
                continue
            if G.in_degree(node) == 1 and G.has_edge(node, root_label) or G.in_degree(node) == 0 and G.has_edge(node,root_label):
                top_level_count[node] += 1
            elif G.out_degree(node) > 0 and G.in_degree(node) > 0:
                middle_level_count[node] += 1
            else:
                bottom_level_count[node] += 1

    return top_level_count, middle_level_count, bottom_level_count


from collections import defaultdict, Counter
def classify_nodes_from_multiple_graphs_new(graphs, root_label='root'):
    top_level_count = defaultdict(int)
    middle_level_count = defaultdict(int)
    bottom_level_count = defaultdict(int)
    node_layer_count = defaultdict(Counter)

    for G in graphs:
        for node in G.nodes():
            if node == root_label:
                continue
            if (G.out_degree(node) == 1 and G.has_edge(node,root_label)) or (G.in_degree(node) == 0 and G.has_edge(node,root_label)):
                top_level_count[node] += 1
                node_layer_count[node]['top'] += 1
            elif G.out_degree(node) == 1 and G.in_degree(node) > 0:
                middle_level_count[node] += 1
                node_layer_count[node]['middle'] += 1
            elif G.out_degree(node) == 1 and G.in_degree(node) == 0:
                bottom_level_count[node] += 1
                node_layer_count[node]['bottom'] += 1

    # print(top_level_count)
    # print(middle_level_count)
    # print(bottom_level_count)

    # 根据占比保留最多的层次
    final_top_level = set()
    final_middle_level = set()
    final_bottom_level = set()

    for node, counts in node_layer_count.items():
        most_common_layer = counts.most_common(1)[0][0]
        if most_common_layer == 'top':
            final_top_level.add(node)
        elif most_common_layer == 'middle':
            final_middle_level.add(node)
        else:
            final_bottom_level.add(node)

    return final_top_level, final_middle_level, final_bottom_level






def classify_nodes_in_current_graph(G, root_label='root',
                                    top_level_count=None,
                                    middle_level_count=None,
                                    bottom_level_count=None,
                                    weight=0.5):
    top_level_nodes = set()
    middle_level_nodes = set()
    bottom_level_nodes = set()

    for node in G.nodes():
        if node == root_label:
            continue

        top_score = top_level_count.get(node, 0)
        middle_score = middle_level_count.get(node, 0)
        bottom_score = bottom_level_count.get(node, 0)

        if G.in_degree(node) == 1 and G.has_edge(node,root_label) or G.in_degree(node) == 0 and G.has_edge(node,root_label):
            top_score += weight
        elif G.out_degree(node) > 0 and G.in_degree(node)>0:
            middle_score += weight
        else:
            bottom_score += weight

        if top_score >= middle_score and top_score >= bottom_score:
            top_level_nodes.add(node)
        elif middle_score >= top_score and middle_score >= bottom_score:
            middle_level_nodes.add(node)
        else:
            bottom_level_nodes.add(node)

    return top_level_nodes, middle_level_nodes, bottom_level_nodes

def classify_nodes_from_one_graph(G, root_label='root'):
    top_level_count = defaultdict(int)
    middle_level_count = defaultdict(int)
    bottom_level_count = defaultdict(int)
    node_layer_count = defaultdict(Counter)


    for node in G.nodes():
        if node == root_label:
            continue
        if (G.out_degree(node) == 1 and G.has_edge(node, root_label)) or (G.in_degree(node) == 0 and G.has_edge(node, root_label)):

            top_level_count[node] += 1
            node_layer_count[node]['top'] += 1
        elif G.out_degree(node) > 0 and G.in_degree(node) > 0:

            middle_level_count[node] += 1
            node_layer_count[node]['middle'] += 1
        elif G.in_degree(node) == 0 and G.out_degree(node) > 0:

            bottom_level_count[node] += 1
            node_layer_count[node]['bottom'] += 1


    # print("top level:")
    # print(top_level_count)
    # print("middle level:")
    # print(middle_level_count)
    # print("bottom level:")
    # print(bottom_level_count)



    sub_levels=further_classify_middle_level(middle_level_count,G)
    # print("sub_level")
    # print(sub_levels)

    return top_level_count, sub_levels, bottom_level_count

from sklearn.cluster import DBSCAN
import numpy as np
def further_classify_middle_level(middle_level_nodes, G):
    if len(middle_level_nodes) == 0:
        return {}
    elif len(middle_level_nodes) == 1:
        return {0: set(middle_level_nodes)}

    node_features = defaultdict(list)

    for node in G.nodes():
        if node in middle_level_nodes:
            # 出度
            out_degree = G.out_degree(node)
            # 入度
            in_degree = G.in_degree(node)
            # pagerank,衡量节点的重要性和影响力，基于链接分析的算法.
            pagerank = nx.pagerank(G)[node]
            # 聚类系数,节点邻居节点之间的连接密度.反映节点局部连接紧密度.
            clustering_coeff = nx.clustering(G, node)
            # 中介中心性,节点在所有最短路径中作为中介的次数.衡量节点在图中作为桥梁的作用.
            betweenness = nx.betweenness_centrality(G)[node]  # 使用节点的中介中心性作为特征
            node_features[node].extend([out_degree, in_degree, pagerank, clustering_coeff, betweenness])

    if not node_features:
        return {}

    max_length = max(len(features) for features in node_features.values())
    node_features_list = []
    node_labels_list = []

    for node, features in node_features.items():
        while len(features) < max_length:
            features.append(0)
        node_features_list.append(features)
        node_labels_list.append(node)

    node_features_array = np.array(node_features_list)
    dbscan = DBSCAN(eps=0.5, min_samples=2).fit(node_features_array)
    clusters = dbscan.labels_

    unique_clusters = set(clusters)
    sub_levels = {i: set() for i in unique_clusters}

    for i, node in enumerate(node_labels_list):
        sub_levels[clusters[i]].add(node)

    return sub_levels
import hashlib
import pickle

def get_graph_hash(G):
    graph_data = nx.to_numpy_array(G).tobytes()
    nodes_data = ''.join([str(n) + str(G.nodes[n]) for n in G.nodes()]).encode()
    edges_data = ''.join([str(e) + str(G.edges[e]) for e in G.edges()]).encode()
    combined_data = graph_data + nodes_data + edges_data
    return hashlib.sha256(combined_data).hexdigest()


import networkx as nx



# 随机分层的方法
def random_classify_nodes(G):
    nodes = list(G.nodes())
    random.shuffle(nodes)

    num_nodes = len(nodes)
    num_top = num_nodes // 3
    num_middle = num_nodes // 3
    num_bottom = num_nodes - num_top - num_middle

    top_level_nodes = set(nodes[:num_top])
    middle_level_nodes = set(nodes[num_top:num_top + num_middle])
    bottom_level_nodes = set(nodes[num_top + num_middle:])

    return top_level_nodes, middle_level_nodes, bottom_level_nodes


def construct_random_layered_graph(G, root_label='root'):
    # # 缓存文件名
    # graph_hash = get_graph_hash(G)
    # cache_file = f"D:/lizhenhua/结果集/15.自定义新数据集5_node_gru_result/图随机缓存/{graph_hash}_random_layered_graph.pkl"
    #
    # try:
    #     # 尝试从缓存中加载
    #     with open(cache_file, 'rb') as f:
    #         new_G = pickle.load(f)
    #     # print(f"Loaded from cache: {cache_file}")
    #     return new_G
    # except FileNotFoundError:
    #     print(f"Cache not found, processing: {cache_file}")

    # 随机分层
    top_level_nodes, middle_level_nodes, bottom_level_nodes = random_classify_nodes(G)
    nodes = list(G.nodes)

    def build_layer_adjacency_matrix(layer_nodes):
        num_nodes = len(layer_nodes)
        layer_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                layer_edges.append([layer_nodes[i], layer_nodes[j]])
                layer_edges.append([layer_nodes[j], layer_nodes[i]])
        return layer_edges

    new_G = nx.DiGraph()

    new_G.add_nodes_from(nodes)

    # 构建顶层、中间层和底层的边
    top_layer_edges = build_layer_adjacency_matrix(list(top_level_nodes))
    new_G.add_edges_from(top_layer_edges)

    for sub_level in [middle_level_nodes]:
        middle_layer_edges = build_layer_adjacency_matrix(list(sub_level))
        new_G.add_edges_from(middle_layer_edges)

    bottom_layer_edges = build_layer_adjacency_matrix(list(bottom_level_nodes))
    new_G.add_edges_from(bottom_layer_edges)

    # # 将结果保存到缓存
    # with open(cache_file, 'wb') as f:
    #     pickle.dump(new_G, f)
    # print(f"Saved to cache: {cache_file}")

    return new_G
