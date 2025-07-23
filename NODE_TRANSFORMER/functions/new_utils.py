import torch
import numpy as np
from torch_geometric.data import Data
def parse_graphs_to_dataset(graphs):
    """将文件变成图，图节点有特征和标签，然后变成Data类型数据"""
    max_num_nodes = 0
    # 找到所有图中节点数量的最大值
    for graph in graphs:
        num_nodes = len(graph.nodes())
        max_num_nodes = max(max_num_nodes, num_nodes)
    dataset = []
    for graph in graphs:
        feats = torch.tensor([graph.nodes[node]["feature"] for node in graph.nodes()], dtype=torch.float)
        edges = [[list(graph.nodes).index(u), list(graph.nodes).index(v)] for u, v in
                 graph.edges]
        weights = torch.tensor([graph.nodes[node]["weight"] for node in graph.nodes()], dtype=torch.float)
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([graph.nodes[node]["label"] for node in graph.nodes()], dtype=torch.long)
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(),max_len=max_num_nodes, y=labels, w=weights ,z=graph,edge=torch.tensor(edges),edge_len=torch.tensor(len(graph.edges)))
        dataset.append(data)
    return dataset

def parse_graphs_to_seq_dataset(graphs):
    """将文件变成图，图节点有特征和标签，然后变成Data类型数据"""
    max_num_nodes = 0
    # 找到所有图中节点数量的最大值
    for graph in graphs:
        num_nodes = len(graph.nodes())
        max_num_nodes = max(max_num_nodes, num_nodes)

    dataset = []
    batch = len(graphs)

    for graph in graphs:
        feats = torch.tensor([graph.nodes[node]["feature"] for node in graph.nodes()], dtype=torch.float)
        edges = [[list(graph.nodes).index(u), list(graph.nodes).index(v)] for u, v in
                 graph.edges]
        weights = torch.tensor([graph.nodes[node]["weight"] for node in graph.nodes()], dtype=torch.float)
        edge_index = torch.tensor(edges).T.long()
        labels = torch.tensor([graph.nodes[node]["label"] for node in graph.nodes()], dtype=torch.long)

        num_nodes = len(graph.nodes())
        input_hidden = feats.size(1)

        # 初始化序列和掩码
        sequences = torch.zeros(1, max_num_nodes, input_hidden, dtype=torch.float)
        masks = torch.zeros(1, max_num_nodes, dtype=torch.float)

        # 根据节点权重进行排序
        sorted_indices = torch.argsort(weights, descending=False)
        sorted_feats = feats[sorted_indices]

        # 填充序列和掩码
        sequences[0, :num_nodes] = sorted_feats
        masks[0, :num_nodes] = 1

        data = Data(x=feats, edge_index=edge_index, y=labels, w=weights, z=graph,
                    edge=torch.tensor(edges), edge_len=torch.tensor(len(graph.edges)),
                    sequences=sequences, masks=masks)
        dataset.append(data)

    return dataset


from scipy import sparse
from scipy.sparse import csgraph

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_graphs_to_dataset_RQGNN(graphs):
    """将文件变成图，图节点有特征和标签，然后变成Data类型数据"""
    dataset = []
    for graph in graphs:
        feats = torch.tensor([graph.nodes[node]["feature"] for node in graph.nodes()], dtype=torch.float)

        # 获取边和边权重
        edges = []
        weights = []
        for u, v, data in graph.edges(data=True):
            edges.append([list(graph.nodes).index(u), list(graph.nodes).index(v)])
            weights.append(data['weight'])

        weights = torch.tensor(weights, dtype=torch.float)

        edges = np.array(edges).T
        # 构建邻接矩阵
        adj = sparse.csr_matrix((weights.numpy(), edges), shape=(len(graph.nodes()), len(graph.nodes())))


        edges = [[list(graph.nodes).index(u), list(graph.nodes).index(v)] for u, v in graph.edges]
        weights = torch.tensor([graph.nodes[node]["weight"] for node in graph.nodes()], dtype=torch.float)
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([graph.nodes[node]["label"] for node in graph.nodes()], dtype=torch.long)



        # 计算图拉普拉斯矩阵
        laplacian = csgraph.laplacian(adj, normed=True)
        lap_list = sparse_mx_to_torch_sparse_tensor(laplacian)

        # 计算 xLx_batch
        temp_L = sparse_mx_to_torch_sparse_tensor(laplacian)
        temp_x = feats
        xLx_batch = torch.diag(torch.mm(torch.mm(temp_x.T, temp_L.to_dense()), temp_x))
        xLx_batch = xLx_batch.view(1, -1)  # 调整形状为 (1, 247)

         # 将 xLx_batch 复制为与图中节点数量相同的维度

        num_nodes = len(graph.nodes())

        xLx_batch = xLx_batch.repeat(num_nodes, 1)  # 调整形状为 (num_nodes, hidden_dim)

        # temp_L = sparse_mx_to_torch_sparse_tensor(laplacian)
        # temp_x = feats
        # xLx_batch = torch.mm(temp_x.T, temp_L.to_dense())
        # xLx_batch = torch.mm(xLx_batch, temp_x)

        # 构建 graphpool_list
        graphpool_list = torch.zeros((1, len(graph.nodes())))
        graphpool_list[0, :] = 1

        # 构建 Data 对象
        data = Data(
            x=feats,
            edge_index=torch.tensor(edge_index).long(),
            y=labels,
            w=weights,
            z=graph,
            edge=torch.tensor(edges),
            edge_len=torch.tensor(len(graph.edges)),
            xLx_batch=xLx_batch
        )
        dataset.append(data)
    return dataset


def parse_graphs_to_dataset_forum(graphs):
    """将文件变成图，图节点有特征和标签，然后变成Data类型数据"""
    dataset = []
    for graph in graphs:
        feats = torch.tensor([graph.nodes[node]["feature"] for node in graph.nodes()], dtype=torch.float)
        layer_feats = torch.tensor([graph.nodes[node]["layer_feature"] for node in graph.nodes()], dtype=torch.float)
        edges = [[list(graph.nodes).index(u), list(graph.nodes).index(v)] for u, v in
                 graph.edges]
        weights = torch.tensor([graph.nodes[node]["weight"] for node in graph.nodes()], dtype=torch.float)
        edge_index = np.transpose(edges).tolist()

        nodes = [node for node in graph.nodes()]
        egde_index_new = torch.tensor(edge_index)

        if (egde_index_new.size(0) != 2):
            continue

        labels = torch.tensor([graph.nodes[node]["label"] for node in graph.nodes()], dtype=torch.long)
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, layer_feats=layer_feats, w=weights ,z=graph,n=nodes)
        dataset.append(data)
    return dataset