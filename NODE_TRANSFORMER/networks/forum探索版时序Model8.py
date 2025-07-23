import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv,GCNConv,GATConv
from sklearn.cluster import kmeans_plusplus

""" 分层采用的模型
    只使用两层的TransformerConv，并且不进行batch单独分
"""
class SequentialRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, rnn_type="GRU"):
        """
        时序 RNN 模型 (支持 GRU / LSTM)
        :param input_dim: 输入特征维度
        :param hidden_dim: RNN 隐藏层维度
        :param num_layers: RNN 层数
        :param rnn_type: 使用 "GRU" 或 "LSTM"
        """
        super(SequentialRNN, self).__init__()

        # 输入特征映射到 RNN 隐藏维度
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        # self.positional_encoding = TimeBasedPositionalEncoding(hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        # 选择 RNN 类型
        if rnn_type == "GRU":
            self.rnn1 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
            self.seq_tn1 = nn.LayerNorm(hidden_dim)
            # self.rnn2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
            # self.seq_tn2 = nn.LayerNorm(hidden_dim)

        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        else:
            raise ValueError("Unsupported RNN type: Choose 'GRU' or 'LSTM'")

    def forward(self, x, mask):
        """
        前向传播
        :param x: 输入时序数据，形状为 (batch_size, seq_len, input_dim)
        :param mask: 填充掩码，形状为 (batch_size, seq_len)，填充值为 True
        :return: RNN 输出特征，形状为 (batch_size, seq_len, hidden_dim)
        """
        # 输入投影到 RNN 隐藏维度
        x = self.input_projection(x)

        # 位置编码
        x = self.positional_encoding(x)
        # x = self.positional_encoding(x, node_weights)  # 添加时序位置编码

        # 运行 RNN
        packed_output, _ = self.rnn1(x)
        packed_output=self.seq_tn1(packed_output)
        packed_output=F.relu(packed_output)

        return packed_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class HierarchicalClassifier(nn.Module):
    def __init__(self, hidden_dim, max_levels=4, temperature=0.1):
        super().__init__()
        # 先不初始化聚类中心，后续在专门方法中进行
        self.cluster_centers = None

        # 超参数
        self.max_levels = max_levels
        self.temperature = temperature

    def initialize_cluster_centers(self, x):
        """
        使用 K-means++ 方法初始化聚类中心
        :param x: 节点特征 [N, D]
        """
        centers, _ = kmeans_plusplus(x.detach().cpu().numpy(), n_clusters=self.max_levels)
        self.cluster_centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=x.device))

    def forward(self, x, edge_index, batch):
        """
        x: 节点特征 [N, D]
        edge_index: 图的边索引 [2, E]
        batch: 图划分索引 [N]
        """
        if self.cluster_centers is None:
            self.initialize_cluster_centers(x)

        # 2. 批量计算聚类分配，使用余弦相似度
        # 计算节点特征与聚类中心之间的余弦相似度
        sim = F.cosine_similarity(x.unsqueeze(1), self.cluster_centers.unsqueeze(0), dim=-1)
        # 使用 softmax 分配
        assign = F.softmax(sim / self.temperature, dim=-1)  # [N, max_levels]

        distance = self.variance_distance_loss()

        return assign, distance

    def variance_distance_loss(self):
        cluster_centers = self.cluster_centers
        num_clusters = cluster_centers.size(0)
        distances = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                distance = torch.norm(cluster_centers[i] - cluster_centers[j])
                distances.append(distance)
        distances = torch.stack(distances)

        if num_clusters<3:
            return distance

        return -torch.var(distances)

"""
    (同上)forum模型
    4+1
"""
class TransGRUNet_1_4(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2,hidden_dim3,hidden_dim4,output_dim,max_levels):
        super(TransGRUNet_1_4, self).__init__()

        self.hierarchical_classifier = HierarchicalClassifier(hidden_dim4,max_levels=max_levels)

        self.seq_transformer = SequentialRNN(hidden_dim4, hidden_dim4, num_layers=1)
        # self.seq_transformer = SequentialTrans(hidden_dim4, hidden_dim4, num_layers=1)

        self.conv1 = TransformerConv(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)

        self.conv2 = TransformerConv(hidden_dim, hidden_dim2)
        self.bn2 = nn.LayerNorm(hidden_dim2)

        self.conv3 = TransformerConv(hidden_dim2, hidden_dim3)
        self.bn3 = nn.LayerNorm(hidden_dim3)

        self.conv4 = TransformerConv(hidden_dim3, hidden_dim4)
        self.bn4 = nn.LayerNorm(hidden_dim4)

        # self.dropout = dropout
        self.activation = nn.GELU()
        self.linear1 = torch.nn.Linear(hidden_dim4, 8)
        self.linear2 = torch.nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, tree_edge_index,call_sequences,batch,prin=0):

        x = self.conv1(x, tree_edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, tree_edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        #
        x = self.conv3(x, tree_edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        #
        x = self.conv4(x, tree_edge_index)
        x = self.bn4(x)
        x = F.relu(x)

        # 层次聚类
        cluster_assignments,distance = self.hierarchical_classifier(x, tree_edge_index, batch)  # [N, max_levels]
        level_ids = torch.argmax(cluster_assignments, dim=1)  # [N]

        if(prin==1):
            print(level_ids)


        # 调用时序 Transformer 处理每张图的层次化数据
        global_outputs = self.process_graph_with_transformer(level_ids, call_sequences, x, batch)
        # global_outputs = self.process_graph_with_transformer(level_ids, call_sequences, x_with_cluster, batch)

        x = self.linear1(global_outputs)
        x = self.activation(x)
        x = self.linear2(x)

        x = self.sigmoid(x)
        return x,distance



    def process_graph_with_transformer(self, cluster_assignments, call_sequences, x, batch):
        """
        按批次级别统一构建时序输入，并通过时序 Transformer 一次性学习
        :param cluster_assignments: 每个节点的层次分配结果
        :param call_sequences: 节点的时序信息
        :param x: 节点特征 (num_nodes, input_hidden)
        :param batch: 每个节点对应的图索引
        :return: 每个节点的最终特征 (num_nodes, hidden_dim)
        """
        # 初始化全局结果张量，与输入 x 形状相同
        global_outputs = torch.zeros_like(x, device=x.device)

        batch_size = batch.max().item() + 1
        global_max_seq_len = 0
        layer_features_list = []  # 存储所有层次特征
        layer_masks_list = []  # 存储所有层次掩码
        layer_indices_list = []  # 存储所有层次节点索引

        # 遍历每张图，找到全局最大层次节点数
        for i in range(batch_size):
            nodes_in_batch = (batch == i).nonzero(as_tuple=True)[0]
            graph_clusters = cluster_assignments[nodes_in_batch]
            graph_clusters = torch.tensor(graph_clusters)

            for layer_id in range(graph_clusters.max().item() + 1):
                layer_nodes = nodes_in_batch[graph_clusters == layer_id]
                global_max_seq_len = max(global_max_seq_len, layer_nodes.size(0))

        # 构建所有图的层次特征和掩码
        for i in range(batch_size):
            nodes_in_batch = (batch == i).nonzero(as_tuple=True)[0]
            graph_features = x[nodes_in_batch]
            graph_calls = call_sequences[nodes_in_batch]
            graph_clusters = cluster_assignments[nodes_in_batch]
            graph_clusters = torch.tensor(graph_clusters)

            for layer_id in range(graph_clusters.max().item() + 1):
                layer_nodes = nodes_in_batch[graph_clusters == layer_id]
                layer_features_raw = graph_features[graph_clusters == layer_id]
                layer_calls = graph_calls[graph_clusters == layer_id]

                # 如果只有一个节点，直接保留原始特征
                if layer_nodes.numel() == 1:
                    global_outputs[layer_nodes] = layer_features_raw
                    continue

                # 按时序排序
                sorted_indices = torch.argsort(layer_calls, descending=False)
                sorted_features = layer_features_raw[sorted_indices]
                sorted_original_indices = layer_nodes[sorted_indices]

                # 填充到全局最大长度
                seq_len = sorted_features.size(0)
                pad_size = global_max_seq_len - seq_len
                if pad_size > 0:
                    padding = torch.zeros((pad_size, sorted_features.size(1)), device=x.device)
                    sorted_features = torch.cat([sorted_features, padding], dim=0)
                    padding_indices = -torch.ones(pad_size, dtype=torch.long, device=x.device)
                    sorted_original_indices = torch.cat([sorted_original_indices, padding_indices], dim=0)

                # 保存特征、掩码和索引
                layer_features_list.append(sorted_features)
                layer_masks_list.append(torch.tensor([0] * seq_len + [1] * pad_size, dtype=torch.bool, device=x.device))
                layer_indices_list.append(sorted_original_indices)

        # 将所有层次堆叠为统一大小的张量
        sequential_data = torch.stack(layer_features_list, dim=0)  # (num_layers, global_max_seq_len, input_hidden)
        mask = torch.stack(layer_masks_list, dim=0)  # (num_layers, global_max_seq_len)
        indices = torch.stack(layer_indices_list, dim=0)  # (num_layers, global_max_seq_len)

        # 批量输入到时序 Transformer
        layer_outputs = self.seq_transformer(sequential_data, mask)  # (num_layers, global_max_seq_len, hidden_dim)

        # 将 Transformer 输出还原到全局结果
        for layer_output, layer_indices, layer_mask in zip(layer_outputs, indices, mask):
            valid_indices = layer_indices[layer_indices >= 0]  # 有效节点索引
            global_outputs[valid_indices] = layer_output[layer_mask == 0]

        return global_outputs





class TransGRUNet_2_4(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2,hidden_dim3,hidden_dim4,output_dim,max_levels):
        super(TransGRUNet_2_4, self).__init__()

        self.hierarchical_classifier = HierarchicalClassifier(hidden_dim4,max_levels=max_levels)

        self.seq_transformer = SequentialRNN(hidden_dim4, hidden_dim4, num_layers=1)

        self.conv1 = TransformerConv(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)

        self.conv2 = TransformerConv(hidden_dim, hidden_dim3)
        self.bn2 = nn.LayerNorm(hidden_dim3)

        self.conv4 = TransformerConv(hidden_dim3, hidden_dim4)
        self.bn4 = nn.LayerNorm(hidden_dim4)

        # self.dropout = dropout
        self.activation = nn.GELU()
        self.linear1 = torch.nn.Linear(hidden_dim4, 8)
        self.linear2 = torch.nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, tree_edge_index,call_sequences,batch,prin=0):

        x = self.conv1(x, tree_edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, tree_edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x, tree_edge_index)
        x = self.bn4(x)
        x = F.relu(x)

        # 层次聚类
        cluster_assignments, distance = self.hierarchical_classifier(x, tree_edge_index, batch)  # [N, max_levels]
        level_ids = torch.argmax(cluster_assignments, dim=1)  # [N]


        # 调用时序 Transformer 处理每张图的层次化数据
        global_outputs = self.process_graph_with_transformer(level_ids, call_sequences, x, batch)

        x = self.linear1(global_outputs)
        x = self.activation(x)
        x = self.linear2(x)

        x = self.sigmoid(x)

        return x,distance

    def process_graph_with_transformer(self, cluster_assignments, call_sequences, x, batch):
        """
        按批次级别统一构建时序输入，并通过时序 Transformer 一次性学习
        :param cluster_assignments: 每个节点的层次分配结果
        :param call_sequences: 节点的时序信息
        :param x: 节点特征 (num_nodes, input_hidden)
        :param batch: 每个节点对应的图索引
        :return: 每个节点的最终特征 (num_nodes, hidden_dim)
        """
        # 初始化全局结果张量，与输入 x 形状相同
        global_outputs = torch.zeros_like(x, device=x.device)

        batch_size = batch.max().item() + 1
        global_max_seq_len = 0
        layer_features_list = []  # 存储所有层次特征
        layer_masks_list = []  # 存储所有层次掩码
        layer_indices_list = []  # 存储所有层次节点索引

        # 遍历每张图，找到全局最大层次节点数
        for i in range(batch_size):
            nodes_in_batch = (batch == i).nonzero(as_tuple=True)[0]
            graph_clusters = cluster_assignments[nodes_in_batch]
            graph_clusters = torch.tensor(graph_clusters)

            for layer_id in range(graph_clusters.max().item() + 1):
                layer_nodes = nodes_in_batch[graph_clusters == layer_id]
                global_max_seq_len = max(global_max_seq_len, layer_nodes.size(0))

        # 构建所有图的层次特征和掩码
        for i in range(batch_size):
            nodes_in_batch = (batch == i).nonzero(as_tuple=True)[0]
            graph_features = x[nodes_in_batch]
            graph_calls = call_sequences[nodes_in_batch]
            graph_clusters = cluster_assignments[nodes_in_batch]
            graph_clusters = torch.tensor(graph_clusters)

            for layer_id in range(graph_clusters.max().item() + 1):
                layer_nodes = nodes_in_batch[graph_clusters == layer_id]
                layer_features_raw = graph_features[graph_clusters == layer_id]
                layer_calls = graph_calls[graph_clusters == layer_id]

                # 如果只有一个节点，直接保留原始特征
                if layer_nodes.numel() == 1:
                    global_outputs[layer_nodes] = layer_features_raw
                    continue

                # 按时序排序
                sorted_indices = torch.argsort(layer_calls, descending=False)
                sorted_features = layer_features_raw[sorted_indices]
                sorted_original_indices = layer_nodes[sorted_indices]

                # 填充到全局最大长度
                seq_len = sorted_features.size(0)
                pad_size = global_max_seq_len - seq_len
                if pad_size > 0:
                    padding = torch.zeros((pad_size, sorted_features.size(1)), device=x.device)
                    sorted_features = torch.cat([sorted_features, padding], dim=0)
                    padding_indices = -torch.ones(pad_size, dtype=torch.long, device=x.device)
                    sorted_original_indices = torch.cat([sorted_original_indices, padding_indices], dim=0)

                # 保存特征、掩码和索引
                layer_features_list.append(sorted_features)
                layer_masks_list.append(torch.tensor([0] * seq_len + [1] * pad_size, dtype=torch.bool, device=x.device))
                layer_indices_list.append(sorted_original_indices)

        # 将所有层次堆叠为统一大小的张量
        sequential_data = torch.stack(layer_features_list, dim=0)  # (num_layers, global_max_seq_len, input_hidden)
        mask = torch.stack(layer_masks_list, dim=0)  # (num_layers, global_max_seq_len)
        indices = torch.stack(layer_indices_list, dim=0)  # (num_layers, global_max_seq_len)

        # 批量输入到时序 Transformer
        layer_outputs = self.seq_transformer(sequential_data, mask)  # (num_layers, global_max_seq_len, hidden_dim)

        # 将 Transformer 输出还原到全局结果
        for layer_output, layer_indices, layer_mask in zip(layer_outputs, indices, mask):
            valid_indices = layer_indices[layer_indices >= 0]  # 有效节点索引
            global_outputs[valid_indices] = layer_output[layer_mask == 0]

        return global_outputs
