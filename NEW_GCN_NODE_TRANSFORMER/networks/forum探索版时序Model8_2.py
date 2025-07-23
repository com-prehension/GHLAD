import torch
from torch_geometric.nn import TransformerConv, GCNConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import kmeans_plusplus


class SequentialRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, rnn_type="GRU"):
        super(SequentialRNN, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        if rnn_type == "GRU":
            self.rnn1 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
            self.seq_tn1 = nn.LayerNorm(hidden_dim)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        else:
            raise ValueError("Unsupported RNN type: Choose 'GRU' or 'LSTM'")

    def forward(self, x, mask):
        # print(x.device)
        x = self.input_projection(x)

        x = self.positional_encoding(x)
        packed_output, _ = self.rnn1(x)
        packed_output = self.seq_tn1(packed_output)
        packed_output = F.relu(packed_output)
        return packed_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, device=None):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if device is not None:
            pe = pe.to(device)  # 显式将 pe 移动到指定设备
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class HierarchicalClassifier(nn.Module):
    def __init__(self, hidden_dim, max_levels=4, temperature=0.1):
        super().__init__()
        self.cluster_centers = None
        self.max_levels = max_levels
        self.temperature = temperature

    def initialize_cluster_centers(self, x):
        """
        在 CPU 上进行聚类中心初始化，因为 sklearn 的 kmeans_plusplus 不支持 GPU 计算。
        如果要在 GPU 上执行，可以考虑使用基于 GPU 的聚类算法（如 faiss 库）。
        """
        x_cpu = x.detach().cpu().numpy()
        centers, _ = kmeans_plusplus(x_cpu, n_clusters=self.max_levels)
        self.cluster_centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=x.device))

    def forward(self, x, edge_index, batch):
        if self.cluster_centers is None:
            self.initialize_cluster_centers(x)
        sim = F.cosine_similarity(x.unsqueeze(1), self.cluster_centers.unsqueeze(0), dim=-1)
        assign = F.softmax(sim / self.temperature, dim=-1)
        distance = self.variance_distance_loss()
        return assign, distance

    def variance_distance_loss(self):
        cluster_centers = self.cluster_centers
        num_clusters = cluster_centers.size(0)
        if num_clusters < 3:
            return 0
        distances = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                distances.append(torch.norm(cluster_centers[i] - cluster_centers[j]))
        distances = torch.stack(distances)
        return -torch.var(distances)


class TransGRUNet_1_4(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, max_levels):
        super(TransGRUNet_1_4, self).__init__()
        self.hierarchical_classifier = HierarchicalClassifier(hidden_dim4, max_levels=max_levels)
        self.seq_transformer = SequentialRNN(hidden_dim4, hidden_dim4, num_layers=1)
        self.conv1 = TransformerConv(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim2)
        self.bn2 = nn.LayerNorm(hidden_dim2)
        self.conv3 = TransformerConv(hidden_dim2, hidden_dim3)
        self.bn3 = nn.LayerNorm(hidden_dim3)
        self.conv4 = TransformerConv(hidden_dim3, hidden_dim4)
        self.bn4 = nn.LayerNorm(hidden_dim4)
        self.activation = nn.GELU()
        self.linear1 = torch.nn.Linear(hidden_dim4, 8)
        self.linear2 = torch.nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, tree_edge_index, call_sequences,max_len, batch, prin=0):

        x = self.conv1(x, tree_edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, tree_edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, tree_edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, tree_edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        cluster_assignments, distance = self.hierarchical_classifier(x, tree_edge_index, batch)
        level_ids = torch.argmax(cluster_assignments, dim=1)
        if prin == 1:
            print(level_ids)
        global_outputs = self.process_graph_with_transformer(level_ids, call_sequences, x, batch,max_len)

        x = self.linear1(global_outputs)
        # x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x, distance


    def process_graph_with_transformer(self, cluster_assignments, call_sequences, x, batch,max_len):
        """
        按批次级别统一构建时序输入，并通过时序 Transformer 一次性学习
        :param cluster_assignments: 每个节点的层次分配结果
        :param call_sequences: 节点的时序信息
        :param x: 节点特征 (num_nodes, input_hidden)
        :param batch: 每个节点对应的图索引
        :return: 每个节点的最终特征 (num_nodes, hidden_dim)
        """
        # 将输入数据移动到 CPU
        device=x.device
        # 初始化全局结果张量，与输入 x 形状相同，且在 CPU 上
        global_outputs = torch.zeros_like(x)

        batch_size = batch.max().item() + 1

        # 缓存每个批次的节点索引
        # 预计算每个批次的节点索引
        batch_nodes = [torch.nonzero(batch == i, as_tuple=True)[0].to(x.device) for i in range(batch_size)]

        global_max_seq_len = max_len

        # 初始化存储所有层次特征、掩码和索引的张量
        all_layer_features = torch.zeros(
            (batch_size, len(torch.unique(cluster_assignments)), global_max_seq_len, x.size(1)),
            device=device)
        all_layer_masks = torch.zeros((batch_size, len(torch.unique(cluster_assignments)), global_max_seq_len),
                                      dtype=torch.bool, device=device)
        all_layer_indices = torch.full((batch_size, len(torch.unique(cluster_assignments)), global_max_seq_len), -1,
                                       dtype=torch.long, device=device)

        for i in range(batch_size):
            nodes_in_batch = batch_nodes[i]
            graph_features = x[nodes_in_batch]
            graph_calls = call_sequences[nodes_in_batch]
            graph_clusters = cluster_assignments[nodes_in_batch]

            unique_layers = torch.unique(graph_clusters)
            for j, layer_id in enumerate(unique_layers):
                layer_mask = graph_clusters == layer_id
                layer_nodes = nodes_in_batch[layer_mask]
                layer_features_raw = graph_features[layer_mask]
                layer_calls = graph_calls[layer_mask]

                if layer_nodes.numel() == 1:
                    global_outputs[layer_nodes] = layer_features_raw
                    continue

                sorted_indices = torch.argsort(layer_calls, descending=False)
                sorted_features = layer_features_raw[sorted_indices]
                sorted_original_indices = layer_nodes[sorted_indices]

                seq_len = sorted_features.size(0)
                all_layer_features[i, j, :seq_len] = sorted_features
                all_layer_masks[i, j, :seq_len] = 1
                all_layer_indices[i, j, :seq_len] = sorted_original_indices

        # 展平并堆叠相关张量
        sequential_data = all_layer_features.view(-1, global_max_seq_len, x.size(1))
        mask = all_layer_masks.view(-1, global_max_seq_len)
        indices = all_layer_indices.view(-1, global_max_seq_len)

        # 批量输入到时序 Transformer
        layer_outputs = self.seq_transformer(sequential_data, mask)  # (num_layers, global_max_seq_len, hidden_dim)

        # 优化输出赋值
        valid_mask = indices >= 0
        valid_outputs = layer_outputs[valid_mask.unsqueeze(-1).expand_as(layer_outputs)].reshape(-1,
                                                                                                 layer_outputs.size(-1))
        valid_indices = indices[valid_mask]
        global_outputs[valid_indices] = valid_outputs

        return global_outputs


class TransGRUNet_2_4(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, max_levels):
        super(TransGRUNet_2_4, self).__init__()
        self.hierarchical_classifier = HierarchicalClassifier(hidden_dim4, max_levels=max_levels)
        self.seq_transformer = SequentialRNN(hidden_dim4, hidden_dim4, num_layers=1)
        self.conv1 = TransformerConv(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim3)
        self.bn2 = nn.LayerNorm(hidden_dim3)
        self.conv4 = TransformerConv(hidden_dim3, hidden_dim4)
        self.bn4 = nn.LayerNorm(hidden_dim4)
        self.activation = nn.GELU()
        self.linear1 = torch.nn.Linear(hidden_dim4, 8)
        self.linear2 = torch.nn.Linear(8, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, tree_edge_index, call_sequences, batch, prin=0):
        x = self.conv1(x, tree_edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, tree_edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x, tree_edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        cluster_assignments, distance = self.hierarchical_classifier(x, tree_edge_index, batch)
        level_ids = torch.argmax(cluster_assignments, dim=1)
        global_outputs = self.process_graph_with_transformer(level_ids, call_sequences, x, batch)
        x = self.linear1(global_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x, distance

    def process_graph_with_transformer(self, cluster_assignments, call_sequences, x, batch):
        global_outputs = torch.zeros_like(x, device=x.device)
        batch_size = batch.max().item() + 1
        layer_features_list = []
        layer_masks_list = []
        layer_indices_list = []

        # 提前计算每个图的最大层次节点数
        graph_max_seq_lens = [0] * batch_size
        for i in range(batch_size):
            nodes_in_batch = (batch == i).nonzero(as_tuple=True)[0]
            graph_clusters = cluster_assignments[nodes_in_batch]
            for layer_id in range(graph_clusters.max().item() + 1):
                layer_nodes = nodes_in_batch[graph_clusters == layer_id]
                graph_max_seq_lens[i] = max(graph_max_seq_lens[i], layer_nodes.size(0))
        global_max_seq_len = max(graph_max_seq_lens)

        for i in range(batch_size):
            nodes_in_batch = (batch == i).nonzero(as_tuple=True)[0]
            graph_features = x[nodes_in_batch]
            graph_calls = call_sequences[nodes_in_batch]
            graph_clusters = cluster_assignments[nodes_in_batch]
            for layer_id in range(graph_clusters.max().item() + 1):
                layer_nodes = nodes_in_batch[graph_clusters == layer_id]
                layer_features_raw = graph_features[graph_clusters == layer_id]
                layer_calls = graph_calls[graph_clusters == layer_id]
                if layer_nodes.numel() == 1:
                    global_outputs[layer_nodes] = layer_features_raw
                    continue
                sorted_indices = torch.argsort(layer_calls, descending=False)
                sorted_features = layer_features_raw[sorted_indices]
                sorted_original_indices = layer_nodes[sorted_indices]
                seq_len = sorted_features.size(0)
                pad_size = global_max_seq_len - seq_len
                if pad_size > 0:
                    padding = torch.zeros((pad_size, sorted_features.size(1)), device=x.device)
                    sorted_features = torch.cat([sorted_features, padding], dim=0)
                    padding_indices = -torch.ones(pad_size, dtype=torch.long, device=x.device)
                    sorted_original_indices = torch.cat([sorted_original_indices, padding_indices], dim=0)
                layer_features_list.append(sorted_features)
                layer_masks_list.append(torch.tensor([0] * seq_len + [1] * pad_size, dtype=torch.bool, device=x.device))
                layer_indices_list.append(sorted_original_indices)

        sequential_data = torch.stack(layer_features_list, dim=0)
        mask = torch.stack(layer_masks_list, dim=0)
        indices = torch.stack(layer_indices_list, dim=0)
        layer_outputs = self.seq_transformer(sequential_data, mask)
        for layer_output, layer_indices, layer_mask in zip(layer_outputs, indices, mask):
            valid_indices = layer_indices[layer_indices >= 0]
            global_outputs[valid_indices] = layer_output[layer_mask == 0]
        return global_outputs

