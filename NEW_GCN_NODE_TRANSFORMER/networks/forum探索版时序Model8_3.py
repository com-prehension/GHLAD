import torch
from torch_geometric.nn import TransformerConv, GCNConv, GATConv
import torch.nn as nn
import torch.nn.functional as F



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
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # 直接注册缓冲区，设备由模型自动管理

    def forward(self, x):
        # PyTorch 会自动将缓冲区 `pe` 移动到与输入 x 相同的设备
        return x + self.pe[:, :x.size(1), :]


from sklearn.cluster import kmeans_plusplus



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

    def forward(self, x):
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
        # 确保所有操作都在和 cluster_centers 相同的设备上进行
        device = cluster_centers.device

        # 生成所有聚类中心对的组合，在对应设备上运行
        indices_i, indices_j = torch.triu_indices(num_clusters, num_clusters, offset=1, device=device)
        # 计算所有聚类中心对的差值
        diffs = cluster_centers[indices_i] - cluster_centers[indices_j]
        # 计算所有聚类中心对的距离
        distances = torch.norm(diffs, dim=1)

        if num_clusters < 3:
            return distances[0] if distances.numel() > 0 else torch.tensor(0.0, device=device)

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
        self.max_layer=max_levels

    def forward(self, x, tree_edge_index, call_sequences, batch,max_len, prin=0):

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
        cluster_assignments, distance = self.hierarchical_classifier(x)
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


    def process_graph_with_transformer(self, cluster_assignments, call_sequences, x, batch, max_len):
        device = x.device
        global_outputs = torch.zeros_like(x)
        batch_size = batch.max().item() + 1
        global_max_seq_len = max_len[0]

        # 生成批次和层次的组合索引
        batch_layer_indices = torch.cartesian_prod(torch.arange(batch_size, device=device),
                                                   torch.arange(self.max_layer, device=device))
        batch_indices = batch_layer_indices[:, 0]
        layer_indices = batch_layer_indices[:, 1]

        # 为每个批次和层次创建掩码，标记哪些节点属于该批次和层次
        batch_layer_masks = (batch.unsqueeze(0) == batch_indices.unsqueeze(1)) & \
                            (cluster_assignments.unsqueeze(0) == layer_indices.unsqueeze(1))

        # 筛选出每个批次和层次的节点索引
        batch_layer_nodes = [torch.nonzero(mask, as_tuple=True)[0] for mask in batch_layer_masks]

        # 初始化存储所有层次特征、掩码和索引的张量
        all_layer_features = torch.zeros((batch_size * self.max_layer, global_max_seq_len, x.size(1)), device=device)
        all_layer_masks = torch.zeros((batch_size * self.max_layer, global_max_seq_len), dtype=torch.bool,
                                      device=device)
        all_layer_indices = torch.full((batch_size * self.max_layer, global_max_seq_len), -1, dtype=torch.long,
                                       device=device)

        for idx, nodes in enumerate(batch_layer_nodes):
            if nodes.numel() == 0:
                continue
            layer_features_raw = x[nodes]
            layer_calls = call_sequences[nodes]

            if nodes.numel() == 1:
                global_outputs[nodes] = layer_features_raw
                continue

            sorted_indices = torch.argsort(layer_calls, descending=False)
            sorted_features = layer_features_raw[sorted_indices]
            sorted_original_indices = nodes[sorted_indices]

            seq_len = sorted_features.size(0)
            all_layer_features[idx, :seq_len] = sorted_features
            all_layer_masks[idx, :seq_len] = 1
            all_layer_indices[idx, :seq_len] = sorted_original_indices

        # 批量输入到时序 Transformer
        layer_outputs = self.seq_transformer(all_layer_features, all_layer_masks)

        # 优化输出赋值
        valid_mask = all_layer_indices >= 0
        valid_outputs = layer_outputs[valid_mask.unsqueeze(-1).expand_as(layer_outputs)].reshape(-1,
                                                                                                 layer_outputs.size(-1))
        valid_indices = all_layer_indices[valid_mask]
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
        self.max_layer=max_levels

    def forward(self, x, tree_edge_index, call_sequences, batch,max_len, prin=0):
        x = self.conv1(x, tree_edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, tree_edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x, tree_edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        cluster_assignments, distance = self.hierarchical_classifier(x)
        level_ids = torch.argmax(cluster_assignments, dim=1)
        global_outputs = self.process_graph_with_transformer(level_ids, call_sequences, x, batch,max_len)
        x = self.linear1(global_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x, distance

    def process_graph_with_transformer(self, cluster_assignments, call_sequences, x, batch, max_len):
        device = x.device
        global_outputs = torch.zeros_like(x)
        batch_size = batch.max().item() + 1
        global_max_seq_len = max_len[0]

        # 生成批次和层次的组合索引
        batch_layer_indices = torch.cartesian_prod(torch.arange(batch_size, device=device),
                                                   torch.arange(self.max_layer, device=device))
        batch_indices = batch_layer_indices[:, 0]
        layer_indices = batch_layer_indices[:, 1]

        # 为每个批次和层次创建掩码，标记哪些节点属于该批次和层次
        batch_layer_masks = (batch.unsqueeze(0) == batch_indices.unsqueeze(1)) & \
                            (cluster_assignments.unsqueeze(0) == layer_indices.unsqueeze(1))

        # 筛选出每个批次和层次的节点索引
        batch_layer_nodes = [torch.nonzero(mask, as_tuple=True)[0] for mask in batch_layer_masks]

        # 初始化存储所有层次特征、掩码和索引的张量
        all_layer_features = torch.zeros((batch_size * self.max_layer, global_max_seq_len, x.size(1)), device=device)
        all_layer_masks = torch.zeros((batch_size * self.max_layer, global_max_seq_len), dtype=torch.bool,
                                      device=device)
        all_layer_indices = torch.full((batch_size * self.max_layer, global_max_seq_len), -1, dtype=torch.long,
                                       device=device)

        for idx, nodes in enumerate(batch_layer_nodes):
            if nodes.numel() == 0:
                continue
            layer_features_raw = x[nodes]
            layer_calls = call_sequences[nodes]

            if nodes.numel() == 1:
                global_outputs[nodes] = layer_features_raw
                continue

            sorted_indices = torch.argsort(layer_calls, descending=False)
            sorted_features = layer_features_raw[sorted_indices]
            sorted_original_indices = nodes[sorted_indices]

            seq_len = sorted_features.size(0)
            all_layer_features[idx, :seq_len] = sorted_features
            all_layer_masks[idx, :seq_len] = 1
            all_layer_indices[idx, :seq_len] = sorted_original_indices

        # 批量输入到时序 Transformer
        layer_outputs = self.seq_transformer(all_layer_features, all_layer_masks)

        # 优化输出赋值
        valid_mask = all_layer_indices >= 0
        valid_outputs = layer_outputs[valid_mask.unsqueeze(-1).expand_as(layer_outputs)].reshape(-1,
                                                                                                 layer_outputs.size(-1))
        valid_indices = all_layer_indices[valid_mask]
        global_outputs[valid_indices] = valid_outputs

        return global_outputs

