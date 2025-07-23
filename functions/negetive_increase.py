import  torch
def balance_features_labels(features, labels,device, up_sample_scale_factor):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # 计算标签为 0 和 1 的节点个数
    count_0 = torch.sum(labels == 0).item()
    count_1 = torch.sum(labels == 1).item()

    # 找到标签数量较少的那一类的索引
    minority_label = 0 if count_0 < count_1 else 1
    minority_indices = torch.where(labels == minority_label)[0]

    if len(minority_indices) == 0 or up_sample_scale_factor == 0:
        return features, labels

    # sampling
    selected_indices = minority_indices.repeat(up_sample_scale_factor)

    # 复制并添加特征
    selected_features = features[selected_indices]
    features = torch.cat([features, selected_features], dim=0)
    # 更新标签
    added_labels = torch.ones(len(minority_indices) * up_sample_scale_factor, dtype=labels.dtype) * minority_label
    labels=labels.to(device)
    added_labels=added_labels.to(device)
    # if torch.cuda.is_available():
    # added_labels = added_labels.cuda()
    labels = torch.cat([labels, added_labels])
    # if device=='cuda':
    #     added_labels = added_labels.cuda()
    #     labels = torch.cat([labels, added_labels])
    # else:
    #     labels = torch.cat([labels, added_labels])
    return features, labels

from imblearn.over_sampling import SMOTE
def balance_features_labels_new(features, labels,device, over_sample_scale_factor, sample_method="SMOTE"):

    # 计算标签为 0 和 1 的节点个数
    count_0 = torch.sum(labels == 0).item()
    count_1 = torch.sum(labels == 1).item()

    # 找到标签数量较少的那一类的索引
    minority_label = 0 if count_0 < count_1 else 1
    minority_indices = torch.where(labels == minority_label)[0]

    # device='cpu'

    if len(minority_indices) <= 1 or over_sample_scale_factor == 0:
        return features, labels

    if sample_method == "SMOTE":
        """SMOTE sampling"""
        n_neighbor = len(minority_indices) - 1 if len(
            minority_indices) - 1 < over_sample_scale_factor else over_sample_scale_factor
        smote = SMOTE(sampling_strategy={minority_label: len(minority_indices) * over_sample_scale_factor},
                      random_state=42, k_neighbors=n_neighbor)
        features, labels = smote.fit_resample(features, labels)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features = torch.tensor(features, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        return features.squeeze(), labels
    if sample_method == "copy":
        """copy minority"""
        selected_indices = minority_indices.repeat(over_sample_scale_factor)
        # 复制并添加特征
        selected_features = features[selected_indices]
        features = torch.cat([features, selected_features], dim=0)
        # 更新标签
        added_labels = torch.ones(len(minority_indices) * over_sample_scale_factor, dtype=labels.dtype) * minority_label
        if torch.cuda.is_available():
            added_labels = added_labels.cuda()
        labels = torch.cat([labels, added_labels])
        return features, labels
