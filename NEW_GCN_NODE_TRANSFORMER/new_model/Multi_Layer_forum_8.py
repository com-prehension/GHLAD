import os
import gc
import torch
import logging
import numpy as np
import random
from NEW_GCN_NODE_TRANSFORMER.networks.forum探索版时序Model8  import TransGRUNet_1_4,TransGRUNet_2_4
import torch.nn as nn
from tqdm import tqdm
from NEW_GCN_NODE_TRANSFORMER.functions.utils import read_one_hot,deal_Graph
from torch_geometric.loader import DataLoader
from functions.precision_index import show_metrics_AUPRC_new
from functions.negetive_increase import balance_features_labels,balance_features_labels_new
from NEW_GCN_NODE_TRANSFORMER.functions.new_utils import parse_graphs_to_dataset_forum,parse_graphs_to_dataset
from torch import optim



def validation(net, t_val_dataloader,val_result_list,device,storage_path, rewrite):
    net.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    # total_node_num = 0
    val_true_labels = []
    val_predicted_labels = []
    val_predicted_prob=[]
    with torch.no_grad():
        for tree in t_val_dataloader:
            tree=tree.to(device)
            output,distance = net(tree.x , tree.edge_index,tree.w,tree.batch,tree.max_len)
            loss = criterion(output.squeeze().float(), tree.y.float())
            total_loss += loss.item()
            # total_node_num += batch.num_nodes

            # y_pred = torch.argmax(output, dim=1)
            # 存储真实标签和预测标签
            pred_prob=output.squeeze().cpu().numpy()

            val_true_labels.extend(tree.y.cpu().numpy())
            val_predicted_prob.extend(pred_prob)
    val_predicted_labels = (np.array(val_predicted_prob) >= 0.5).astype(int)
    show_metrics_AUPRC_new(val_true_labels, val_predicted_labels,val_predicted_prob, storage_path,val_result_list, rewrite)
    # 计算 F1 分数、召回率、精确度、auc
    avg_loss = total_loss / len(t_val_dataloader)

    print("validation avg loss:", avg_loss)
    # with open(storage_path, "a") as file:
    #     file.write("validation avg loss:  "+ str(avg_loss)+ '\n')
    #     file.close()
    val_result_list += ["validation avg loss:" + str(avg_loss) + '\n']
    return avg_loss

def train(net,train_dataloader,device,xent,optimizer,logger,train_loss_list,scheduler,main_weight,distance_weight):
    net.train()
    loss_epoch = 0.0
    n_batches = 0
    for tree in train_dataloader:
        # print(data)
        tree = tree.to(device)

        optimizer.zero_grad()
        outputs,distance = net(tree.x, tree.edge_index, tree.w, tree.batch,tree.max_len)

        # print(outputs)

        # y_pred_score, y = balance_features_labels_new(outputs.squeeze(), tree.y, device, 9,"copy")
        y_pred_score, y = balance_features_labels_new(outputs.squeeze(), tree.y, device, 6,"copy")
        # y_pred_score, y = balance_features_labels_new(outputs, tree.y, device, 6,"SMOTE")
        # y_pred_score, y = balance_features_labels_new(outputs, tree.y, device, 6)
        labels = y.float()

        # print(y_pred_score)
        # print(labels)
        loss = xent(y_pred_score, labels)

        total_loss = main_weight * loss + distance_weight * distance
        # loss = distance_weight * distance
        total_loss.backward()

        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

        optimizer.step()
        loss_epoch += total_loss.item()
        n_batches += 1


    scheduler.step()
    logger.info('\nLoss: {:.8f}'.format(loss_epoch / len(train_dataloader)))
    train_loss_list += ["Average Loss:" + str(loss_epoch / len(train_dataloader))]

def increase_negetive_labels(pred_probs,true_labels):
    new_true_labels=[]
    new_pred_probs=[]
    for i in range(len(true_labels)):
        if true_labels[i] ==1:
            new_true_labels.extend(true_labels[i] for x in range(9))
            new_pred_probs.extend(pred_probs[i] for x in range(9))
        else:
            new_true_labels.append(true_labels[i])
            new_pred_probs.append(pred_probs[i])
    new_pred_probs=torch.tensor(new_pred_probs,requires_grad=True)
    new_true_labels=torch.tensor(new_true_labels,requires_grad=True)

    return new_pred_probs,new_true_labels

def set_random_seed(seed=42):
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def print_weights(model):
    for name, param in model.named_parameters():
        # if 'conv' in name:
            print(f"{name} - mean: {param.mean().item()}, std: {param.std().item()}")

def main(dataname,train_tree_set, val_tree_set, test_data_set, test_one_set,storage_path,loss_path,test_one_path):
    torch.set_printoptions(threshold=np.inf)
    """
    Pytorch implementation for LogGD
    author: deleteme
    time: 2023/2/6
    """
    seed = 42
    set_random_seed(seed)
    # Get configuration
    # cfg = Config(locals().copy())
    dataset_name='数据集3'
    # seed=-1
    # lr=0.0001

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = 'log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Dataset: %s' % dataset_name)

    # batch_size=128
    batch_size=512

    train_dataset = parse_graphs_to_dataset(train_tree_set)

    val_dataset = parse_graphs_to_dataset(val_tree_set)

    test_one_dataset = parse_graphs_to_dataset(test_one_set)

    test_dataset = parse_graphs_to_dataset(test_data_set)


    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=30)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=30)
    # test_one_dataloader = DataLoader(test_one_dataset, batch_size=batch_size, shuffle=False,num_workers=30)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=30)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_one_dataloader = DataLoader(test_one_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    input_dim = 247

    if dataname == "halo":
        input_dim = 484
    elif dataname == "novel":
        input_dim = 499

    # input_dim = 247
    # input_dim = 484
    hidden_dim = 128
    # hidden_dim = 128
    # hidden_dim = 64
    hidden_dim2 = 32
    # hidden_dim2 = 64
    hidden_dim3 = 32
    # hidden_dim4 = 64
    # hidden_dim4 = 32
    hidden_dim4 = 16
    output_dim = 1
    main_weight = 1
    # distance_weight =0.001
    distance_weight =0.001
    # distance_weight =1

    max_levels = 3
    '''测试4+1+全局'''
    # net = TransGRUNet_1(input_dim,hidden_dim,hidden_dim2,hidden_dim3,hidden_dim4,output_dim,cluster_dim,num_clusters)
    # '''测试4+全局+1'''
    # net = TransGRUNet_1_2(input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, cluster_dim,num_clusters)
    # '''测试4+全局'''
    # net = TransGRUNet_1_3(input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, cluster_dim,num_clusters)
    # net = TransGRUNet_1_3_1(input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, cluster_dim,num_clusters)
    # '''测试4+1'''
    # net = TransGRUNet_1_4(input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, max_levels)
    net = TransGRUNet_2_4(input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, max_levels)

    xent = nn.BCELoss()

    optimizer = torch.optim.AdamW(net.parameters(),  lr=1e-3, weight_decay=1e-3)
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Weights after initialization:")
    print_weights(net)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2,5,7], gamma=0.3)

    set_random_seed(seed)
    net.train()
    net=net.to(device)

    train_loss_list=[]
    val_result_list=[]
    rewrite=1

    with open(loss_path, "w") as file:
        file.write("数据result" + '\n')
        file.close()

    for epoch in tqdm(range(10)):

        for param_group in optimizer.param_groups:
            print(f"\n Epoch {epoch + 1}, Learning Rate: {param_group['lr']}")
            with open(loss_path, "a") as file:
                file.write("Learning Rate:  " + str(param_group['lr']) + '\n')
                file.close()

        train(net,train_dataloader,device,xent,optimizer,logger,train_loss_list,scheduler,main_weight,distance_weight)
        print("===========================")
        print("==========开始验证==========")
        if epoch !=0:
            rewrite=0

        validation(net, val_dataloader, val_result_list,device, loss_path, rewrite)

    print("Weights after training:")
    print_weights(net)

    # Test model
    net.eval()
    y_true_all = []
    y_pred_all = []
    y_pred_all_prob = []
    test_one_list=[]

    print("start one sort test")
    with torch.no_grad():
        for tree in test_one_dataloader:
            # print(tree.n)

            # 一张图
            tree = tree.to(device)
            embeds,distance = net(tree.x, tree.edge_index,tree.w,tree.batch,tree.max_len,prin=1)

            pred_prob = embeds.squeeze().cpu().numpy()

            predicted_labels = (np.array(pred_prob) >= 0.5).astype(int)

            new_labels = []
            new_predicted_labels = []
            new_predicted_probs = []

            graph_len, graphs_nodes = deal_Graph(tree.z)
            labels = tree.y  # 真实类别标签

            # 处理每张图的标签
            start = 0
            end = graph_len[0]
            z = 0
            graph_len_new = [0]
            for i in range(len(graphs_nodes)):
                if start >= labels.size(0):
                    break
                node_labels = labels.tolist()[start:end]
                node_pre_labels = predicted_labels.tolist()[start:end]
                node_pre_probs = pred_prob.tolist()[start:end]
                z += len(graphs_nodes[i])
                graph_len_new.append(z)
                for key in graphs_nodes[i].keys():
                    node_label = []
                    node_pre_label = []
                    node_pre_prob = []
                    l1 = graphs_nodes[i][key]

                    for j in l1:
                        node_label.append(node_labels[j])
                        node_pre_label.append(node_pre_labels[j])
                        node_pre_prob.append(node_pre_probs[j])
                    if node_labels[l1[0]] == 1 and 1 in node_pre_label:
                        new_labels.append(1)
                        new_predicted_labels.append(1)
                    elif node_labels[l1[0]] == 1 and 1 not in node_pre_label:
                        new_labels.append(1)
                        new_predicted_labels.append(0)
                    elif node_labels[l1[0]] == 0 and node_label == node_pre_label:
                        new_labels.append(0)
                        new_predicted_labels.append(0)
                    else:
                        new_labels.append(0)
                        new_predicted_labels.append(1)
                    new_predicted_probs.append(node_pre_prob[0])
                start = end
                if end < labels.size(0):
                    end += graph_len[i + 1]
            y_true_all.append(torch.tensor(new_labels))
            y_pred_all.append(torch.tensor(new_predicted_labels))
            y_pred_all_prob.append(torch.tensor(new_predicted_probs))

    y_true_all = torch.cat(y_true_all, dim=0)
    y_pred_all = torch.cat(y_pred_all, dim=0)
    y_pred_all_prob = torch.cat(y_pred_all_prob, dim=0)

    show_metrics_AUPRC_new(y_true_all, y_pred_all, y_pred_all_prob, test_one_path,test_one_list)

    y_true_all = []
    y_pred_all = []
    y_pred_all_prob = []
    test_list=[]
    net.eval()

    print("start test")
    with torch.no_grad():
        for tree in test_dataloader:
            # 一张图
            tree = tree.to(device)

            embeds,distance = net(tree.x, tree.edge_index,tree.w,tree.batch,tree.max_len)

            pred_prob = embeds.squeeze().cpu().numpy()

            predicted_labels = (np.array(pred_prob) >= 0.5).astype(int)

            new_labels = []
            new_predicted_labels = []
            new_predicted_probs = []

            graph_len, graphs_nodes = deal_Graph(tree.z)
            labels = tree.y  # 真实类别标签

            # 处理每张图的标签
            start = 0
            end = graph_len[0]
            z = 0
            graph_len_new = [0]
            for i in range(len(graphs_nodes)):
                if start >= labels.size(0):
                    break
                node_labels = labels.tolist()[start:end]
                node_pre_labels = predicted_labels.tolist()[start:end]
                node_pre_probs = pred_prob.tolist()[start:end]
                z += len(graphs_nodes[i])
                graph_len_new.append(z)
                for key in graphs_nodes[i].keys():
                    node_label = []
                    node_pre_label = []
                    node_pre_prob = []
                    l1 = graphs_nodes[i][key]

                    for j in l1:
                        node_label.append(node_labels[j])
                        node_pre_label.append(node_pre_labels[j])
                        node_pre_prob.append(node_pre_probs[j])
                    if node_labels[l1[0]] == 1 and 1 in node_pre_label:
                        new_labels.append(1)
                        new_predicted_labels.append(1)
                    elif node_labels[l1[0]] == 1 and 1 not in node_pre_label:
                        new_labels.append(1)
                        new_predicted_labels.append(0)
                    elif node_labels[l1[0]] == 0 and node_label == node_pre_label:
                        new_labels.append(0)
                        new_predicted_labels.append(0)
                    else:
                        new_labels.append(0)
                        new_predicted_labels.append(1)
                    new_predicted_probs.append(node_pre_prob[0])
                start = end
                if end < labels.size(0):
                    end += graph_len[i + 1]
            y_true_all.append(torch.tensor(new_labels))
            y_pred_all.append(torch.tensor(new_predicted_labels))
            y_pred_all_prob.append(torch.tensor(new_predicted_probs))

    y_true_all = torch.cat(y_true_all, dim=0)
    y_pred_all = torch.cat(y_pred_all, dim=0)
    y_pred_all_prob = torch.cat(y_pred_all_prob, dim=0)

    show_metrics_AUPRC_new(y_true_all, y_pred_all, y_pred_all_prob, storage_path,test_list, rewrite=0)

    with open(loss_path, "w") as file:
        for item in val_result_list:
            file.write(item)
        for item in train_loss_list:
            file.write(item + "\n")
        file.close()

    with open(storage_path, "w") as file:
        for item in test_list:
            file.write(item)
        file.close()

    # 强制执行垃圾回收
    gc.collect()
    # 释放GPU资源
    torch.cuda.empty_cache()

    print("Training and testing completed.")

if __name__ == '__main__':
    print("hello")