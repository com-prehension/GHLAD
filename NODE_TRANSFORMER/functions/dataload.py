import os
from functions.构建树 import build_tree_from_txt,construct_tree_to_nx
import random
import networkx as nx
from functions.utils import read_file_one_hot_new,read_exception_one_hot_new,read_one_hot
def open_file(directory,result):

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            log_nodes1(file_path,result)
            # fallback_encoding(file_path)

def fallback_encoding(file_path):
    lines=[]
    with open(file_path, "r",encoding="ISO-8859-1") as file:
        lines = file.readlines()
        file.close()

    file_path=file_path.replace("novel","novel_2")
    with open(file_path, "w",encoding="utf-8") as f:
        for line in lines:
            f.write(line)

def open_file_new(directory,result):

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_name=file_path.split("/")[-1]
            file_struct=file_name.split("-")[0]
            if file_struct in result:
                result[file_struct] = result[file_struct] + [file_path]
            else:
                result[file_struct] = [file_path]


def log_nodes1(file_path,result):
    with open(file_path, "r") as file:
        lines = file.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        substructures_all=""
        for line in lines[start_filter + 1:]:

            chain, time, invokeInfo, cost, event, exception = line.split(",")

            ##   edge修改这里
            edge_all=chain+ ',' + time+ ',' + invokeInfo+ ',' + event+ ',' + exception
            if substructures_all=="":
                substructures_all=substructures_all+edge_all.replace("\n","")
            else:
                substructures_all = substructures_all + "," + edge_all.replace("\n", "")
        if substructures_all in result:
            result[substructures_all]=result[substructures_all]+[file_path]
        else:
            result[substructures_all] = [file_path]

def process_files_to_construct_trees(train_paths,event_embedding_path):
    event_map = read_one_hot(event_embedding_path)
    tree_dataset = []
    for file_path in train_paths:
        root = build_tree_from_txt(file_path)
        g = construct_tree_to_nx(root, event_map)
        # g.remove_node("root")
        tree_dataset.append((root, g, file_path))
    return tree_dataset

def read_graph_structure(ID_paths,file_path):
    with open(file_path, "r",encoding="utf-8") as file:
        lines=file.readlines()
        for line in lines:
            ID=line.strip().split(" : ")[0].strip()
            path=line.strip().split(" : ")[1].strip()
            if ID not in ID_paths:
                ID_paths[ID]=[path]
            else:
                ID_paths[ID] += [path]
def read_graph_structure_new(ID_paths,file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            ID = line.split(" : ")[0]
            path = line.split(" : ")[1]
            all_path = path.split("->")[0]
            simple_path = path.split("->")[1].replace("\n", "")
            if ID not in ID_paths:
                ID_paths[ID] = all_path
    return ID_paths

def read_specific_data(file_path,train_paths,val_paths,test_paths):
    with open(file_path, "r",encoding="utf-8") as file:
        lines = file.readlines()
        # train_start_filter = lines.index("train set paths:\n")
        val_start_filter = lines.index("val dataset:\n")
        test_start_filter = lines.index("test dataset:\n")

        for line in lines[1:]:
            if "val" in line:
                break
            else:
                path = line.strip().split(" : ")[1].strip()
                train_paths.append(path)
        for line in lines[val_start_filter + 1:]:
            if "test" in line:
                break
            else:
                path = line.strip().split(" : ")[1].strip()
                val_paths.append(path)
        for line in lines[test_start_filter + 1:]:
            path = line.strip().split(" : ")[1].strip()
            test_paths.append(path)

    return train_paths,val_paths,test_paths


def read_specific_data_forum(file_path,train_paths,val_paths,test_paths,test_one_paths):
    test_one_dict={}
    with open(file_path, "r",encoding="utf-8") as file:
        lines = file.readlines()
        # train_start_filter = lines.index("train set paths:\n")
        val_start_filter = lines.index("val dataset:\n")
        test_start_filter = lines.index("test dataset:\n")

        for line in lines[1:]:
            if "val" in line:
                break
            else:
                path = line.strip().split(" : ")[1].strip()
                train_paths.append(path)
        for line in lines[val_start_filter + 1:]:
            if "test" in line:
                break
            else:
                path = line.strip().split(" : ")[1].strip()
                val_paths.append(path)
        for line in lines[test_start_filter + 1:]:
            ID = line.strip().split(" : ")[0].strip()

            path = line.strip().split(" : ")[1].strip()
            test_paths.append(path)

            if ID not in test_one_dict:
                test_one_dict[ID]=1
                test_one_paths.append(path)



    return train_paths,val_paths,test_paths

def record_substructure_to_datatxt(dataname, train_ratio, val_ratio, record, specify_number):
    result = {}
    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    directory = f"/root/autodl-tmp/project/data/{dataname}"

    i = 0
    sub_result = []

    open_file(directory, result)

    for key in result.keys():
        for path in result[key]:
            sub_result.append(str(i) + " : " + path)
        i = i + 1

    if record == 1:
        with open(graph_structure_path, "w", encoding="utf-8") as file:
            for line in sub_result:
                file.write(line)
                file.write("\n")
            file.close()

    key_list1 = list(result.keys())
    key_list = list(result.keys())
    random.shuffle(key_list)

    # key_list=key_list[:18]

    train_num = int(len(key_list) * train_ratio)
    val_num = int(len(key_list) * val_ratio)
    train_set_key = key_list[0: train_num]
    val_set_key = key_list[train_num: train_num + val_num]
    test_set_key = key_list[train_num + val_num:]

    specify_data = ["train dataset:"]

    for key in train_set_key:
        number = key_list1.index(key)
        train_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("val dataset:")
    for key in val_set_key:
        number = key_list1.index(key)
        val_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("test dataset:")
    for key in test_set_key:
        number = key_list1.index(key)
        test_paths += result[key]
        length = len(result[key])
        num = random.randint(0, length - 1)
        test_one_paths.append(result[key][num])
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    if record == 1:
        with open(specific_dataset_path, "w", encoding="utf-8") as file:
            for line in specify_data:
                file.write(line)
                file.write("\n")
            file.close()

    event_embedding_path = root + "events_compress_one_hot.txt"

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset = process_files_to_construct_trees(train_paths, event_embedding_path)
    val_dataset = process_files_to_construct_trees(val_paths, event_embedding_path)
    test_dataset = process_files_to_construct_trees(test_paths, event_embedding_path)
    test_one_dataset = process_files_to_construct_trees(test_one_paths, event_embedding_path)

    return train_dataset, val_dataset, test_dataset, test_one_dataset

def read_exist_substructure_to_datatxt(dataname, train_ratio, val_ratio, record, specify_number):
    result = {}
    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    directory = f"/root/autodl-tmp/project/data/{dataname}"

    i = 0
    sub_result = []

    open_file_new(directory, result)

    for key in result.keys():
        for path in result[key]:
            sub_result.append(str(i) + " : " + path)
        i = i + 1

    if record == 1:
        with open(graph_structure_path, "w", encoding="utf-8") as file:
            for line in sub_result:
                file.write(line)
                file.write("\n")
            file.close()

    key_list1 = list(result.keys())
    key_list = list(result.keys())
    random.shuffle(key_list)

    # key_list=key_list[:18]

    train_num = int(len(key_list) * train_ratio)
    val_num = int(len(key_list) * val_ratio)
    train_set_key = key_list[0: train_num]
    val_set_key = key_list[train_num: train_num + val_num]
    test_set_key = key_list[train_num + val_num:]

    specify_data = ["train dataset:"]

    for key in train_set_key:
        number = key_list1.index(key)
        train_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("val dataset:")
    for key in val_set_key:
        number = key_list1.index(key)
        val_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("test dataset:")
    for key in test_set_key:
        number = key_list1.index(key)
        test_paths += result[key]
        length = len(result[key])
        num = random.randint(0, length - 1)
        test_one_paths.append(result[key][num])
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    if record == 1:
        with open(specific_dataset_path, "w", encoding="utf-8") as file:
            for line in specify_data:
                file.write(line)
                file.write("\n")
            file.close()

    event_embedding_path = root + "events_compress_one_hot.txt"

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset = process_files_to_construct_trees(train_paths, event_embedding_path)
    val_dataset = process_files_to_construct_trees(val_paths, event_embedding_path)
    test_dataset = process_files_to_construct_trees(test_paths, event_embedding_path)
    test_one_dataset = process_files_to_construct_trees(test_one_paths, event_embedding_path)

    return train_dataset, val_dataset, test_dataset, test_one_dataset

def construct_tree_to_nx_layerfeatures_forum(tree, event_map, file_name_map, exception_map, feature_size):
    # 创建一个有向图
    G = nx.DiGraph()

    weight_list=[]
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        # print(node.name)
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                # 如果该文件名没有在map中，初始化为0
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding\
                                             +exception_map[exception.split("=")[1]]+ [float(cost.split("=")[1].split("m")[0])]
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            weight_list.append(weight)
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    G.add_node("root")
    G.nodes["root"]["event"] = ""
    G.nodes["root"]["feature"] = [float(0)] * (feature_size*3+1)
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["weight"] = max(weight_list) + 1
    G.nodes["root"]["cost"] = 0.0

    return G



def construct_tree_to_nx_layerfeatures_GAELog(tree, event_map, file_name_map, exception_map, feature_size,knowledge_graph):
    # 创建一个有向图
    G = nx.DiGraph()

    weight_list=[]
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        # print(node.name)
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                # 如果该文件名没有在map中，初始化为0
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            if child.name=="pub.developers.forum.facade.impl.UserApiServiceImpl:1" and child.name not in knowledge_graph.nodes:
                print()
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding + exception_map[exception.split("=")[1]]+ [float(cost.split("=")[1].split("m")[0])] + knowledge_graph.nodes[child.name]['feature'].tolist()
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            weight_list.append(weight)
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    G.add_node("root")
    G.nodes["root"]["event"] = ""
    G.nodes["root"]["feature"] = [float(0)] * (feature_size*4+1)
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["weight"] = max(weight_list) + 1
    G.nodes["root"]["cost"] = 0.0

    return G

def construct_tree_to_nx_layerfeatures_forum_layerfeats(tree, event_map, file_name_map, exception_map, feature_size):
    # 创建一个有向图
    G = nx.DiGraph()

    weight_list=[]
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                # 如果该文件名没有在map中，初始化为0
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding\
                                             +exception_map[exception.split("=")[1]]+ [float(cost.split("=")[1].split("m")[0])]
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            weight_list.append(weight)
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)

    # 计算图特征并添加到每个节点
    out_degrees = G.out_degree()
    in_degrees = G.in_degree()
    pageranks = nx.pagerank(G)
    betweennesses = nx.betweenness_centrality(G)

    # 将有向图转换为无向图来计算最短路径长度
    G_undirected = G.to_undirected()
    shortest_paths = nx.shortest_path_length(G_undirected, source="root")

    for node in G.nodes():
        layer_feature = [
            # G.nodes[node]["weight"],  # 节点权重
            out_degrees[node],
            in_degrees[node],
            pageranks[node],
            shortest_paths.get(node, float('inf')),  # 如果没有路径，则使用无穷大表示
            betweennesses[node]
        ]
        G.nodes[node]["layer_feature"]=layer_feature
        # G.nodes[node]["layer_feature"]=[float(0)] * 5

    G.add_node("root")
    G.nodes["root"]["event"] = ""
    G.nodes["root"]["feature"] = [float(0)] * (feature_size*3+1)
    G.nodes["root"]["layer_feature"] = [float(0)] * 5
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["weight"] = max(weight_list) + 1
    G.nodes["root"]["cost"] = 0.0

    return G


"""用于halo数据集的读取——logGD寻找划分使用"""
def read_exist_substructure_to_datatxt_for_halo(dataname,dataset, train_ratio, val_ratio, record, specify_number):
    result = {}
    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设项目根目录GHLAD是当前文件的三级父目录
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # 构建dataset目录的绝对路径
    datatxt_root = os.path.join(project_root, "data")

    root = datatxt_root + f"/{dataname}/新"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    dataset_root = os.path.join(project_root, "dataset")

    # directory = dataset_root + f"/{dataname}"
    directory = dataset

    i = 0
    sub_result = []

    open_file_new(directory, result)

    for key in result.keys():
        for path in result[key]:
            sub_result.append(str(i) + " : " + path)
        i = i + 1

    if record == 1:
        with open(graph_structure_path, "w", encoding="utf-8") as file:
            for line in sub_result:
                file.write(line)
                file.write("\n")
            file.close()

    key_list1 = list(result.keys())
    key_list = list(result.keys())
    random.shuffle(key_list)

    # key_list=key_list[:18]

    train_num = int(len(key_list) * train_ratio)
    val_num = int(len(key_list) * val_ratio)
    train_set_key = key_list[0: train_num]
    val_set_key = key_list[train_num: train_num + val_num]
    test_set_key = key_list[train_num + val_num:]

    specify_data = ["train dataset:"]

    for key in train_set_key:
        number = key_list1.index(key)
        train_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("val dataset:")
    for key in val_set_key:
        number = key_list1.index(key)
        val_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("test dataset:")
    for key in test_set_key:
        number = key_list1.index(key)
        test_paths += result[key]
        length = len(result[key])
        num = random.randint(0, length - 1)
        test_one_paths.append(result[key][num])
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    if record == 1:
        with open(specific_dataset_path, "w", encoding="utf-8") as file:
            for line in specify_data:
                file.write(line)
                file.write("\n")
            file.close()

    root = datatxt_root + f"/{dataname}/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset=[]
    val_dataset=[]
    test_dataset=[]
    test_one_dataset=[]

    for train_path in train_paths:
        g=construct_graph_to_nx_with_feature(train_path, event_map, file_name_map, exception_map)
        train_dataset.append(g)
    for val_path in val_paths:
        g=construct_graph_to_nx_with_feature(val_path, event_map, file_name_map, exception_map)
        val_dataset.append(g)
    for test_path in test_paths:
        g=construct_graph_to_nx_with_feature(test_path, event_map, file_name_map, exception_map)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        g=construct_graph_to_nx_with_feature(test_one_path, event_map, file_name_map, exception_map)
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset



"""用于halo数据集包含layerfeats的读取"""
def search_specify_data_from_dataname_for_halo_layerfeats(dataname, specify_number):
    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    ID_paths = {}
    trainID = []
    valID = []
    testID = []

    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    read_graph_structure(ID_paths, graph_structure_path)

    read_specific_data(specific_dataset_path, trainID, valID, testID)

    for ID in trainID:
        train_paths += ID_paths[ID]
    for ID in valID:
        val_paths += ID_paths[ID]
    for ID in testID:
        test_paths += ID_paths[ID]
        length = len(ID_paths[ID])
        num = random.randint(0, length - 1)

        test_one_paths.append(ID_paths[ID][num])

    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []

    for train_path in train_paths:
        g=construct_graph_to_nx_with_feature_layerfeats(train_path, event_map, file_name_map, exception_map)
        train_dataset.append(g)
    for val_path in val_paths:
        g=construct_graph_to_nx_with_feature_layerfeats(val_path, event_map, file_name_map, exception_map)
        val_dataset.append(g)
    for test_path in test_paths:
        g=construct_graph_to_nx_with_feature_layerfeats(test_path, event_map, file_name_map, exception_map)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        g=construct_graph_to_nx_with_feature_layerfeats(test_one_path, event_map, file_name_map, exception_map)
        test_one_dataset.append(g)
    return train_dataset, val_dataset, test_dataset, test_one_dataset


def construct_graph_to_nx_with_feature(file_path, event_map, file_name_map, exception_map):
    anomalous_nodes = []  # anomalous nodes
    edges_with_props = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("traceID="):
                trace_id = line.strip().split("=")[1]
            if line.startswith("label="):
                ano_node = line.strip().split("=")[1]
                anomalous_nodes.append(ano_node)

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                edge = parts[0].split("<-")
                weight = int(parts[1])
                edge_info = [edge ,  parts[3], parts[4], parts[5],
                             1 if edge[0] in anomalous_nodes else 0, weight]  # edge, cost, event, exception
                edges_with_props.append(edge_info)

    # 创建一个图
    G = nx.DiGraph()
    feature_dim = []
    weight_list=[]

    for edge_info in edges_with_props:
        target, source = edge_info[0]

        cost = edge_info[1]
        event = edge_info[2]
        exception = edge_info[3]
        label = edge_info[4]
        weight = edge_info[5]

        if not G.has_edge(target, source):
            G.add_edge(target, source)
        target_feature = event_map[event.replace("event=", "", 1)] + list(file_name_map[target]) + \
                         exception_map[
                             exception.replace("exception=", "", 1)] + [
                             float(cost.split("=")[1].split("m")[0])]
        if not feature_dim:
            feature_dim.append(len(target_feature))
        G.nodes[target]["label"] = label
        G.nodes[target]["name"] = target
        G.nodes[target]["weight"] = weight
        G.edges[(target, source)]["weight"] = weight
        weight_list.append(weight)

        if G.nodes[target].get('exception', None) is None:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True
            else:
                G.nodes[target]["exception"] = False
        else:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True

        feature_value = G.nodes[target].get('feature', None)
        if feature_value is not None:
            feature_value = [x + y for x, y in zip(target_feature, feature_value)]
            G.nodes[target]["feature"] = feature_value
        else:
            G.nodes[target]["feature"] = target_feature
    root_path = []
    for i, node in enumerate(list(G.nodes())):
        if node != "root":
            G.nodes[node]["call_paths"] = [i]
            root_path.append(i)

    """add root node attributes"""
    G.nodes["root"]["feature"] = [float(0)] * feature_dim[0]
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["exception"] = False
    G.nodes["root"]["name"] = "root"
    G.nodes["root"]["weight"] = max(weight_list)+1
    G.nodes["root"]["call_paths"] = root_path
    return G

"""用于halo数据集的layerfeats读取方法"""
def construct_graph_to_nx_with_feature_layerfeats(file_path, event_map, file_name_map, exception_map):
    anomalous_nodes = []  # anomalous nodes
    edges_with_props = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("traceID="):
                trace_id = line.strip().split("=")[1]
            if line.startswith("label="):
                ano_node = line.strip().split("=")[1]
                anomalous_nodes.append(ano_node)

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                edge = parts[0].split("<-")
                weight = int(parts[1])
                edge_info = [edge ,  parts[3], parts[4], parts[5],
                             1 if edge[0] in anomalous_nodes else 0, weight]  # edge, cost, event, exception
                edges_with_props.append(edge_info)

    # 创建一个图
    G = nx.DiGraph()
    feature_dim = []
    weight_list=[]

    for edge_info in edges_with_props:
        target, source = edge_info[0]

        cost = edge_info[1]
        event = edge_info[2]
        exception = edge_info[3]
        label = edge_info[4]
        weight = edge_info[5]

        if not G.has_edge(target, source):
            G.add_edge(target, source)
        target_feature = event_map[event.replace("event=", "", 1)] + list(file_name_map[target]) + \
                         exception_map[
                             exception.replace("exception=", "", 1)] + [
                             float(cost.split("=")[1].split("m")[0])]
        if not feature_dim:
            feature_dim.append(len(target_feature))
        G.nodes[target]["label"] = label
        G.nodes[target]["name"] = target
        G.nodes[target]["weight"] = weight
        G.edges[(target, source)]["weight"] = weight
        weight_list.append(weight)

        if G.nodes[target].get('exception', None) is None:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True
            else:
                G.nodes[target]["exception"] = False
        else:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True

        feature_value = G.nodes[target].get('feature', None)
        if feature_value is not None:
            feature_value = [x + y for x, y in zip(target_feature, feature_value)]
            G.nodes[target]["feature"] = feature_value
        else:
            G.nodes[target]["feature"] = target_feature
    root_path = []
    for i, node in enumerate(list(G.nodes())):
        if node != "root":
            G.nodes[node]["call_paths"] = [i]
            root_path.append(i)

    # 计算图特征并添加到每个节点
    out_degrees = G.out_degree()
    in_degrees = G.in_degree()
    pageranks = nx.pagerank(G)
    betweennesses = nx.betweenness_centrality(G)

    # 将有向图转换为无向图来计算最短路径长度
    G_undirected = G.to_undirected()
    shortest_paths = nx.shortest_path_length(G_undirected, source="root")

    for node in G.nodes():
        layer_feature = [
            # G.nodes[node]["weight"],  # 节点权重
            out_degrees[node],
            in_degrees[node],
            pageranks[node],
            shortest_paths.get(node, float('inf')),  # 如果没有路径，则使用无穷大表示
            betweennesses[node]
        ]
        G.nodes[node]["layer_feature"] = layer_feature

    """add root node attributes"""
    G.nodes["root"]["feature"] = [float(0)] * feature_dim[0]
    G.nodes["root"]["layer_feature"] = [float(0)] * 5
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["exception"] = False
    G.nodes["root"]["name"] = "root"
    G.nodes["root"]["weight"] = max(weight_list)+1
    G.nodes["root"]["call_paths"] = root_path
    return G

def search_specify_data_from_dataname(dataname, specify_number):
    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    ID_paths = {}
    trainID = []
    valID = []
    testID = []

    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    event_embedding_path = root + "events_compress_one_hot.txt"

    read_graph_structure(ID_paths, graph_structure_path)

    read_specific_data(specific_dataset_path, trainID, valID, testID)

    for ID in trainID:
        train_paths += ID_paths[ID]
    for ID in valID:
        val_paths += ID_paths[ID]
    for ID in testID:
        test_paths += ID_paths[ID]
        length = len(ID_paths[ID])
        num = random.randint(0, length - 1)

        test_one_paths.append(ID_paths[ID][num])

    train_data = process_files_to_construct_trees(train_paths, event_embedding_path)
    val_data = process_files_to_construct_trees(val_paths, event_embedding_path)
    test_data = process_files_to_construct_trees(test_paths, event_embedding_path)
    test_one_data = process_files_to_construct_trees(test_one_paths, event_embedding_path)

    return train_data, val_data, test_data, test_one_data


"""用于forum数据集的读取"""
def search_specify_data_from_dataname_for_forum_layerfeats(dataname, specify_number):
    # root = f"/root/autodl-tmp/project/data/{dataname}_data/新划分/"
    root = f"/root/autodl-tmp/project/data_2/{dataname}_data/新"

    # graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    # ID_paths = {}
    # trainID = []
    # valID = []
    # testID = []

    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)


    # read_graph_structure(ID_paths, graph_structure_path)

    read_specific_data_forum(specific_dataset_path, train_paths, val_paths, test_paths,test_one_paths)

    print(len(train_paths))
    print(len(val_paths))
    print(len(test_paths))
    print(len(test_one_paths))


    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []

    for train_path in train_paths:
        root = build_tree_from_txt(train_path)
        g = construct_tree_to_nx_layerfeatures_forum_layerfeats(root, event_map, file_name_map, exception_map,feature_size)
        train_dataset.append(g)
    for val_path in val_paths:
        root = build_tree_from_txt(val_path)
        g = construct_tree_to_nx_layerfeatures_forum_layerfeats(root, event_map, file_name_map, exception_map,feature_size)
        val_dataset.append(g)
    for test_path in test_paths:
        root = build_tree_from_txt(test_path)
        g = construct_tree_to_nx_layerfeatures_forum_layerfeats(root, event_map, file_name_map, exception_map,feature_size)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        root = build_tree_from_txt(test_one_path)
        g = construct_tree_to_nx_layerfeatures_forum_layerfeats(root, event_map, file_name_map, exception_map,feature_size)
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset


def search_specify_data_from_dataname_for_forum(dataname, specify_number):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设项目根目录GHLAD是当前文件的三级父目录
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # 构建dataset目录的绝对路径
    dataset_root = os.path.join(project_root, "data")

    # root = f"/root/autodl-tmp/project/data/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data_2/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data_4/{dataname}_data/新"

    root = dataset_root + f"/{dataname}/新"

    # root = f"/root/autodl-tmp/project/data/{dataname}_data/新划分/"
    # root = f"/root/autodl-tmp/project/data/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data/{dataname}_data/"
    # root = f"/root/autodl-tmp/project/data_2/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data_3/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data_4/{dataname}_data/新"

    # graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = dataset_root + f"/{dataname}/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    read_specific_data_forum(specific_dataset_path, train_paths, val_paths, test_paths, test_one_paths)

    print(len(train_paths))
    print(len(val_paths))
    print(len(test_paths))
    print(len(test_one_paths))

    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []

    for train_path in train_paths:
        root = build_tree_from_txt(train_path)

        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        train_dataset.append(g)
    for val_path in val_paths:
        root = build_tree_from_txt(val_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        val_dataset.append(g)
    for test_path in test_paths:
        root = build_tree_from_txt(test_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        root = build_tree_from_txt(test_one_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset

def parse_log_file(log_file_path):
    """
    解析日志文件，提取服务实体、事件实体、异常实体等信息。
    """
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # 初始化集合，用于存储提取的实体
    service_entities = set()
    event_entities = set()
    exception_entities = set()
    label = None

    # 解析日志文件
    for line in lines:
        line = line.strip()
        if line.startswith("traceID="):
            continue
        elif line.startswith("label="):
            label = line.split('=')[1]
        elif line.startswith("exception="):
            exception = line.split('=')[1]
            if exception != "perfect":
                exception_entities.add(exception)
        elif line.startswith("network[son<-parent]="):
            # 解析调用链信息
            calls = line.split('=')[1].split('\n')
            for call in calls:
                if not call.strip():
                    continue
                parts = call.split(',')
                src, dst = parts[0].split('<-')
                src = src.strip()
                dst = dst.strip()
                service_entities.add(src)
                service_entities.add(dst)
                event = parts[3].split('=')[1]
                event_entities.add(event)
                exception = parts[4].split('=')[1]
                if exception != "null":
                    exception_entities.add(exception)

    return service_entities, event_entities, exception_entities, label



"""用于forum数据集的读取——logGD寻找划分使用"""
def read_exist_substructure_to_datatxt_for_forum(dataname,dataset, train_ratio, val_ratio, record, specify_number):
    result = {}
    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设项目根目录GHLAD是当前文件的三级父目录
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # 构建dataset目录的绝对路径
    datatxt_root = os.path.join(project_root, "data")

    root = datatxt_root + f"/{dataname}/新"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    dataset_root = os.path.join(project_root, "dataset")

    # directory = dataset

    directory = dataset
    # root = f"/root/autodl-tmp/project/data_4/{dataname}_data/新/新"
    #
    # graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"
    #
    # specific_dataset_path = root + f"specific_dataset{specify_number}.txt"
    #
    # directory = f"/root/autodl-tmp/project/data/{dataname}"
    # directory = f"/root/autodl-tmp/project/data/{dataname}/负样本/手动注错/陈威负样本/1-陈威日志测试 -call_change/abstra/文章审核不通过"

    i = 0
    sub_result = []

    open_file(directory, result)

    for key in result.keys():
        for path in result[key]:
            sub_result.append(str(i) + " : " + path)
        i = i + 1

    if record == 1:
        with open(graph_structure_path, "w", encoding="utf-8") as file:
            for line in sub_result:
                file.write(line)
                file.write("\n")
            file.close()

    key_list1 = list(result.keys())
    key_list = list(result.keys())
    random.shuffle(key_list)

    # key_list=key_list[:18]

    train_num = int(len(key_list) * train_ratio)
    val_num = int(len(key_list) * val_ratio)
    train_set_key = key_list[0: train_num]
    val_set_key = key_list[train_num: train_num + val_num]
    test_set_key = key_list[train_num + val_num:]

    # train_set_key = key_list[0: ]
    # val_set_key = key_list[0: ]
    # test_set_key = key_list[0: ]

    specify_data = ["train dataset:"]

    for key in train_set_key:
        number = key_list1.index(key)
        train_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("val dataset:")
    for key in val_set_key:
        number = key_list1.index(key)
        val_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("test dataset:")
    for key in test_set_key:
        number = key_list1.index(key)
        test_paths += result[key]
        length = len(result[key])
        num = random.randint(0, length - 1)
        test_one_paths.append(result[key][num])
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    if record == 1:
        with open(specific_dataset_path, "w", encoding="utf-8") as file:
            for line in specify_data:
                file.write(line)
                file.write("\n")
            file.close()

    root = datatxt_root + f"/{dataname}/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []

    for train_path in train_paths:
        root = build_tree_from_txt(train_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        # print(g.nodes)
        train_dataset.append(g)
    for val_path in val_paths:
        root = build_tree_from_txt(val_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        val_dataset.append(g)
    for test_path in test_paths:
        root = build_tree_from_txt(test_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        root = build_tree_from_txt(test_one_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset

def add_edge_attr_to_graph(G):
    """
    为图添加 edge_attr（边权重），并计算图标签
    """

    # 创建一个字典，记录每条边的出现次数
    edge_counts = {}

    # 遍历图中的所有边，统计出现次数
    for u, v in G.edges():
        if (u, v) in edge_counts:
            edge_counts[(u, v)] += 1
        else:
            edge_counts[(u, v)] = 1

    # 将边的出现次数作为 edge_attr 添加到图中
    for u, v in G.edges():
        G[u][v]["edge_attr"] = edge_counts[(u, v)]


    # # 遍历图中的所有边，添加 edge_attr
    # for u, v, data in G.edges(data=True):
    #     if "weight" in data:
    #         data["edge_attr"] = data["weight"]  # 将 weight 复制到 edge_attr
    #     else:
    #         data["edge_attr"] = 1  # 如果边没有 weight，则默认 edge_attr 为 1

    # 计算图标签
    node_labels = [G.nodes[node]['label'] for node in G.nodes]
    if all(label == 0 for label in node_labels):
        G.graph["label"] = 0  # 所有节点标签为 0，图标签为 1（正常）
    else:
        G.graph["label"] = 1  # 存在至少一个节点标签为 1，图标签为 0（异常）

    return G


def search_specify_data_from_dataname_for_Logs2(dataname, specify_number):
    root = f"/root/autodl-tmp/project/data/{dataname}_data/新划分/"
    # root = f"/root/autodl-tmp/project/data/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data/{dataname}_data/"
    # root = f"/root/autodl-tmp/project/data_2/{dataname}_data/新"

    # graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    read_specific_data_forum(specific_dataset_path, train_paths, val_paths, test_paths, test_one_paths)

    print(len(train_paths))
    print(len(val_paths))
    print(len(test_paths))
    print(len(test_one_paths))

    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []



    for train_path in train_paths:
        root = build_tree_from_txt(train_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        g = add_edge_attr_to_graph(g)  # 为图添加 edge_attr
        train_dataset.append(g)
    for val_path in val_paths:
        root = build_tree_from_txt(val_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        g = add_edge_attr_to_graph(g)  # 为图添加 edge_attr
        val_dataset.append(g)
    for test_path in test_paths:
        root = build_tree_from_txt(test_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        g = add_edge_attr_to_graph(g)  # 为图添加 edge_attr
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        root = build_tree_from_txt(test_one_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map,feature_size)
        g = add_edge_attr_to_graph(g)  # 为图添加 edge_attr
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset




def read_exist_substructure_to_datatxt_for_Logs2(dataname, train_ratio, val_ratio, record, specify_number):
    result = {}
    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = f"/root/autodl-tmp/project/data_2/{dataname}_data/新"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    directory = f"/root/autodl-tmp/project/data/{dataname}"
    # directory = f"/root/autodl-tmp/project/data/{dataname}/负样本/手动注错"

    i = 0
    sub_result = []

    open_file(directory, result)

    for key in result.keys():
        for path in result[key]:
            sub_result.append(str(i) + " : " + path)
        i = i + 1

    if record == 1:
        with open(graph_structure_path, "w", encoding="utf-8") as file:
            for line in sub_result:
                file.write(line)
                file.write("\n")
            file.close()

    key_list1 = list(result.keys())
    key_list = list(result.keys())
    random.shuffle(key_list)

    # key_list=key_list[:18]

    train_num = int(len(key_list) * train_ratio)
    val_num = int(len(key_list) * val_ratio)
    train_set_key = key_list[0: train_num]
    val_set_key = key_list[train_num: train_num + val_num]
    test_set_key = key_list[train_num + val_num:]

    specify_data = ["train dataset:"]

    for key in train_set_key:
        number = key_list1.index(key)
        train_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("val dataset:")
    for key in val_set_key:
        number = key_list1.index(key)
        val_paths += result[key]
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    specify_data.append("test dataset:")
    for key in test_set_key:
        number = key_list1.index(key)
        test_paths += result[key]
        length = len(result[key])
        num = random.randint(0, length - 1)
        test_one_paths.append(result[key][num])
        for path in result[key]:
            specify_data.append(str(number) + " : " + path)
    if record == 1:
        with open(specific_dataset_path, "w", encoding="utf-8") as file:
            for line in specify_data:
                file.write(line)
                file.write("\n")
            file.close()

    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []

    for train_path in train_paths:
        root = build_tree_from_txt(train_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        train_dataset.append(g)
    for val_path in val_paths:
        root = build_tree_from_txt(val_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        val_dataset.append(g)
    for test_path in test_paths:
        root = build_tree_from_txt(test_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        root = build_tree_from_txt(test_one_path)
        g = construct_tree_to_nx_layerfeatures_forum(root, event_map, file_name_map, exception_map, feature_size)
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset

def construct_tree_to_nx_with_filemap_layerfeatures(tree, event_map, file_name_map,exception_map):
    # 创建一个有向图
    G = nx.DiGraph()

    # G.add_node("root")
    # G.nodes['root']['weight']=-1
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                # 如果该文件名没有在map中，初始化为0
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding+exception_map[exception.split("=")[1]]
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)

    # 计算图特征并添加到每个节点
    out_degrees = G.out_degree()
    in_degrees = G.in_degree()
    pageranks = nx.pagerank(G)
    betweennesses = nx.betweenness_centrality(G)

    # 将有向图转换为无向图来计算最短路径长度
    G_undirected = G.to_undirected()
    shortest_paths = nx.shortest_path_length(G_undirected, source="root")

    for node in G.nodes():
        layer_feature = [
            # G.nodes[node]["weight"],  # 节点权重
            out_degrees[node],
            in_degrees[node],
            pageranks[node],
            shortest_paths.get(node, float('inf')),  # 如果没有路径，则使用无穷大表示
            betweennesses[node]
        ]
        G.nodes[node]["layer_feature"]=layer_feature

    return G

import os
def search_specify_data_from_dataname_for_halo(dataname, specify_number):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设项目根目录GHLAD是当前文件的三级父目录
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # 构建dataset目录的绝对路径
    dataset_root = os.path.join(project_root, "data")

    # root = f"/root/autodl-tmp/project/data/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data_2/{dataname}_data/新"
    # root = f"/root/autodl-tmp/project/data_4/{dataname}_data/新"


    root = dataset_root + f"/{dataname}_data/新"

    # graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    ID_paths = {}
    # train = []
    # valID = []
    # testID = []

    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = dataset_root + f"/{dataname}_data/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    # read_graph_structure(ID_paths, graph_structure_path)

    read_specific_data(specific_dataset_path, train_paths, val_paths, test_paths)

    test_one_paths=test_paths

    # for ID in trainID:
    #     train_paths += ID_paths[ID]
    # for ID in valID:
    #     val_paths += ID_paths[ID]
    # for ID in testID:
    #     test_paths += ID_paths[ID]
    #     length = len(ID_paths[ID])
    #     num = random.randint(0, length - 1)
    #
    #     test_one_paths.append(ID_paths[ID][num])

    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    for train_path in train_paths:
        g = construct_graph_to_nx_with_feature(train_path, event_map, file_name_map, exception_map)
        train_dataset.append(g)
    for val_path in val_paths:
        g = construct_graph_to_nx_with_feature(val_path, event_map, file_name_map, exception_map)
        val_dataset.append(g)
    for test_path in test_paths:
        g = construct_graph_to_nx_with_feature(test_path, event_map, file_name_map, exception_map)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        g = construct_graph_to_nx_with_feature(test_one_path, event_map, file_name_map, exception_map)
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset

def search_specify_data_from_dataname_for_halo_withlayerfeats(dataname, specify_number):
    root = f"/root/autodl-tmp/project/data/{dataname}_data/新划分/"

    # graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    ID_paths = {}
    # train = []
    # valID = []
    # testID = []

    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = f"/root/autodl-tmp/project/data/{dataname}_data/"

    event_embedding_path = root + r"events_compress_one_hot.txt"
    file_embedding_path = root + r"file_name_one_hot.txt"
    exception_embedding_path = root + r"exception_list.txt"

    if dataname == "forum":
        feature_size = 82
    elif dataname == "novel":
        feature_size = 166
    elif dataname == "halo":
        feature_size = 161

    event_map = read_one_hot(event_embedding_path)
    file_name_map = read_file_one_hot_new(file_embedding_path, feature_size)
    exception_map = read_exception_one_hot_new(exception_embedding_path, feature_size)

    # read_graph_structure(ID_paths, graph_structure_path)

    read_specific_data(specific_dataset_path, train_paths, val_paths, test_paths)

    test_one_paths=test_paths

    # for ID in trainID:
    #     train_paths += ID_paths[ID]
    # for ID in valID:
    #     val_paths += ID_paths[ID]
    # for ID in testID:
    #     test_paths += ID_paths[ID]
    #     length = len(ID_paths[ID])
    #     num = random.randint(0, length - 1)
    #
    #     test_one_paths.append(ID_paths[ID][num])

    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_one_dataset = []

    for train_path in train_paths:
        g = construct_graph_to_nx_with_feature_layerfeats(train_path, event_map, file_name_map, exception_map)
        train_dataset.append(g)
    for val_path in val_paths:
        g = construct_graph_to_nx_with_feature_layerfeats(val_path, event_map, file_name_map, exception_map)
        val_dataset.append(g)
    for test_path in test_paths:
        g = construct_graph_to_nx_with_feature_layerfeats(test_path, event_map, file_name_map, exception_map)
        test_dataset.append(g)
    for test_one_path in test_one_paths:
        g = construct_graph_to_nx_with_feature_layerfeats(test_one_path, event_map, file_name_map, exception_map)
        test_one_dataset.append(g)

    return train_dataset, val_dataset, test_dataset, test_one_dataset