import os
import networkx as nx

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

class Node:
    def __init__(self, trace_id, name):
        self.trace_id = trace_id
        self.name = name
        self.children = []


def build_tree_from_txt(file_path):
    root = Node("", "root")
    nodes = {"root:0": (root, -1, '', 0)}
    edge_info_tuples = []  # (target, source, weight)
    dup_node_count = {}  # key = node name, value = repeated number of the node
    anomalous_nodes = []  # anomalous nodes

    with open(file_path, "r") as file:
        lines = file.readlines()
        # print(file_path)
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("traceID="):
                trace_id = line.strip().split("=")[1]
                root.trace_id = trace_id
            if line.startswith("label="):
                ano_node = line.strip().split("=")[1]
                anomalous_nodes.append(ano_node)

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                edge = parts[0].split("<-")
                weight = int(parts[1])
                target_node_name = edge[0].strip()
                source_node_name = edge[1].strip()
                target_node_props = ",".join(parts[2:])

                # 将当前节点信息加入到节点信息列表中
                if target_node_name in dup_node_count.keys():
                    dup_count = dup_node_count[target_node_name]
                    dup_node_count[target_node_name] = dup_count + 1
                else:
                    dup_count = 0
                    dup_node_count[target_node_name] = 1

                # 记录边信息
                edge_info_tuples.append((target_node_name + ":" + str(dup_count), source_node_name, weight))
                if target_node_name in anomalous_nodes:  # 1代表异常， 0代表正常
                    node_label = 1
                else:
                    node_label = 0
                nodes[target_node_name + ":" + str(dup_count)] = (
                    Node(trace_id, target_node_name + ":" + str(dup_count)), weight, target_node_props,
                    node_label)  # 相同节点名字当作不同节点
    for target_node_name, source_node_name, weight in edge_info_tuples:
        target_node, t_weight, t_props, t_node_label = nodes[target_node_name]
        # print(target_node, ",", t_weight, ",", t_props, ",", t_node_label)
        src_nodes_candidate = []
        for k in nodes.keys():
            k1 = k.split(":")[0]
            if k1 == source_node_name:
                src_nodes_candidate.append(nodes[k])  # 寻找指定名字的父节点候选集
        src_nodes_candidate.sort(key=lambda t: t[1])  # 按照时序排序
        for src_node, src_weight, _, _ in src_nodes_candidate:
            if weight < src_weight or source_node_name == "root":  # 目标节点的时序小于源节点名字的时序的第一个节点
                src_node.children.append((target_node, weight, t_props, t_node_label))
                break

    return root


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
            # self.events.append(event_map[event.split("=")[1]])
            exc = exception.split("=")[1]
            if exc == "null":
                self.exceptions.append(float(0))
            else:
                # self.exceptions.append(float(exc))
                self.exceptions.append(exc)

        self.labels = [x for x in labels1]
        self.labels.reverse()
        self.names = [x for x in node_names1]
        self.names.reverse()
        self.timestamps = [x for x in timestamps1]
        self.timestamps.reverse()


def traverse_tree(node, attributes, labels, node_names, timestamps, chains, event_map):
    if not node.children:
        chains.append(CallChain(attributes, labels, node_names, timestamps, event_map))
    for child, weight, props, label in node.children:
        attributes.append(props)
        labels.append(label)
        node_names.append(child.name)
        timestamps.append(float(weight))
        traverse_tree(child, attributes, labels, node_names, timestamps, chains, event_map)
        labels.pop()
        node_names.pop()
        attributes.pop()
        timestamps.pop()


def traverse_tree_new(node, attributes, labels, node_names, timestamps, chains):
    if not node.children:
        chains.append(CallChain(attributes, labels, node_names, timestamps))
    for child, weight, props, label in node.children:
        attributes.append(props)
        labels.append(label)
        node_names.append(child.name)
        timestamps.append(float(weight))
        traverse_tree_new(child, attributes, labels, node_names, timestamps, chains)
        labels.pop()
        node_names.pop()
        attributes.pop()
        timestamps.pop()

def construct_to_nx(tree,event_map):
    # 创建一个有向图
    G = nx.DiGraph()

    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            # print(G.nodes)
            event=str(event).replace("event=","")
            G.nodes[child.name]["event"] = event_map[event]
            G.nodes[child.name]["weight"] = weight
            G.nodes[child.name]["label"] = label
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    return G

def construct_tree_to_nx_with_filemap(tree, event_map,file_name_map,exception_map):
    # 创建一个有向图
    G = nx.DiGraph()

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

            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding + exception_map[exception.split("=")[1]]
            G.nodes[child.name]["event"] = event.split("=")[1]
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)

    return G

def process_trees_to_construct_tree_chains(tree):

    attributes = []
    labels = []
    node_names = []
    timestamps = []
    chains = []
    traverse_tree_new(tree, attributes, labels, node_names, timestamps, chains)

    return chains

import torch
def deal_tree_to_chain(tree,graph):
    if 'root' in graph.nodes:
        graph.remove_node('root')

    all_nodes = graph.nodes

    # 建立节点索引映射字典
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}

    tree_chains = process_trees_to_construct_tree_chains(tree)

    chains=[]
    for call_chain in tree_chains:
        call_chain_new=call_chain.names
        # call_chain_new=call_chain_new+["root"]
        # 将调用链条中的节点替换成它在图中节点列表中的索引
        # indexed_call_chains = [node_to_index[node] for node in call_chain.names]
        indexed_call_chains = [node_to_index[node] for node in call_chain_new]
        indexed_call_chains.reverse()

        chains.append(indexed_call_chains)
        # # 输出结果
        # print(indexed_call_chains)
        # print(call_chain.names)
    # 找到最大长度
    # max_length = max(len(chain) for chain in chains)
    max_length = 15
    # print(max_length)

    # 填充序列
    padded_call_chains = [chain + [-1] * (max_length - len(chain)) for chain in chains]

    # 转换为 torch.Tensor
    tensor_call_chains = torch.tensor(padded_call_chains, dtype=torch.long)

    # 归一化处理：例如，将填充值归一化为 0
    normalized_call_chains = torch.where(tensor_call_chains == -1, torch.tensor(-1), tensor_call_chains)

    return normalized_call_chains
if __name__ == "__main__":
    print("hello")

