import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import networkx as nx
import matplotlib.pyplot as plt
# from 提取语义 import get_event_embedding_from_file
# from node2vec_args import args_parser
# from node2vec_main import traverse_graphs_handle
# from functions.Seacher_high_Graph import find_high_order_subgraph

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
    def __init__(self, attributes1, labels1, node_names1, timestamps1, event_map):
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
            self.events.append(event_map[event.split("=")[1]])
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

def draw_tree(G):
    # 绘制树形布局
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", edge_color="gray")

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # 绘制边权重
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # 设置图形样式
    plt.axis("off")
    plt.tight_layout()

    # 显示图形
    plt.show()


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

            # G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding + exception_map[exception.split("=")[1]]
            G.nodes[child.name]["feature"] = event_map[event.replace("event=","")] + file_name_embedding + exception_map[exception.replace("exception=","")]
            G.nodes[child.name]["event"] = event.split("=")[1]
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    return G


def construct_tree_to_nx_with_filemap_and_high_subgraph(tree, event_map,file_name_map,exception_map):
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

    order=3
    # G_high_order = find_high_order_subgraph(G,order)
    G_high_order=1

    return G,G_high_order

def construct_tree_to_nx(tree, event_map):
    # 创建一个有向图
    G = nx.DiGraph()
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]

            # G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + [float(cost.replace("ms","").split("=")[1])]
            G.nodes[child.name]["feature"] = event_map[event.replace("event=","")] + [float(cost.replace("ms","").split("=")[1])]

            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    return G

def construct_layered_graph(tree, event_map):
    # 创建一个有向图
    G = nx.DiGraph()
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]

            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + [float(cost.replace("ms","").split("=")[1])]

            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    return G

def construct_tree_to_nx_with_filemap_position(tree, event_map):
    # 创建一个有向图
    G = nx.DiGraph()
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]

            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + [float(cost.replace("ms","").split("=")[1])]

            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    return G

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

def construct_tree_to_nx_with_filemap_layerfeatures_new(tree, event_map, file_name_map,exception_map):
    # 创建一个有向图
    G = nx.DiGraph()
    # 递归遍历树，并将节点和边添加到图中
    G.add_node("root")
    G.nodes["root"]['weight']=-1
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

    # 假设 G 是有向图
    G_undirected = G.to_undirected()  # 将其转换为无向图

    # 计算图特征并添加到每个节点
    #出度
    out_degrees = dict(G.out_degree())
    #入度
    in_degrees = dict(G.in_degree())
    #PageRank
    pageranks = nx.pagerank(G)
    #中介中心性
    # betweennesses = nx.betweenness_centrality(G)  #慢

    clustering_coeffs = nx.clustering(G)  # 局部聚类系数
    # eigenvector_centralities = nx.eigenvector_centrality(G)  # 特征向量中心性
    # katz_centralities = nx.katz_centrality(G)  # Katz 中心性
    # triangles = nx.triangles(G_undirected)  # 三角形计数
    # current_flow_betweennesses = nx.current_flow_betweenness_centrality(G_undirected)  # 介数中心性（流量） 慢

    # # 将有向图转换为无向图来计算最短路径长度
    # G_undirected = G.to_undirected()
    shortest_paths = dict(nx.single_source_shortest_path_length(G_undirected, source="root"))

    # 计算每个节点的深度（到根节点的距离）
    node_depths = {node: shortest_paths.get(node, float('inf')) for node in G.nodes()}

    # 计算每个节点的后继节点数量
    successor_counts = {node: len(list(nx.descendants(G, node))) for node in G.nodes()}

    # 为每个节点添加特征
    for node in G.nodes():
        layer_feature = [
            G.nodes[node]["weight"],  # 节点权重
            out_degrees.get(node, 0),  # 出度
            in_degrees.get(node, 0),  # 入度
            pageranks.get(node, 0.0),  # PageRank
            node_depths.get(node, float('inf')),  # 节点深度
            # betweennesses.get(node, 0.0),  # 中介中心性
            clustering_coeffs.get(node, 0.0),  # 聚类系数
            # eigenvector_centralities.get(node, 0.0),  # 特征向量中心性
            # katz_centralities.get(node, 0.0),  # Katz 中心性
            # triangles.get(node, 0),  # 三角形计数
            # current_flow_betweennesses.get(node, 0.0),  # 介数中心性（流量）
            successor_counts.get(node, 0)  # 后继节点数量
        ]
        G.nodes[node]["layer_feature"] = layer_feature

    return G


def construct_tree_to_nx_with_filemap_layerfeatures_new_edges_new(tree, event_map, file_name_map,exception_map):
    # 创建一个有向图
    G = nx.DiGraph()
    # 递归遍历树，并将节点和边添加到图中
    G.add_node("root")
    G.nodes["root"]['weight']=-1

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
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding + exception_map[
                exception.split("=")[1]]
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)

    def new_edges(tree):
        # 首先判断当前节点是否只与根节点root有一条边
        for node, weight, props, label in tree.children:
            # 检查节点的入度和出度
            in_edges = list(G.in_edges(node.name))
            out_edges = list(G.out_edges(node.name))
            if len(out_edges) == 1 and out_edges[0][1] == "root" and len(in_edges) == 0:
                # 获取同层次兄弟节点
                for child, weight, props, label in tree.children:
                    if child != node:
                        # 为孤立节点与其兄弟节点建立双向边
                        G.add_edge(node.name, child.name)
                        G.add_edge(child.name, node.name)
    new_edges(tree)

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
            # G.nodes[node]['weight'],
            out_degrees[node],
            in_degrees[node],
            pageranks[node],
            shortest_paths.get(node, float('inf')),  # 如果没有路径，则使用无穷大表示
            betweennesses[node]
        ]
        G.nodes[node]["layer_feature"]=layer_feature

    return G

def construct_tree_to_nx_with_HimNet(tree, event_map):
    # 创建一个有向图
    G = nx.DiGraph()
    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]

            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + [float(cost.replace("ms","").split("=")[1])]

            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    return G

def construct_graph_with_only_brother_connected_new(tree, event_map, file_name_map,exception_map):
    # 创建一个有向图,原有的调用树的边全部删除，最后加上树中兄弟按时序相连的图
    G = nx.DiGraph()

    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        current_bro = []
        bro_weight_dic = {}
        for child, weight, props, label in node.children:
            current_bro.append(child)
            bro_weight_dic[child] = weight
        for bro in current_bro:
            for bro1 in current_bro:
                if bro1 != bro and bro_weight_dic[bro] > bro_weight_dic[bro1]:
                    # 在兄弟中，时序靠后的指向时序靠前的
                    G.add_edge(bro.name, bro1.name)
                    G.edges[(bro.name, bro1.name)]["weight"] = -1

        for child, weight, props, label in node.children:
            G.add_node(child.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                # 如果该文件名没有在map中，初始化为0
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding + exception_map[exception.split("=")[1]]
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            traverse(child)

    G.add_node("root")
    traverse(tree)
    return G

def construct_graph_with_only_brother_connected(tree, event_map, file_name_map):
    # 创建一个有向图,原有的调用树的边全部删除，最后加上树中兄弟按时序相连的图
    G = nx.DiGraph()

    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        current_bro = []
        bro_weight_dic = {}
        for child, weight, props, label in node.children:
            current_bro.append(child)
            bro_weight_dic[child] = weight
        for bro in current_bro:
            for bro1 in current_bro:
                if bro1 != bro and bro_weight_dic[bro] > bro_weight_dic[bro1]:
                    # 在兄弟中，时序靠后的指向时序靠前的
                    G.add_edge(bro.name, bro1.name)
                    G.edges[(bro.name, bro1.name)]["weight"] = -1

        for child, weight, props, label in node.children:
            G.add_node(child.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                # 如果该文件名没有在map中，初始化为0
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            traverse(child)

    G.add_node("root")
    traverse(tree)
    return G

def construct_graph_with_new_edges_connected(tree,root):
    # 创建一个有向图,原有的调用树的边全部删除，最后加上树中兄弟按时序相连的图
    G = tree

    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):

        # 首先判断当前节点是否只与根节点root有一条边
        for node, weight, props, label in node.children:
            # 检查节点的入度和出度
            in_edges = list(G.in_edges(node.name))
            out_edges = list(G.out_edges(node.name))


            if len(out_edges) == 1 and out_edges[0][1] == "root" and len(in_edges) == 0:
                # 获取同层次兄弟节点
                for child, weight, props, label in node.children:
                    if child != node:
                        # 为孤立节点与其兄弟节点建立双向边
                        G.add_edge(node.name, child.name)
                        G.add_edge(child.name, node.name)

    # G.add_node("root")
    traverse(root)
    return G


def construct_tree_to_nx_with_brother_connected(tree, event_map, file_name_map):
    # 创建一个有向图
    G = nx.DiGraph()

    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        current_bro = []
        bro_weight_dic = {}
        for child, weight, props, label in node.children:
            current_bro.append(child)
            bro_weight_dic[child] = weight
        for bro in current_bro:
            for bro1 in current_bro:
                if bro1 != bro and bro_weight_dic[bro] > bro_weight_dic[bro1]:
                    # 在兄弟中，时序靠后的指向时序靠前的
                    G.add_edge(bro.name, bro1.name)
                    G.edges[(bro.name, bro1.name)]["weight"] = -1

        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                # 如果该文件名没有在map中，初始化为0
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            G.nodes[child.name]["feature"] = event_map[event.split("=")[1]] + file_name_embedding
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.edges[(child.name, node.name)]["weight"] = weight
            traverse(child)

    traverse(tree)
    return G


"""
遍历文件夹，构建树
"""


def process_files_to_construct_tree_chains(file_directory, event_map):
    all_chains = []
    for root, dirs, files in os.walk(file_directory):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                tree = build_tree_from_txt(file_path)
                attributes = []
                labels = []
                node_names = []
                timestamps = []
                chains = []
                # print(root,"//",file_name)
                traverse_tree(tree, attributes, labels, node_names, timestamps, chains, event_map)
                all_chains.append(chains)
    return all_chains


def process_trees_to_construct_tree_chains(trees, event_map):
    all_chains = []
    for tree1 in trees:
        attributes = []
        labels = []
        node_names = []
        timestamps = []
        chains = []
        traverse_tree(tree1, attributes, labels, node_names, timestamps, chains, event_map)
        all_chains.append(chains)

    return all_chains


if __name__ == "__main__":
    print("hello")
