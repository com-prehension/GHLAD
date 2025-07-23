import networkx as nx
import matplotlib.pyplot as plt

def find_high_order_subgraph(G, order):
    """
    找到有向图G的特定阶数的高阶关系子图。

    参数:
    G -- NetworkX有向图
    order -- 寻找的高阶关系的阶数

    返回:
    G_high_order -- 包含高阶关系的子图
    """
    G_high_order = nx.DiGraph()

    for node in G.nodes():
        G_high_order.add_node(node)

    # 遍历图中的所有节点
    for node in G.nodes():
        # 使用single_source_shortest_path_length找到从当前节点到其他所有节点的最短路径长度
        lengths = nx.single_source_shortest_path_length(G, node)

        # 遍历长度，添加长度等于order的节点对
        for target, length in lengths.items():
            if length == order:  # 长度等于order的路径
                # 将高阶关系添加到高阶关系子图中
                G_high_order.add_edge(node, target)

    return G_high_order

def draw_tree(G):
    # 绘制树形布局
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", edge_color="gray")

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # 绘制边权重
    # edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # 设置图形样式
    plt.axis("off")
    plt.tight_layout()

    # 显示图形
    plt.show()

if __name__ == '__main__':
    # 创建一个有向图
    G = nx.DiGraph()
    # 添加一些边
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4), (4, 2), (2, 5), (5, 6), (6, 3)])

    draw_tree(G)

    # 定义寻找的高阶关系的阶数
    order = 2

    # 获取特定阶数的高阶关系子图
    G_high_order = find_high_order_subgraph(G, order)

    draw_tree(G_high_order)

    # 打印原始图的边
    print("Edges in the original graph G:", G.edges())

    # 打印特定阶数的高阶关系子图的边
    print(f"Edges in the {order}-order subgraph G_high_order:", G_high_order.edges())