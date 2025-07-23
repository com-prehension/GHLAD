import os
import networkx as nx
import matplotlib.pyplot as plt
# import pygraphviz
import re
import numpy as np
import random

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []


def build_tree_from_txt(file_path):
    root = Node("root")
    nodes = {"root:0": (root, -1, '', 0)}
    edge_info_tuples = []  # (target, source, weight)
    dup_node_count = {}  # key = node name, value = repeated number of the node
    anomalous_nodes = []  # anomalous nodes

    with open(file_path, "r") as file:
        lines = file.readlines()
        print(lines)
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
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
                    Node(target_node_name + ":" + str(dup_count)), weight, target_node_props,
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


def traverse_tree(node, stack,result,events):
    if not node.children:
        print(stack)
        # deal_event(stack, result)
        # deal_event(stack,result,events)
        # deal_file(statck,result)
    for child, weight, props, label in node.children:
        stack.append(props)
        traverse_tree(child, stack,result,events)
        stack.pop()



def draw_tree(tree):
    # 创建一个有向图
    G = nx.DiGraph()

    # 递归遍历树，并将节点和边添加到图中
    def traverse(node):
        for child, weight, props, label in node.children:
            G.add_edge(node.name, child.name, weight=weight, props=props, label=label)
            traverse(child)

    traverse(tree)

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


'''遍历文件夹，构建树'''


def process_files(directorys):
    result={}
    events={}
    for directory in directorys:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(root, file_name)
                    tree = build_tree_from_txt(file_path)
                    stack = [0]
                    traverse_tree(tree, stack,result,events)
    # print(result)
    # storage(result,events)


def deal_event(stack,result):
    str_result = ['0']
    for idx, s in enumerate(stack):
        # print(idx)
        if s != 0:
            s1 = str(s)
            ret = re.search("event.*", s1).group().replace("event=", "")
            ret1 = ret.split(",except", 1)[0]
            str_result.append(ret1)
    for i in range(len(str_result) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            str_key = str_result[i] + "-->" + str_result[j]
            if str_key in result:
                tmp = int(result[str_key]) + 1
                result[str_key] = tmp
            else:
                result[str_key] = 1

def deal_event(stack,result,events):
    str_result=['0']
    for idx,s in enumerate(stack):
        # print(idx)
        if s !=0:
            s1=str(s)
            ret = re.search("event.*", s1).group().replace("event=","")
            ret1= ret.split(",except", 1)[0]
            if ret1 in events:
                events[ret1] += 1
            else:
                events[ret1] = 1
            str_result.append(ret1)

    for i in range(len(str_result)-1,0,-1):
        for j in range(i-1,-1,-1):
            str_key=str_result[i]+"-->"+str_result[j]
            if str_key in result:
                tmp=int(result[str_key])+1
                result[str_key] = tmp
            else:
                result[str_key] = 1
    # print(result)

def storage(result,events):
    with open('D:\研究生工作\ result.txt', 'w', encoding='utf-8') as f:
        for key in result.keys():
            f.write(key)
            f.write("\t:")
            f.write(str(result[key])+'次')
            f.write("\n")
        f.close()
    with open('D:\研究生工作\events.txt', 'a+', encoding='utf-8') as f:
        for key in events.keys():
            f.write(key)
            f.write("\n")
        f.close()
    print("存储完成")

def deal_cost(file_path):
    with open(file_path, "r",encoding="ISO-8859-1") as file:
        lines = file.readlines()
        lines.append("0")
        # print(file_path)
        # print(lines)
        cost_average=0
        costs=[]
        new_costs=[]
        cost_standard=0
        start_filter = lines.index("network[son<-parent]=\n")

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line!= "0":
                # print(line)
                cost = re.search("cost=.*", line).group().split(",event=", 1)[0]
                # print(cost)

                if len(re.findall("\d+\.?\d*", cost)) !=0:
                    costs.append(int(re.findall("\d+",cost)[0]))
                else:
                    costs.append(0)
            else:
                cost_average=round(np.sum(np.array(costs))/len(costs),4)
                for c in costs:
                    cost_standard+=(c-cost_average)*(c-cost_average)
                cost_standard=round(cost_standard/len(costs),4)
                # if cost_standard == 0:
                    # print(file_path)
                for c in costs:
                    if cost_standard ==0:
                        new_costs.append(c)
                    else:
                        new_cost=round((c-cost_average)/cost_standard,2)
                        if(str(new_cost)=="1.38.38" or str(new_cost)=="-1.38.38"):
                            print("输出1.38.38")
                            print(file_path)
                        new_costs.append(new_cost)
                # print(costs)
                # print(cost_average)
                # print(cost_standard)
                # print(new_costs)
        storage_cost(file_path,costs,new_costs)



def open_file(directorys):
    reflection=events_reflect()
    for directory in directorys:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(root, file_name)
                    # deal_cost(file_path)   处理cost
                    # prin_dig(file_path)
                    events_compress(file_path,reflection)


def storage_cost(file_path,costs,new_costs):
    new_lines=[]
    with open(file_path, "r",encoding="ISO-8859-1") as file:
        lines = file.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        gap=len(lines)-len(lines[start_filter + 1:])
        for i in range(len(lines)-gap):
            old_str="cost="+str(costs[i])
            new_str="cost="+str(new_costs[i])
            cost = re.search("cost=.*", lines[i + gap]).group().split(",event=", 1)[0]
            if len(re.findall("\d+\.?\d*", cost)) != 0:
                lines[i + gap] = lines[i + gap].replace(old_str, new_str)
            # print(lines[i+3])
        new_lines=lines
    file.close()
    with open(file_path, "w",encoding="ISO-8859-1") as f:
        for line in new_lines:
            f.write(line)
    f.close()
    print(file_path)
    print("Done！！")

def one_hot(file_path):
    result=[]
    with open(file_path, "r") as file:
        lines = file.readlines()
        T = np.zeros((len(lines), 57))
        # print(T)
        for idx,row in enumerate(T):
            row[idx] = 1
            line=str(lines[idx].replace("\n",""))+" -> "+str(row).replace("\n","").replace(".",",")
            result.append(line)
            print(result)
        print(T)
        file.close()
    with open(r"D:\研究生工作\txt\file_name_one_hot.txt", "w") as f:
        for line in result:
            f.write(line)
            f.write("\n")
        f.close()


def prin_dig(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines.append("0")
        # print(file_path)
        # print(lines)
        cost_average=0
        costs=[]
        new_costs=[]
        cost_standard=0
        start_filter = lines.index("network[son<-parent]=\n")

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line!= "0":
                # print(line)
                cost = re.search("cost=.*", line).group().split(",event=", 1)[0]
                if "1.38." in str(re.findall("\d+", cost)):
                    print(cost)
                    print(file_path)

def events_reflect():
    reflection={}
    with open(r"D:\研究生工作\txt\events.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n","")
            line_new = line.split(" ->", 1)[1].replace("\n","")
            if line_old in reflection:
                continue
            else:
                reflection[line_old] = line_new
    file.close()
    return reflection

def events_compress(file_path,reflection):
    # print(reflection)
    new_lines=[]
    with open(file_path, "r") as file:
        lines = file.readlines()
        if "network[son<-parent]=\n" in lines:
            start_filter = lines.index("network[son<-parent]=\n")
            for line in lines[0:start_filter+1]:
                new_lines.append(line)
        else:
            start_filter=0

        for line in lines[start_filter + 1:]:
            line = line.strip()
            event = re.search("event=.*", line).group().split(",exception=", 1)[0].replace("event=","")
            x=0
            number=0
            if "user id=" in reflection[event] or "user account id=" in reflection[event]:
                x = random.randint(0, 99999999)
                number = re.findall("\d+\.?\d*", reflection[event])[0]
            elif "article id=" in reflection[event]:
                x=random.randint(0,999999999999)
                number = re.findall("\d+\.?\d*", reflection[event])[0]
            elif "tag id=" in reflection[event] or "message id=" in reflection[event]:
                x=random.randint(0,99999999999999)
                number = re.findall("\d+\.?\d*", reflection[event])[0]
            elif "post id=" in reflection[event] or "posts id=" in reflection[event] or "tagposts id=" in reflection[event]:
                x=random.randint(0,9999999)
                number = re.findall("\d+\.?\d*", reflection[event])[0]
            elif "page id=" in reflection[event] or "faq id=" in reflection[event]:
                x=random.randint(0,9999999999)
                number = re.findall("\d+\.?\d*", reflection[event])[0]
            elif "comment id=" in reflection[event] or "email" in reflection[event]:
                x=random.randint(0,99999999999)
                number = re.findall("\d+\.?\d*", reflection[event])[0]
            # reflection[event]=reflection[event].replace("=",":").replace(str(number), str(x))
            x1=reflection[event].replace("=",":").replace(str(number),"*")
            line=line.replace(event,x1)
            new_lines.append(line)
    file.close()
    print(new_lines)
    with open(file_path, "w+") as file:
        lines = file.readlines()
        for line in new_lines:
            if "event" in line:
                file.write(line)
                file.write("\n")
            else:
                file.write(line)
    file.close()


def open_file_new(directory,list):

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # search_exception(file_path,list)
            # search_file_name(file_path,list)
            search_event(file_path,list)


def open_file_new_cost(directory):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            deal_invoke(file_path)

def deal_invoke(file_path):
    with open(file_path, "r",encoding="ISO-8859-1") as file:
        lines = file.readlines()

        start_line=lines.index("network[son<-parent]=\n")

        new_lines = lines[:start_line+1]
        for line in lines[start_line+1:]:
            line=line.replace("invoke","invokeInfo")
            new_lines.append(line)

        # print(new_lines)
        storage_new_lines(file_path,new_lines)


def search_exception(file_path,exception_list):
    print(file_path)
    with open(file_path, "r",encoding="ISO-8859-1") as file:
        lines = file.readlines()

        start_line=lines.index("network[son<-parent]=\n")
        for line in lines[start_line+1:]:
            # print(line)
            exception = line.split(",")[-1].replace("exception=","")

            if exception not in exception_list:
                exception_list[exception]=0
            else:
                exception_list[exception]+=1

def search_event(file_path,event_list):

    with open(file_path, "r",encoding="utf-8") as file:
        lines = file.readlines()

        start_line=lines.index("network[son<-parent]=\n")
        for line in lines[start_line+1:]:
            # print(line)
            chain, time, invokeInfo, cost, event, exception = line.split(",")
            event = event.replace("event=","")

            if event not in event_list:
                event_list[event]=0
            else:
                event_list[event]+=1

def search_file_name(file_path,filename_list):
    print(file_path)
    with open(file_path, "r",encoding="ISO-8859-1") as file:
        lines = file.readlines()

        start_line=lines.index("network[son<-parent]=\n")
        for line in lines[start_line+1:]:
            target_node_name=line.split("<-")[0]
            source_node_name_all=line.split("<-")[1]
            source_node_name=source_node_name_all.split(",")[0]

            # parts = line.split(",")
            # edge = parts[0].split("<-")
            # target_node_name = edge[0].strip()
            # source_node_name = edge[1].strip()

            if target_node_name not in filename_list:
                filename_list[target_node_name]=0
            else:
                filename_list[target_node_name]+=1

            if source_node_name not in filename_list:
                filename_list[source_node_name]=0
            else:
                filename_list[source_node_name]+=1

def storage_sort(list,storage_path):
    with open(storage_path,"w",encoding="utf-8") as file:
        for k in list.keys():
            file.write(k+"\n")


def storage_new_lines(file_path,new_lines):
    with open(file_path, "w",encoding="ISO-8859-1") as file:
        for line in new_lines:
            file.write(line)



'''# event_list'''
event_list={}
directory=r""
storage_path=r""
open_file_new(directory,event_list)
storage_sort(event_list,storage_path)
