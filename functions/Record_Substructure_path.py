
import random
from NODE_TRANSFORMER.functions.构建树 import *

def open_file(directory,result):

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            log_nodes1(file_path,result)


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


def record_substructure(directory,train_ratio,val_ratio):
    result={}
    train_paths=[]
    val_paths=[]
    test_paths=[]

    data_paths = []
    exist = os.path.exists("data_paths.txt")

    open_file(directory,result)

    key_list=list(result.keys())
    random.shuffle(key_list)

    train_num = int(len(key_list) * train_ratio)
    val_num = int(len(key_list) * val_ratio)
    train_set_key = key_list[0: train_num]
    val_set_key = key_list[train_num: train_num + val_num]
    test_set_key = key_list[train_num + val_num:]

    for key in train_set_key:
        for path in result[key]:
            train_paths.append(path)
    for key in val_set_key:
        for path in result[key]:
            val_paths.append(path)
    for key in test_set_key:
        for path in result[key]:

            test_paths.append(path)

    # with open("data_paths.txt","w") as file:
    #     for path in paths:
    #         file.write(path+",")
    # with open("data_paths.txt", "r") as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         paths = line.split(",")[:-1]
    #         data_paths = data_paths + paths

    event_embedding_path = r""

    train_dataset=process_files_to_construct_trees(train_paths,event_embedding_path)
    val_dataset=process_files_to_construct_trees(val_paths,event_embedding_path)
    test_dataset=process_files_to_construct_trees(test_paths,event_embedding_path)

    return train_dataset,val_dataset,test_dataset

def record_substructure_new(directory,train_ratio,val_ratio):
    result={}
    train_paths=[]
    val_paths=[]
    test_paths=[]
    test_one_paths = []
    data_paths = []
    exist = os.path.exists("data_paths.txt")


    open_file(directory,result)


    key_list=list(result.keys())
    random.shuffle(key_list)

    train_num = int(len(key_list) * train_ratio)
    val_num = int(len(key_list) * val_ratio)
    train_set_key = key_list[0: train_num]
    val_set_key = key_list[train_num: train_num + val_num]
    test_set_key = key_list[train_num + val_num:]

    for key in train_set_key:
        train_paths += result[key]
    for key in val_set_key:
        val_paths += result[key]
    for key in test_set_key:
        test_paths += result[key]
        length = len(result[key])
        num = random.randint(0, length - 1)
        test_one_paths.append(result[key][num])


    event_embedding_path = r""

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    return train_paths, val_paths, test_paths, test_one_paths


## GCN+GRU
def record_substructure_to_treeset(directory,train_ratio,val_ratio):
    result={}
    train_paths=[]
    val_paths=[]
    test_paths=[]
    test_one_paths = []
    data_paths = []
    exist = os.path.exists("data_paths.txt")

    i = 0
    sub_result = []

    open_file(directory,result)

    for key in result.keys():
        for path in result[key]:
            sub_result.append(str(i) + " : " + path)
        i = i + 1
    # print(sub_result)


    with open("","w",encoding="utf-8") as file:
        for line in sub_result:
            file.write(line)
            file.write("\n")
        file.close()

    key_list1 = list(result.keys())
    key_list=list(result.keys())
    random.shuffle(key_list)

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

    with open("","w") as file:
        for line in specify_data:
            file.write(line)
            file.write("\n")
        file.close()


    event_embedding_path = r""

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset = process_files_to_construct_trees(train_paths, event_embedding_path)
    val_dataset = process_files_to_construct_trees(val_paths, event_embedding_path)
    test_dataset = process_files_to_construct_trees(test_paths, event_embedding_path)
    test_one_dataset = process_files_to_construct_trees(test_one_paths, event_embedding_path)

    return train_dataset, val_dataset, test_dataset, test_one_dataset


def record_substructure_to_treeset_new(directory, train_ratio, val_ratio, record):
    result = {}
    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []
    # data_paths = []
    # exist = os.path.exists("data_paths.txt")

    i = 0
    sub_result = []

    open_file(directory, result)

    for key in result.keys():
        for path in result[key]:
            sub_result.append(str(i) + " : " + path)
        i = i + 1

    if record == 1:
        with open("/root/autodl-tmp/project/data/graph_structure_path_dict44.txt", "w", encoding="utf-8") as file:
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
        with open("/root/autodl-tmp/project/data/specific_dataset44.txt", "w", encoding="utf-8") as file:
            for line in specify_data:
                file.write(line)
                file.write("\n")
            file.close()

    event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot.txt"
    # event_embedding_path = r"/root/autodl-tmp/project/data/events_compress_one_hot2.txt"

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset = process_files_to_construct_trees(train_paths, event_embedding_path)
    val_dataset = process_files_to_construct_trees(val_paths, event_embedding_path)
    test_dataset = process_files_to_construct_trees(test_paths, event_embedding_path)
    test_one_dataset = process_files_to_construct_trees(test_one_paths, event_embedding_path)

    return train_dataset, val_dataset, test_dataset, test_one_dataset


def record_substructure_to_datatxt(dataname, train_ratio, val_ratio, record, specify_number):
    result = {}
    train_paths = []
    val_paths = []
    test_paths = []
    test_one_paths = []

    root = f"./root/autodl-tmp/project/data/{dataname}_data/"

    graph_structure_path = root + f"graph_structure_path_dict{specify_number}.txt"

    specific_dataset_path = root + f"specific_dataset{specify_number}.txt"

    directory=f"./root/autodl-tmp/project/data/{dataname}"

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

    event_embedding_path = event_embedding_path = root+"events_compress_one_hot.txt"

    print("train_paths:  " + str(len(train_paths)))
    print("val_paths:  " + str(len(val_paths)))
    print("test_paths:  " + str(len(test_paths)))
    print("test_one_paths:  " + str(len(test_one_paths)))

    train_dataset = process_files_to_construct_trees(train_paths, event_embedding_path)
    val_dataset = process_files_to_construct_trees(val_paths, event_embedding_path)
    test_dataset = process_files_to_construct_trees(test_paths, event_embedding_path)
    test_one_dataset = process_files_to_construct_trees(test_one_paths, event_embedding_path)

    return train_dataset, val_dataset, test_dataset, test_one_dataset

if __name__ == '__main__':
    print("hello")



