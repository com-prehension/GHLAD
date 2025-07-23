import warnings
from datetime import datetime
from functions.set_seed import *
from NEW_GCN_NODE_TRANSFORMER.new_model.Multi_Layer_forum_8 import  main as run_forum
from functions.dataload import search_specify_data_from_dataname_for_halo,search_specify_data_from_dataname_for_forum,read_exist_substructure_to_datatxt_for_forum,read_exist_substructure_to_datatxt_for_halo

# 忽略特定的UserWarning
warnings.filterwarnings("ignore")


"""该启动程序主要应用于halo的层次学习"""
def stated_train_run():

    for specify_number in [18]:
        dataname = "novel"
        specify_data = True
        storage_number = "新动态中心-801"

        start_time = datetime.now()
        print("First Model training started at:", start_time)

        if specify_data:
            '''下面是使用指定的811数据集'''
            if dataname == "halo":
                train_tree_set, val_tree_set, test_data_list, test_one_list = search_specify_data_from_dataname_for_halo(
                    dataname, specify_number)
            else:
                train_tree_set, val_tree_set, test_data_list, test_one_list = search_specify_data_from_dataname_for_forum(
                    dataname, specify_number)

            root = f"/root/autodl-tmp/project/新result_8/{dataname}/指定数据集{specify_number}/"

        else:
            '''使用结构划分数据集'''
            if dataname == "halo":
                train_tree_set, val_tree_set, test_data_list, test_one_list = read_exist_substructure_to_datatxt_for_halo(
                    dataname, train_ratio=0.8, val_ratio=0.1, record=1, specify_number=specify_number)
            else:
                train_tree_set, val_tree_set, test_data_list, test_one_list = read_exist_substructure_to_datatxt_for_forum(
                    dataname, train_ratio=0.8, val_ratio=0.1, record=1, specify_number=specify_number)
            root = f"/root/autodl-tmp/project/新result_5/novel/划分数据集/"

        storage_path = root + f"no.{storage_number}_result.txt"
        storage_loss_path = root + f"no.{storage_number}_loss.txt"
        storage_test_path = root + f"no.{storage_number}_test_one.txt"

        # 4局部+层次使用

        # 用于验证4局部+层次
        run_forum(dataname,train_tree_set, val_tree_set, test_data_list, test_one_list, storage_path, storage_loss_path, storage_test_path)

        end_time = datetime.now()
        print("First Model training ended at:", end_time)
        # 计算耗时
        elapsed_time = end_time - start_time
        print("First Model training elapsed time:", elapsed_time)


        with open(storage_path, "a") as file:
            file.write("First Model training elapsed time:"+ str(elapsed_time))
            file.close()


if __name__ == "__main__":
    seed=42
    set_seed(seed)
    stated_train_run()







