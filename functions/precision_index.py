from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score,precision_recall_curve,auc
def show_metrics_AUPRC_new(true_labels, predicted_labels, predicted_prob,storage_path,val_result_list,rewrite=1):
    f1 = f1_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    print(f"true F1 Score: {f1}, true Recall: {recall}, true Precision: {precision}")

    false_f1 = f1_score(true_labels, predicted_labels, pos_label=0)
    false_recall = recall_score(true_labels, predicted_labels, pos_label=0)
    false_precision = precision_score(true_labels, predicted_labels, pos_label=0)
    print(f"false F1 Score: {false_f1}, false Recall: {false_recall}, false Precision: {false_precision}")

    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    micro_recall = recall_score(true_labels, predicted_labels, average='micro')
    micro_precision = precision_score(true_labels, predicted_labels, average='micro')
    micro_auc = roc_auc_score(true_labels, predicted_prob, average='micro')
    print(f"micro F1 Score: {micro_f1}, micro Recall: {micro_recall}, micro Precision: {micro_precision}, micro AUC Score: {micro_auc}")

    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    macro_precision = precision_score(true_labels, predicted_labels, average='macro')
    macro_auc = roc_auc_score(true_labels, predicted_prob, average='macro')
    print(f"macro F1 Score: {macro_f1}, macro Recall: {macro_recall}, macro Precision: {macro_precision}, macro AUC Score: {macro_auc}")

    auc_value = roc_auc_score(true_labels, predicted_prob)
    print(f"AUC Score: {auc_value}")

    p, r, t = precision_recall_curve(true_labels, predicted_prob)
    auprc = auc(r, p)
    print(f"AUPRC Score: {auprc}")

    write_sort = ""
    if rewrite == 1:
        write_sort = "w"
    else:
        write_sort = "a"

    val_result_list.append("测试集总精度:" + '\n')
    val_result_list.append("==========================" + '\n')
    val_result_list.append("true F1 Score:  " + str(f1) + "\n")
    val_result_list.append("true Recall:  " + str(recall) + "\n")
    val_result_list.append("true Precision:  " + str(precision) + "\n")
    val_result_list.append("false F1 Score:  " + str(false_f1) + "\n")
    val_result_list.append("false Recall:  " + str(false_recall) + "\n")
    val_result_list.append("false Precision:  " + str(false_precision) + "\n")
    val_result_list.append("micro F1 Score:  " + str(micro_f1) + "\n")
    val_result_list.append("micro Recall:  " + str(micro_recall) + "\n")
    val_result_list.append("micro Precision:  " + str(micro_precision) + "\n")
    val_result_list.append("micro AUC Score:  " + str(micro_auc) + "\n")
    val_result_list.append("macro F1 Score:  " + str(macro_f1) + "\n")
    val_result_list.append("macro Recall:  " + str(macro_recall) + "\n")
    val_result_list.append("macro Precision:  " + str(macro_precision) + "\n")
    val_result_list.append("macro AUC Score:  " + str(macro_auc) + "\n")
    val_result_list.append("AUC Score:  " + str(auc_value) + "\n")
    val_result_list.append("AUPRC Score:  " + str(auprc) + "\n")

    # with open(storage_path, write_sort) as file:
    #     file.write("测试集总精度:" + '\n')
    #     file.write("==========================" + '\n')
    #     file.write("true F1 Score:  " + str(f1) + "\n")
    #     file.write("true Recall:  " + str(recall) + "\n")
    #     file.write("true Precision:  " + str(precision) + "\n")
    #     file.write("false F1 Score:  " + str(false_f1) + "\n")
    #     file.write("false Recall:  " + str(false_recall) + "\n")
    #     file.write("false Precision:  " + str(false_precision) + "\n")
    #     file.write("micro F1 Score:  " + str(micro_f1) + "\n")
    #     file.write("micro Recall:  " + str(micro_recall) + "\n")
    #     file.write("micro Precision:  " + str(micro_precision) + "\n")
    #     file.write("micro AUC Score:  " + str(micro_auc) + "\n")
    #     file.write("macro F1 Score:  " + str(macro_f1) + "\n")
    #     file.write("macro Recall:  " + str(macro_recall) + "\n")
    #     file.write("macro Precision:  " + str(macro_precision) + "\n")
    #     file.write("macro AUC Score:  " + str(macro_auc) + "\n")
    #     file.write("AUC Score:  " + str(auc_value) + "\n")
    #     file.write("AUPRC Score:  " + str(auprc) + "\n")
    #     file.close()
    return f1, recall, precision, auc_value, auprc


def show_metrics_new(true_labels, predicted_labels, predicted_prob,storage_path,rewrite=1):
    f1 = f1_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    print(f"true F1 Score: {f1}, true Recall: {recall}, true Precision: {precision}")
    false_f1 = f1_score(true_labels, predicted_labels, pos_label=0)
    false_recall = recall_score(true_labels, predicted_labels, pos_label=0)
    false_precision = precision_score(true_labels, predicted_labels, pos_label=0)
    print(f"false F1 Score: {false_f1}, false Recall: {false_recall}, false Precision: {false_precision}")
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    micro_recall = recall_score(true_labels, predicted_labels, average='micro')
    micro_precision = precision_score(true_labels, predicted_labels, average='micro')
    micro_auc = roc_auc_score(true_labels, predicted_prob, average='micro')
    print(f"micro F1 Score: {micro_f1}, micro Recall: {micro_recall}, micro Precision: {micro_precision}, micro AUC Score: {micro_auc}")
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    macro_precision = precision_score(true_labels, predicted_labels, average='macro')
    macro_auc = roc_auc_score(true_labels, predicted_prob, average='macro')
    print(f"macro F1 Score: {macro_f1}, macro Recall: {macro_recall}, macro Precision: {macro_precision}, macro AUC Score: {macro_auc}")
    write_sort = ""
    if rewrite == 1:
        write_sort = "w"
    else:
        write_sort = "a"

    with open(storage_path, write_sort) as file:
        file.write("测试集总精度:" + '\n')
        file.write("==========================" + '\n')
        file.write("true F1 Score:  " + str(f1) + "\n")
        file.write("true Recall:  " + str(recall) + "\n")
        file.write("true Precision:  " + str(precision) + "\n")
        file.write("false F1 Score:  " + str(false_f1) + "\n")
        file.write("false Recall:  " + str(false_recall) + "\n")
        file.write("false Precision:  " + str(false_precision) + "\n")
        file.write("micro F1 Score:  " + str(micro_f1) + "\n")
        file.write("micro Recall:  " + str(micro_recall) + "\n")
        file.write("micro Precision:  " + str(micro_precision) + "\n")
        file.write("micro AUC Score:  " + str(micro_auc) + "\n")
        file.write("macro F1 Score:  " + str(macro_f1) + "\n")
        file.write("macro Recall:  " + str(macro_recall) + "\n")
        file.write("macro Precision:  " + str(macro_precision) + "\n")
        file.write("macro AUC Score:  " + str(macro_auc) + "\n")

    return f1, recall, precision, micro_auc



def show_metrics(true_labels, predicted_labels,storage_path,rewrite=1):
    f1 = f1_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    print(f"true F1 Score: {f1}, true Recall: {recall}, true Precision: {precision}")
    false_f1 = f1_score(true_labels, predicted_labels, pos_label=0)
    false_recall = recall_score(true_labels, predicted_labels, pos_label=0)
    false_precision = precision_score(true_labels, predicted_labels, pos_label=0)
    print(f"false F1 Score: {false_f1}, false Recall: {false_recall}, false Precision: {false_precision}")
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    micro_recall = recall_score(true_labels, predicted_labels, average='micro')
    micro_precision = precision_score(true_labels, predicted_labels, average='micro')
    micro_auc = roc_auc_score(true_labels, predicted_labels, average='micro')
    print(f"micro F1 Score: {micro_f1}, micro Recall: {micro_recall}, micro Precision: {micro_precision}, micro AUC Score: {micro_auc}")
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    macro_precision = precision_score(true_labels, predicted_labels, average='macro')
    macro_auc = roc_auc_score(true_labels, predicted_labels, average='macro')
    print(f"macro F1 Score: {macro_f1}, macro Recall: {macro_recall}, macro Precision: {macro_precision}, macro AUC Score: {macro_auc}")
    write_sort=""
    if rewrite==1:
        write_sort="w"
    else:
        write_sort="a"

    with open(storage_path, write_sort) as file:
        file.write("测试集总精度:" + '\n')
        file.write("==========================" + '\n')
        file.write("true F1 Score:  "+str(f1) + "\n")
        file.write("true Recall:  "+str(recall) + "\n")
        file.write("true Precision:  "+str(precision) + "\n")
        file.write("false F1 Score:  "+str(false_f1) + "\n")
        file.write("false Recall:  "+str(false_recall) + "\n")
        file.write("false Precision:  "+str(false_precision) + "\n")
        file.write("micro F1 Score:  "+str(micro_f1) + "\n")
        file.write("micro Recall:  "+str(micro_recall) + "\n")
        file.write("micro Precision:  "+str(micro_precision) + "\n")
        file.write("micro AUC Score:  "+str(micro_auc) + "\n")
        file.write("macro F1 Score:  "+str(macro_f1) + "\n")
        file.write("macro Recall:  "+str(macro_recall) + "\n")
        file.write("macro Precision:  "+str(macro_precision) + "\n")
        file.write("macro AUC Score:  "+str(macro_auc) + "\n")

