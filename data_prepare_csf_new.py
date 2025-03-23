# import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import os

from data_prepare_const import *
from utils import MultiSubplotDraw



def one_time_deal_CSF(csf_path=None, dictionary_pickle_path=None):
    if not csf_path:
        csf_path = "data/CSF_Bio_All_WF.csv"
    if not dictionary_pickle_path:
        dictionary_pickle_path = "data/CSF/ptid_dictionary.pkl"
    with open(dictionary_pickle_path, "rb") as f:
        ptid_dic = pickle.load(f)
    df = pd.read_csv(csf_path)[["RID", "ABETA", "TAU", "PTAU"]]
    class_list = list(LABELS.keys())
    counts = np.zeros(len(class_list))
    collection = np.zeros([len(class_list), 3])

    csf_patient_wise_dict = dict()
    # collection_acsf = [[] for i in range(5)]
    for one_key in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:
        csf_patient_wise_dict[one_key] = [[] for i in range(5)]
    for index, row in df.iterrows():
        ptid_key = str(int(row["RID"])).zfill(4)
        if ptid_key not in ptid_dic:
            print("ptid key {} not found! Skip it!".format(ptid_key))
            continue
        label = ptid_dic[ptid_key]
        if not (np.isnan(row[COLUMN_NAMES_CSF[0]]) or np.isnan(row[COLUMN_NAMES_CSF[1]]) or np.isnan(row[COLUMN_NAMES_CSF[2]])):
            counts[LABEL_ID[label]] += 1
            for i in range(3):
                collection[LABEL_ID[label]][i] += float(row[COLUMN_NAMES_CSF[i]])
            csf_patient_wise_dict["ACSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[0]]))
            csf_patient_wise_dict["TpCSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[2]]))
            csf_patient_wise_dict["TCSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[1]]) - float(row[COLUMN_NAMES_CSF[2]]))
            csf_patient_wise_dict["TtCSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[1]]))
    for one_key in LABELS:
        if counts[LABEL_ID[one_key]] != 0:
            avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
            np.save("data/CSF/CSF_{}".format(one_key), avg)
            print("CSF_{} counts={} avg={}".format(one_key, counts[LABEL_ID[one_key]], avg))
    print("CSF counts:", counts)
    # print(csf_patient_wise_dict)
    for one_key in csf_patient_wise_dict:
        print(one_key)
        for i in range(5):
            print(len(csf_patient_wise_dict[one_key][i]))
    with open("test/CSF_dict.pkl", "wb") as f:
        pickle.dump(csf_patient_wise_dict, f)


def get_year(vis_code):
    if vis_code == "bl":
        return 0
    else:
        y_string = vis_code[1:]
        y_int = int(y_string)
        assert y_int % 12 == 0
        return y_int


def one_time_build_CSF_upenn():
    dic_path = "data/dx_dictionary_5.pkl"
    with open(dic_path, "rb") as f:
        old_dic = pickle.load(f)
    new_dic = dict()
    for one_key in old_dic:
        if one_key in new_dic:
            print("conflict at {}!".format(one_key))
        new_dic[one_key[-4:]] = old_dic[one_key]

    csf_path = "data/UPENNBIOMK_MASTER.csv"
    df = pd.read_csv(csf_path, engine="python")[["RID", "VISCODE", "ABETA", "TAU", "PTAU"]]
    tmp_ptid_key = ""
    tmp_vis_code = ""
    csf_dic_data = dict()
    # csf_dic_res = dict()
    missed_count = 0
    for index, row in df.iterrows():
        ptid_key = str(int(row["RID"])).zfill(4)
        vis_code = str(row["VISCODE"])
        # if ptid_key == tmp_ptid_key and vis_code == tmp_vis_code:
        #     continue

        if ptid_key not in new_dic:
            if ptid_key != tmp_ptid_key:
                print("old key {}. key {} missed. skip!".format(tmp_ptid_key, ptid_key))
                missed_count += 1
            tmp_ptid_key = ptid_key
            tmp_vis_code = vis_code
            continue
        tmp_ptid_key = ptid_key
        tmp_vis_code = vis_code
        ad_class = new_dic[ptid_key]
        year = get_year(vis_code)
        """csf_patient_wise_dict["ACSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[0]]))
            csf_patient_wise_dict["TpCSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[2]]))
            csf_patient_wise_dict["TCSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[1]]) - float(row[COLUMN_NAMES_CSF[2]]))
            csf_patient_wise_dict["TtCSF"][LABEL_ID[label]].append(float(row[COLUMN_NAMES_CSF[1]]))"""
        acsf = float(row["ABETA"])
        tpcsf = float(row["PTAU"])
        tcsf = float(row["TAU"]) - float(row["PTAU"])
        ttcsf = float(row["TAU"])
        if np.isnan(acsf) or np.isnan(tpcsf) or np.isnan(ttcsf):
            continue
        if ptid_key not in csf_dic_data:
            csf_dic_data[ptid_key] = []
        new_pair = [ptid_key, ad_class, year, acsf, tpcsf, tcsf, ttcsf]
        # print(new_pair)
        csf_dic_data[ptid_key].append(new_pair)
    print(len(list(csf_dic_data.keys())))
    # 1086 matched all
    # 213 CN
    with open("data/CSF/csf_new_dictionary_all.pkl", "wb") as f:
        pickle.dump(csf_dic_data, f)
    print("missed count: {}".format(missed_count))

def one_time_draw_CSF_upenn():
    with open("data/CSF/csf_new_dictionary_CN.pkl", "rb") as f:
        dic = pickle.load(f)
    # print(sum([len(dic[one_key]) for one_key in dic]))


    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("ACSF")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("TpCSF")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("TCSF")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("TtCSF")
    for one_key in dic:
        time_list = []
        acsf_list = []
        tpcsf_list = []
        tcsf_list = []
        ttcsf_list = []
        for one_pair in dic[one_key]:
            time_list.append(one_pair[2])
            acsf_list.append(one_pair[3])
            tpcsf_list.append(one_pair[4])
            tcsf_list.append(one_pair[5])
            ttcsf_list.append(one_pair[6])

        ax1.scatter(x=time_list, y=acsf_list, s=10, facecolor="red", alpha=0.5, marker="o", edgecolors='black', linewidths=1, zorder=10)
        ax1.plot(time_list, acsf_list, c="red")

        ax2.scatter(x=time_list, y=tpcsf_list, s=10, facecolor="g", alpha=0.5, marker="o", edgecolors='black',
                    linewidths=1, zorder=10)
        ax2.plot(time_list, tpcsf_list, c="g")

        ax3.scatter(x=time_list, y=tcsf_list, s=10, facecolor="b", alpha=0.5, marker="o", edgecolors='black',
                    linewidths=1, zorder=10)
        ax3.plot(time_list, tcsf_list, c="b")

        ax4.scatter(x=time_list, y=ttcsf_list, s=10, facecolor="orange", alpha=0.5, marker="o", edgecolors='black',
                    linewidths=1, zorder=10)
        ax4.plot(time_list, ttcsf_list, c="orange")
    plt.savefig("test/csf_new.png", dpi=400)
    plt.show()
    plt.close()





if __name__ == "__main__":
    one_time_build_CSF_upenn()
    # one_time_draw_CSF_upenn()
    pass
