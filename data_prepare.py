# import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import os

from data_prepare_const import *
from utils import MultiSubplotDraw

def one_time_deal_PET(data_path_list=None):
    if not data_path_list:
        data_path_list = ["data/271amyloid.csv", "data/271tau.csv", "data/271fdg.csv"]
    data_a = pd.read_csv(data_path_list[0])
    data_t = pd.read_csv(data_path_list[1])
    data_n = pd.read_csv(data_path_list[2])
    data_a = data_a[COLUMN_NAMES + TITLE_NAMES]
    data_t = data_t[COLUMN_NAMES + TITLE_NAMES]
    data_n = data_n[COLUMN_NAMES + TITLE_NAMES]

    class_number = 5

    for type_name, df in zip(["PET-A", "PET-T", "PET-N"], [data_a, data_t, data_n]):
        save_path = "data/PET/{}_{{}}.npy".format(type_name)
        collection = np.zeros((class_number, 160))
        counts = np.zeros(class_number)
        for index, row in df.iterrows():
            label = None
            for one_key in LABELS:
                if row["DX"] in LABELS[one_key]:
                    label = one_key
                    counts[LABEL_ID[label]] += 1
                    break

            if not label:
                # print("key not found: \"{}\"".format(row["DX"]))
                continue
            for i in range(160):
                collection[LABEL_ID[label]][i] += float(row[COLUMN_NAMES[i]])
        for one_key in LABELS:
            if counts[LABEL_ID[one_key]] != 0:
                avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
                # print(type_name, "avg({})".format(collection[LABEL_ID[one_key], :].shape))
                np.save(save_path.format(one_key), avg)
                print(one_key, np.mean(avg))
        print(type_name, "counts:", counts)


def one_time_deal_PET_specified(APOE, gender, data_path_list=None):
    if not data_path_list:
        data_path_list = ["data/271amyloid.csv", "data/271tau.csv", "data/271fdg.csv"]
    assert APOE in ["all", "zero", "one_two"]
    assert gender in ["all", "female", "male"]
    data_a = pd.read_csv(data_path_list[0])
    data_t = pd.read_csv(data_path_list[1])
    data_n = pd.read_csv(data_path_list[2])
    length_a_all = len(data_a)
    length_t_all = len(data_t)
    length_n_all = len(data_n)
    if APOE == "zero":
        data_a = data_a[data_a["APOE4"] == 0]
        data_t = data_t[data_t["APOE4"] == 0]
        data_n = data_n[data_n["APOE4"] == 0]
    elif APOE == "one_two":
        data_a = data_a[(data_a["APOE4"] == 1) | (data_a["APOE4"] == 2)]
        data_t = data_t[(data_t["APOE4"] == 1) | (data_t["APOE4"] == 2)]
        data_n = data_n[(data_n["APOE4"] == 1) | (data_n["APOE4"] == 2)]
    else:
        pass
    if gender == "female":
        data_a = data_a[data_a["PTGENDER"] == "Female"]
        data_t = data_t[data_t["PTGENDER"] == "Female"]
        data_n = data_n[data_n["PTGENDER"] == "Female"]
    elif gender == "male":
        data_a = data_a[data_a["PTGENDER"] == "Male"]
        data_t = data_t[data_t["PTGENDER"] == "Male"]
        data_n = data_n[data_n["PTGENDER"] == "Male"]
    else:
        pass
    length_a = len(data_a)
    length_t = len(data_t)
    length_n = len(data_n)
    print("[Class] APOE = {}, gender = {}".format(APOE, gender))
    print("[Count] A = {}/{}, T = {}/{}, N = {}/{}".format(length_a, length_a_all, length_t, length_t_all, length_n, length_n_all))
    data_a = data_a[COLUMN_NAMES + TITLE_NAMES]
    data_t = data_t[COLUMN_NAMES + TITLE_NAMES]
    data_n = data_n[COLUMN_NAMES + TITLE_NAMES]

    class_number = 5

    for type_name, df in zip(["PET-A", "PET-T", "PET-N"], [data_a, data_t, data_n]):
        save_path = "data/PET_specified/{}_APOE={}_gender={}_{{}}.npy".format(type_name, APOE, gender)
        collection = np.zeros((class_number, 160))
        counts = np.zeros(class_number)
        for index, row in df.iterrows():
            label = None
            for one_key in LABELS:
                if row["DX"] in LABELS[one_key]:
                    label = one_key
                    counts[LABEL_ID[label]] += 1
                    break

            if not label:
                # print("key not found: \"{}\"".format(row["DX"]))
                continue
            for i in range(160):
                collection[LABEL_ID[label]][i] += float(row[COLUMN_NAMES[i]])
        for one_key in LABELS:
            if counts[LABEL_ID[one_key]] != 0:
                avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
                # print(type_name, "avg({})".format(collection[LABEL_ID[one_key], :].shape))
                np.save(save_path.format(one_key), avg)
                print(one_key, np.mean(avg))
        print(type_name, "counts:", counts)
    print()


def one_time_deal_PET_all(data_path_list=None):
    if not data_path_list:
        data_path_list = ["data/Amyloid_Full.xlsx", "data/FDG_Full.xlsx"]
    full_names = ["Node {}".format(i) for i in range(1, 161)]
    data_a = pd.read_excel(data_path_list[0])
    # data_t = pd.read_csv(data_path_list[1])
    data_n = pd.read_excel(data_path_list[1])
    data_a = data_a[full_names + TITLE_NAMES]
    # data_t = data_t[COLUMN_NAMES + TITLE_NAMES]
    data_n = data_n[full_names + TITLE_NAMES]

    class_number = 5

    for type_name, df in zip(["PET-A", "PET-N"], [data_a, data_n]):
        save_path = "data/PET/{}_full_{{}}.npy".format(type_name)
        collection = np.zeros((class_number, 160))
        counts = np.zeros(class_number)
        for index, row in df.iterrows():
            label = None
            for one_key in LABELS:
                if row["DX"] in LABELS[one_key]:
                    label = one_key
                    counts[LABEL_ID[label]] += 1
                    break

            if not label:
                print("key not found: \"{}\"".format(row["DX"]))
                continue
            for i in range(160):
                collection[LABEL_ID[label]][i] += float(row[full_names[i]])
        for one_key in LABELS:
            if counts[LABEL_ID[one_key]] != 0:
                avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
                # print(type_name, "avg({})".format(collection[LABEL_ID[one_key], :].shape))
                np.save(save_path.format(one_key), avg)
                print(one_key, np.mean(avg))
        print(type_name, "counts:", counts)

def one_time_build_ptid_dictionary(dictionary_path=None):
    if not dictionary_path:
        dictionary_path = "data/MRI_information_All_Measurement.xlsx"
    df = pd.read_excel(dictionary_path)[["PTID", "DX"]]
    dic = dict()
    for index, row in df.iterrows():
        if len(row["PTID"]) < 2:
            continue
        ptid = row["PTID"].split("_")[-1]
        assert len(ptid) == 4
        dx = row["DX"]
        if dx not in LABELS and str(dx) != "nan":
            print("DX: \"{}\" not matches any!".format(dx))
            continue
        dic[ptid] = dx
    with open("data/CSF/ptid_dictionary.pkl", "wb") as f:
        pickle.dump(dic, f)
    return dic

def one_time_build_ptid_dictionary_specifed(dictionary_path=None):
    if not dictionary_path:
        dictionary_path = "data/MRI_information_All_Measurement.xlsx"
    df = pd.read_excel(dictionary_path)[["PTID", "DX", "PTGENDER", "APOE4"]]
    dic = dict()
    for index, row in df.iterrows():
        if len(row["PTID"]) < 2:
            continue
        ptid = row["PTID"].split("_")[-1]
        assert len(ptid) == 4
        dx = row["DX"]
        APOE = row["APOE4"]
        gender = row["PTGENDER"]
        if dx not in LABELS and str(dx) != "nan":
            print("DX: \"{}\" not matches any!".format(dx))
            continue
        dic[ptid] = [dx, APOE, gender]
    with open("data/CSF/ptid_dictionary_specified.pkl", "wb") as f:
        pickle.dump(dic, f)
    return dic

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
    for one_key in LABELS:
        if counts[LABEL_ID[one_key]] != 0:
            avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
            np.save("data/CSF/CSF_{}".format(one_key), avg)
            print("CSF_{} counts={} avg={}".format(one_key, counts[LABEL_ID[one_key]], avg))
    print("CSF counts:", counts)


def one_time_deal_CSF_specified(APOE, gender, csf_path=None, dictionary_pickle_path=None):
    if not csf_path:
        csf_path = "data/CSF_Bio_All_WF.csv"
    if not dictionary_pickle_path:
        dictionary_pickle_path = "data/CSF/ptid_dictionary_specified.pkl"
    with open(dictionary_pickle_path, "rb") as f:
        ptid_dic_specified = pickle.load(f)
    df = pd.read_csv(csf_path)
    assert APOE in ["all", "zero", "one_two"]
    assert gender in ["all", "female", "male"]
    print("[Class] APOE = {}, gender = {}".format(APOE, gender))

    df = df[["RID", "ABETA", "TAU", "PTAU"]]
    class_list = list(LABELS.keys())
    counts = np.zeros(len(class_list))
    collection = np.zeros([len(class_list), 3])

    for index, row in df.iterrows():
        ptid_key = str(int(row["RID"])).zfill(4)
        if ptid_key not in ptid_dic_specified:
            print("ptid key {} not found! Skip it!".format(ptid_key))
            continue
        label = ptid_dic_specified[ptid_key][0]
        if not (np.isnan(row[COLUMN_NAMES_CSF[0]]) or np.isnan(row[COLUMN_NAMES_CSF[1]]) or np.isnan(row[COLUMN_NAMES_CSF[2]])):
            if APOE == "zero":
                if ptid_dic_specified[ptid_key][1] not in [0]:
                    continue
            elif APOE == "one_two":
                if ptid_dic_specified[ptid_key][1] not in [1, 2]:
                    continue
            if gender == "male":
                if ptid_dic_specified[ptid_key][2] not in ["Male"]:
                    continue
            elif gender == "female":
                if ptid_dic_specified[ptid_key][2] not in ["Female"]:
                    continue
            counts[LABEL_ID[label]] += 1
            for i in range(3):
                collection[LABEL_ID[label]][i] += float(row[COLUMN_NAMES_CSF[i]])
    for one_key in LABELS:
        if counts[LABEL_ID[one_key]] != 0:
            avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
            np.save("data/CSF_specified/CSF_APOE={}_gender={}_{}".format(APOE, gender, one_key), avg)
            print("CSF_{} counts={} avg={}".format(one_key, counts[LABEL_ID[one_key]], avg))
    print("CSF counts = {}:".format(np.sum(counts)), counts)
    print()


def percent_diff(base, data):
    diff = (data - base) / base
    diff_min = np.min(diff)
    diff_max = np.max(diff)
    diff_avg = np.mean(diff)
    diff_string = "min: {0}{1:.1f}%, avg: {2}{3:.1f}%, max: {4}{5:.1f}%".format(
        "+" if diff_min > 0 else "",
        diff_min * 100,
        "+" if diff_avg > 0 else "",
        diff_avg * 100,
        "+" if diff_max > 0 else "",
        diff_max * 100,
    )
    return diff_string


def one_time_compare(data_name_1, data_name_2, legend_format_list, title, path_format="data/PET/{}_{}.npy"):
    label_list = LABEL_LIST

    m = MultiSubplotDraw(row=5, col=1, fig_size=(25, 30), show_flag=True, save_flag=False, tight_layout_flag=False, title=title, title_size=40)
    for one_label in label_list:
        data1 = np.load(path_format.format(data_name_1, one_label))
        data2 = np.load(path_format.format(data_name_2, one_label))
        diff = percent_diff(data1, data2)
        m.add_subplot(
            y_lists=[data1, data2],
            x_list=range(1, 161),
            color_list=["r", "b"],
            legend_list=[item.format(one_label) for item in legend_format_list],
            line_style_list=["solid"] * 2,
            fig_title="{} ({})".format(one_label, diff),
        )

    m.draw()


class ConstTruthSpecified:
    def __init__(self, **params):
        assert "csf_folder_path" in params and "pet_folder_path" in params, "please provide the save folder paths"
        assert "dataset" in params
        assert "APOE" in params and "gender" in params
        csf_folder_path, pet_folder_path = params["csf_folder_path"], params["pet_folder_path"]
        self.APOE, self.gender = params["APOE"], params["gender"]
        label_list = LABEL_LIST  # [[0, 2, 3, 4]]  # skip the second nodes (SMC)
        self.class_num = len(label_list)
        if "x" not in params:
            self.x_all = np.asarray([3, 6, 9, 11, 12])
        else:
            self.x_all = np.asarray(params.get("x"))
        self.y = dict()
        self.x = dict()
        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
        self.plot_names = ["$A_{PET}$", "$T_{PET}$", "$N_{PET}$", "$A_{CSF}$", "$T_{pCSF}$", "$T_{CSF}$",
                             "$T_{tCSF}$"]
        for one_line in self.lines:
            self.y[one_line] = []
            self.x[one_line] = self.x_all
        for i, class_name in enumerate(label_list):
            csf_data = np.load(os.path.join(csf_folder_path, "CSF_APOE={}_gender={}_{}.npy".format(self.APOE, self.gender, class_name)))
            pet_data_a = np.load(os.path.join(pet_folder_path, "PET-A_APOE={}_gender={}_{}.npy".format(self.APOE, self.gender, class_name)))
            pet_data_t = np.load(os.path.join(pet_folder_path, "PET-T_APOE={}_gender={}_{}.npy".format(self.APOE, self.gender, class_name)))
            pet_data_n = np.load(os.path.join(pet_folder_path, "PET-N_APOE={}_gender={}_{}.npy".format(self.APOE, self.gender, class_name)))
            self.y["APET"] = self.y["APET"] + [np.mean(pet_data_a)]
            self.y["TPET"] = self.y["TPET"] + [np.mean(pet_data_t)]
            self.y["NPET"] = self.y["NPET"] + [np.mean(pet_data_n)]

            self.y["ACSF"] = self.y["ACSF"] + [csf_data[0]]
            self.y["TtCSF"] = self.y["TtCSF"] + [csf_data[1]]
            self.y["TpCSF"] = self.y["TpCSF"] + [csf_data[2]]
            self.y["TCSF"] = self.y["TCSF"] + [csf_data[1] - csf_data[2]]
        for one_key in self.lines:
            self.y[one_key] = np.asarray(self.y[one_key])
        self.y["NPET"] = 2.0 - self.y["NPET"]  # 1.0 - (self.y["NPET"] - np.min(self.y["NPET"])) / (np.max(self.y["NPET"]) - np.min(self.y["NPET"]))
        assert params["dataset"] in ["all", "chosen_0"]
        if params["dataset"] == "chosen_0":
            for one_key in ["NPET"]:
                self.y[one_key] = self.y[one_key][[]]
                self.x[one_key] = self.x[one_key][[]]
            for one_key in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:
                self.y[one_key] = self.y[one_key][[0, 2, 3, 4]]
                self.x[one_key] = self.x[one_key][[0, 2, 3, 4]]
        else:
            pass

def one_time_plot_ct(ct: ConstTruthSpecified):
    save_folder = "data/Figure_specified/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = "{}/APOE={}_gender={}.png".format(save_folder, ct.APOE, ct.gender)
    fig = plt.figure(figsize=(24, 18))
    targets = ct.lines
    for i in range(7):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.scatter(x=ct.x[targets[i]], y=ct.y[targets[i]], s=100, facecolor="red", alpha=0.5, marker="o", edgecolors='black', linewidths=1, zorder=10)
        ax.set_xlim([0, 13])
        ax.set_title(ct.plot_names[i], fontsize=15)
    fig.suptitle("Truth: APOE = {}, Gender = {}".format(ct.APOE, ct.gender), fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("Figure is saved to {}".format(save_path))
    plt.show()
    plt.close()

if __name__ == "__main__":
    # # one_time_deal_PET()
    # # d = one_time_build_ptid_dictionary()
    # # one_time_deal_CSF()
    # # one_time_deal_PET()
    # # one_time_deal_PET_all()
    one_time_compare("PET-A_full", "PET-A", ["PET-A_full_{}", "PET-A_{}"], "PET_A")
    one_time_compare("PET-N_full", "PET-N", ["PET-N_full_{}", "PET-N_{}"], "PET_N")
    # one_time_build_ptid_dictionary_specifed()
    # for one_APOE in ["all", "zero", "one_two"]:
    #     for one_gender in ["all", "male", "female"]:
    #         one_time_deal_PET_specified(APOE=one_APOE, gender=one_gender)
    # for one_APOE in ["all", "zero", "one_two"]:
    #     for one_gender in ["all", "male", "female"]:
    #         one_time_deal_CSF_specified(APOE=one_APOE, gender=one_gender)
    # for one_APOE in ["all", "zero", "one_two"]:
    #     for one_gender in ["all", "male", "female"]:
    #         ct = ConstTruthSpecified(
    #             csf_folder_path="data/CSF_specified/",
    #             pet_folder_path="data/PET_specified/",
    #             dataset="all",
    #             APOE=one_APOE,
    #             gender=one_gender
    #         )
    #         one_time_plot_ct(ct)
    pass
