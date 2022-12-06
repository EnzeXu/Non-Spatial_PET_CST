import os

import matplotlib.pyplot as plt
import numpy as np

import time
from scipy.integrate import odeint
from tqdm import tqdm

# from parameters import *
from config import Start, Config
from utils import MultiSubplotDraw

from const import *
"""
# Chen asked me to add these comments here - if you need :) 
CSF_CN counts=153.0 avg=[203.15882353  64.9379085   27.99281046]
CSF_SMC counts=78.0 avg=[204.64102564  63.8974359   35.9025641 ]
CSF_EMCI counts=41.0 avg=[190.7804878   79.9902439   34.88780488]
CSF_LMCI counts=108.0 avg=[184.77314815  87.7         34.23425926]
CSF_AD counts=307.0 avg=[144.74918567 119.13485342  47.93029316]
CSF counts: [153.  78.  41. 108. 307.]
PET-A counts: [81. 42. 88. 35. 19.]
PET-T counts: [78. 42. 83. 30. 32.]
PET-N counts: [92. 43. 80. 39. 11.]
"""


def get_now_string():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def numpy_safe_pow(x, n):
    return np.sign(x) * (np.abs(x)) ** n


def my_matmul(m, x, expand=160):
    x = np.ones(expand) * x[0]
    return np.mean(np.matmul(m, x)).reshape(1)


class ConstTruth:
    def __init__(self, **params):
        assert "csf_folder_path" in params and "pet_folder_path" in params, "please provide the save folder paths"
        csf_folder_path, pet_folder_path = params["csf_folder_path"], params["pet_folder_path"]
        label_list = LABEL_LIST  # [[0, 2, 3, 4]]  # skip the second nodes (SMC)
        self.class_num = len(label_list)
        if "x" not in params:
            self.x = np.asarray([0, 3, 6, 8, 9])
        else:
            self.x = np.asarray(params.get("x"))
        self.y = dict()
        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
        for one_line in self.lines:
            self.y[one_line] = []
        for i, class_name in enumerate(label_list):
            csf_data = np.load(os.path.join(csf_folder_path, "CSF_{}.npy".format(class_name)))
            pet_data_a = np.load(os.path.join(pet_folder_path, "PET-A_{}.npy".format(class_name)))
            pet_data_t = np.load(os.path.join(pet_folder_path, "PET-T_{}.npy".format(class_name)))
            pet_data_n = np.load(os.path.join(pet_folder_path, "PET-N_{}.npy".format(class_name)))
            # if class_name == "CN":
            #     print("truth APET:", np.mean(pet_data_a))
            #     print("truth TPET:", np.mean(pet_data_t))
            #     print("truth NPET:", np.mean(pet_data_n))
            #     print("truth ACSF", csf_data[0])
            #     print("truth TtCSF:", csf_data[1])
            #     print("truth TpCSF:", csf_data[2])
            #     print("truth TCSF:", csf_data[1] - csf_data[2])
            self.y["APET"] = self.y["APET"] + [np.mean(pet_data_a) ]
            self.y["TPET"] = self.y["TPET"] + [np.mean(pet_data_t) ]
            self.y["NPET"] = self.y["NPET"] + [np.mean(pet_data_n)]

            self.y["ACSF"] = self.y["ACSF"] + [csf_data[0]] 
            self.y["TtCSF"] = self.y["TtCSF"] + [csf_data[1]]
            self.y["TpCSF"] = self.y["TpCSF"] + [csf_data[2]]
            self.y["TCSF"] = self.y["TCSF"] + [csf_data[1] - csf_data[2]]
        for one_key in ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]:
            self.y[one_key] = np.asarray(self.y[one_key])


class ADSolver:
    def __init__(self, class_name, const_truth=None):
        self.n = 1  # Config.N_dim
        self.L = Config.L
        #        self.t = np.linspace(0, 10 - 0.1, 100)
        self.T = 9.01
        self.T_unit = 0.01
        self.t = np.linspace(0.0, self.T - self.T_unit, int(self.T / self.T_unit))  # expand time
        self.class_name = class_name
        self.const_truth = const_truth
        self.y0 = Start(class_name).all
        # print("ODE size: {}".format(self.y0.shape))

        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]

        # print("output has {} curves".format(len(self.output)))
        self.output_names = ["$A_{PET}$", "$T_{PET}$", "$N_{PET}$", "$A_{CSF}$", "$T_{pCSF}$", "$T_{CSF}$",
                             "$T_{tCSF}$"]
        self.output_names_rest = ["$A_{m} Avg$", "$T_{m} Avg$", "$A_{o} Avg$", "$T_{o} Avg$", "$T_{p} Avg$"]
        self.colors = ["red", "green", "blue", "cyan", "orange", "purple", "brown", "gray", "olive"]
        self.y = None
        self.output = None
        self.params = None
        self.tol = 1e-4
        # print("atol = rtol = {}".format(self.tol))

    def step(self, _params=None):
        if _params is not None:
            self.params = np.asarray(_params)
        else:
            self.params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
            print("Params is not given. Using the initial params instead to simulate ...")
        self.y = odeint(self.pend, self.y0, self.t, rtol=self.tol, atol=self.tol)
        self.output = self.get_output()

    def get_output(self):
        Am = self.y[:, 0: self.n]
        Ao = self.y[:, self.n: self.n * 2]
        Af = self.y[:, self.n * 2: self.n * 3]
        ACSF = self.y[:, self.n * 3: self.n * 3 + 1]
        Tm = self.y[:, self.n * 3 + 1: self.n * 4 + 1]
        Tp = self.y[:, self.n * 4 + 1: self.n * 5 + 1]
        To = self.y[:, self.n * 5 + 1: self.n * 6 + 1]
        Tf = self.y[:, self.n * 6 + 1: self.n * 7 + 1]
        TCSF = self.y[:, self.n * 7 + 1: self.n * 7 + 2]
        TpCSF = self.y[:, self.n * 7 + 2: self.n * 7 + 3]
        N = self.y[:, self.n * 7 + 3: self.n * 8 + 3]

        ACSF = np.expand_dims(ACSF[:, 0], axis=0)  # np.expand_dims(k_sA * np.sum(Am, axis=1), axis=0)
        TCSF = np.expand_dims(TCSF[:, 0], axis=0)  # np.expand_dims(k_sT * np.sum(Tm, axis=1), axis=0)
        TpCSF = np.expand_dims(TpCSF[:, 0], axis=0)  # np.expand_dims(k_sTp * np.sum(Tp, axis=1), axis=0)
        APET = np.expand_dims(np.mean(np.swapaxes(Af, 0, 1), axis=0), axis=0)
        TPET = np.expand_dims(np.mean(np.swapaxes(Tf, 0, 1), axis=0), axis=0)
        NPET = np.expand_dims(np.mean(np.swapaxes(N, 0, 1), axis=0), axis=0)
        TtCSF = TpCSF + TCSF
        # print("APET[0]:", APET[0][0])
        # print("TPET[0]:", TPET[0][0])
        # print("NPET[0]:", NPET[0][0])
        # print("ACSF[0]:", ACSF[0][0])
        # print("TCSF[0]:", TCSF[0][0])
        # print("TpCSF[0]:", TpCSF[0][0])
        # print("TtCSF[0]:", TtCSF[0][0])

        Am_avg = np.expand_dims(np.mean(Am, axis=1), axis=0)
        Tm_avg = np.expand_dims(np.mean(Tm, axis=1), axis=0)
        Ao_avg = np.expand_dims(np.mean(Ao, axis=1), axis=0)
        To_avg = np.expand_dims(np.mean(To, axis=1), axis=0)
        Tp_avg = np.expand_dims(np.mean(Tp, axis=1), axis=0)

        # APET_average = np.expand_dims(np.mean(APET, axis=0), axis=0)
        # TPET_average = np.expand_dims(np.mean(TPET, axis=0), axis=0)
        # NPET_average = np.expand_dims(np.mean(NPET, axis=0), axis=0)
        # return [APET, TPET, NPET, ACSF, TpCSF, TCSF, TtCSF, Ao_sum, To_sum]
        return [APET, TPET, NPET, ACSF, TpCSF, TCSF, TtCSF, Am_avg, Tm_avg, Ao_avg, To_avg, Tp_avg]

    def pend(self, y, t):
        # mt.time_start()
        Am = y[0: self.n]
        Ao = y[self.n: self.n * 2]
        Af = y[self.n * 2: self.n * 3]
        ACSF = y[self.n * 3: self.n * 3 + 1]
        Tm = y[self.n * 3 + 1: self.n * 4 + 1]
        Tp = y[self.n * 4 + 1: self.n * 5 + 1]
        To = y[self.n * 5 + 1: self.n * 6 + 1]
        Tf = y[self.n * 6 + 1: self.n * 7 + 1]
        TCSF = y[self.n * 7 + 1: self.n * 7 + 2]
        TpCSF = y[self.n * 7 + 2: self.n * 7 + 3]
        N = y[self.n * 7 + 3: self.n * 8 + 3]


        k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp \
            = iter(self.params)

        # n_TA = 2.0
        # n_cA = 4.0
        # n_AT = 1.0
        # n_cT = 1.0
        # n_cTp = 4.0
        # n_cTo = 1.0
        # n_AN = 2.0
        # n_TN = 2.0
        # n_a1A = 2.0
        # n_a2A = 1.0
        # n_a1T = 1.0
        # n_a2T = 2.0
        # n_a1Tp = 2.0

        d_Am = 1.0
        d_Ao = 1.0
        d_Tm = 1.0
        d_Tp = 1.0
        d_To = 1.0

        sum_func = np.sum
        matmul_func = my_matmul  # np.matmul
        offset = 1e-18


        Am_ = k_p1Am + k_p2Am * 1.0 / (numpy_safe_pow(K_mTA, n_TA) / numpy_safe_pow(To, n_TA) + 1.0) - k_dAm * Am - n_a1A * k_a1A * (
                     numpy_safe_pow(Am, n_a1A)) - n_a2A * k_a2A * Af * numpy_safe_pow(Am, n_a2A) + (
                           n_a1A + n_a2A) * k_diA * Ao - n_cA * k_cA * (
                           numpy_safe_pow(Am, n_cA)) * Ao - k_sA * Am + d_Am * matmul_func(self.L, Am)

        Ao_ = - k_dAo * Ao + k_a1A * numpy_safe_pow(Am, n_a1A) + k_a2A * Af * numpy_safe_pow(Am, n_a2A) - k_diA * Ao - k_cA * numpy_safe_pow(Am, n_cA) * Ao + d_Ao * matmul_func(self.L,
                                                                                                               Ao)
        Af_ = k_cA * numpy_safe_pow(Am, n_cA) * Ao

        ACSF_ = k_sA * sum_func(Am) - k_yA * ACSF

        Tm_ = k_pTm - k_dTm * Tm - (
                    k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao, n_AT) + 1.0)) * Tm + k_deph * Tp - n_a1T * k_a1T * numpy_safe_pow(
                          Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
                          1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (n_a1T + n_a2T) * k_diT * To - n_cT * k_cT * numpy_safe_pow(
                          Tm, n_cT) * (numpy_safe_pow(Tp,n_cTp)) * To - k_sT * Tm + d_Tm * matmul_func(self.L, Tm)
        Tp_ = -k_dTp * Tp + (
                    k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao, n_AT) + 1.0)) * Tm - k_deph * Tp - n_a1Tp * k_a1T * (
                          numpy_safe_pow(Tm, n_a1T)) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
                          1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (n_a1Tp + n_a2T) * k_diT * To - n_cTp * k_cT * numpy_safe_pow(
                          Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * To - k_sTp * Tp + d_Tp * matmul_func(self.L, Tp)

        To_ = - k_dTo * To + k_a1T * numpy_safe_pow(Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) + k_a2T * Tf * 1.0 / (
                    1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) - k_diT * To - k_cT * numpy_safe_pow(Tm, n_cT) * (
                          numpy_safe_pow(Tp, n_cTp)) * To + d_To * matmul_func(self.L, To)

        Tf_ = k_cT * numpy_safe_pow(Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * numpy_safe_pow(To, n_cTo)

        TCSF_ = k_sT * sum_func(Tm) - k_yT * TCSF

        TpCSF_ = k_sTp * sum_func(Tp) - k_yTp * TpCSF

        N_ = k_AN * 1.0 / (numpy_safe_pow(K_mAN, n_AN) / numpy_safe_pow((Ao + Af), n_AN) + 1.0) + k_TN * 1.0 / (
                    numpy_safe_pow(K_mTN, n_TN) / numpy_safe_pow((To + Tf), n_TN) + 1.0)

       
        dy = np.concatenate([Am_, Ao_, Af_, ACSF_, Tm_, Tp_, To_, Tf_, TCSF_, TpCSF_, N_])
        # # print(dy.shape)
        # mt.time_end()
        return dy

    def draw(self, save_flag=True, time_string="test"):

        folder_path = "figure/{}/".format(time_string)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path_target = os.path.join(folder_path, "target.png")
        save_path_rest = os.path.join(folder_path, "rest.png")
        m = MultiSubplotDraw(row=3, col=3, fig_size=(24, 18), tight_layout_flag=True, show_flag=True,
                             save_flag=save_flag,
                             save_path=save_path_target, save_dpi=400)
        for name, data, color, line_string in zip(self.output_names, self.output[:len(self.output_names)],
                                                  self.colors[:len(self.output_names)], self.lines):
            ax = m.add_subplot(
                y_lists=data,
                x_list=self.t,
                color_list=[color],
                line_style_list=["solid"],
                fig_title=name,
                legend_list=[name],
                line_width=2,
            )
            if self.const_truth:
                x = self.const_truth.x
                y = self.const_truth.y[line_string]
                # print(len(x), len(y))
                ax2 = ax.twinx()
                ax2.set_ylabel("truth points val", fontsize=15)
                if line_string in ["TCSF", "NPET"]:
                    ax2.scatter(x=x, y=y, s=100, facecolor="blue", alpha=0.5, marker="d",
                                edgecolors='black', linewidths=1,
                                zorder=10)
                else:
                    ax2.scatter(x=x[[0, 2, 3, 4]], y=y[[0, 2, 3, 4]], s=100, facecolor="red", alpha=0.5, marker="o", edgecolors='black', linewidths=1,
                                zorder=10)
                    ax2.scatter(x=x[1], y=y[1], s=100, facecolor="blue", alpha=0.5, marker="d", edgecolors='black', linewidths=1,
                                zorder=10)
                ax2.tick_params(axis='y', labelcolor="red", labelsize=15)

        m.add_subplot(
            y_lists=np.concatenate(self.output[:len(self.output_names)], axis=0),
            x_list=self.t,
            color_list=self.colors[:len(self.output_names)],
            line_style_list=["solid"] * len(self.output_names),
            fig_title="Seven Target Curves",
            legend_list=self.output_names,
            line_width=2,
        )
        m.draw()
        print("Save flag: {}. Target figure is saved to {}".format(save_flag, save_path_target))

        m = MultiSubplotDraw(row=2, col=3, fig_size=(24, 12), tight_layout_flag=True, show_flag=True,
                             save_flag=save_flag,
                             save_path=save_path_rest, save_dpi=400)
        for name, data, color in zip(self.output_names_rest, self.output[-len(self.output_names_rest):],
                                     self.colors[:len(self.output_names_rest)]):
            m.add_subplot(
                y_lists=data,
                x_list=self.t,
                color_list=[color],
                line_style_list=["solid"],
                fig_title=name,
                legend_list=[name],
                line_width=2,
            )
        m.add_subplot(
            y_lists=np.concatenate(self.output[-len(self.output_names_rest):], axis=0),
            x_list=self.t,
            color_list=self.colors[:len(self.output_names_rest)],
            line_style_list=["solid"] * len(self.output_names_rest),
            fig_title="Rest Curves",
            legend_list=self.output_names_rest,
            line_width=2,
        )
        # plt.suptitle("{} Class ODE Solution".format(self.class_name), fontsize=40)
        m.draw()
        print("Save flag: {}. Rest figure is saved to {}".format(save_flag, save_path_rest))


def loss_func(params, ct):
    # print("calling loss_func..")
    truth = ADSolver("CN")
    truth.step(params)
    targets = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
    record = np.zeros(len(targets))
    for i, one_target in enumerate(targets):
        target_points = np.asarray(ct.y[one_target])[[0, 2, 3, 4]]
        t_fixed = np.asarray([0, 6, 8, 9])
        index_fixed = (t_fixed / truth.T_unit).astype(int)
        predict_points = np.asarray(truth.output[i][0][index_fixed])
        # print("target_points:", target_points.shape)
        # print("predict_points:", predict_points.shape)

        target_points_scaled = (target_points - np.min(target_points)) / (np.max(target_points) - np.min(target_points))
        # print("[loss_func] ({}) predict_points: {}".format(one_target, predict_points))

        if np.max(predict_points) - np.min(predict_points) <= 1e-15:
            predict_points_scaled = np.zeros(len(target_points_scaled))
        else:
            predict_points_scaled = (predict_points - np.min(predict_points)) / (np.max(predict_points) - np.min(predict_points))
        # try:
        #     # assert 0.0 not in list(np.max(predict_points) - np.min(predict_points))
        #     print(predict_points - np.min(predict_points), np.max(predict_points) - np.min(predict_points))
        #     predict_points_scaled = (predict_points - np.min(predict_points)) / (np.max(predict_points) - np.min(predict_points))
        # except Exception as e:
        #     print(e)
        #     predict_points_scaled = np.zeros(len(target_points_scaled))

        # record[i] = np.mean(((predict_points - target_points) / target_points) ** 2)
        record[i] = np.mean((predict_points_scaled - target_points_scaled) ** 2)
    record = record[[0, 1, 3, 4, 6]]
    return record  # remove NPET here


# MyTime is only for debugging
class MyTime:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.tmp = None

    def time_start(self):
        ts = time.time()
        # if self.count > 0:
        #     self.sum += (ts - self.tmp)
        # self.count += 1
        self.tmp = ts

    def time_end(self):
        ts = time.time()
        self.sum += (ts - self.tmp)
        self.count += 1
        self.tmp = None

    def print(self):
        print("count = {}; total time = {} s; avg time = {} s".format(self.count, self.sum, self.sum / self.count))


def run(params=None):
    time_string = get_now_string()
    print("Time String (as folder name): {}".format(time_string))

    class_name = "CN"
    ct = ConstTruth(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/"
    )
    truth = ADSolver(class_name, ct)
    truth.step(params)
    truth.draw(time_string=time_string)


if __name__ == "__main__":
    # run()
    ct = ConstTruth(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/"
    )
    # record1 = loss_func(np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)]), ct)
    # print(record1)
    # print("hhhh")
#    params = np.load("saves/params_20221206_093250.npy")
#    params = np.load("saves/params_20221129_201557.npy")
    params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
     
    # print(len(params))
    # np.save("saves/params_default_46.npy", params)
    
    # p0 = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
    # record1 = loss_func(p0, ct)
    # print(record1)
    # run(p0)


    # p = np.load("saves/params_20221103_090002.npy")
    # record2 = loss_func(p, ct)
    # print(record2)
    # mt = MyTime()
    run(params)
    # mt.print()