import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import time
import json
from datetime import datetime
import argparse
from scipy.integrate import odeint
from tqdm import tqdm

# from parameters import *
from config import Start, Config
from utils import MultiSubplotDraw, ColorCandidate

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
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def numpy_safe_pow(x, n):
    return np.sign(x) * (np.abs(x)) ** n


def my_matmul(m, x, expand=160):
    x = np.ones(expand) * x[0]
    return np.mean(np.matmul(m, x)).reshape(1)


class ConstTruth:
    def __init__(self, **params):
        assert "csf_folder_path" in params and "pet_folder_path" in params, "please provide the save folder paths"
        assert "dataset" in params
        assert "start" in params
        assert "option" in params
        assert "tcsf_scaler" in params
        self.params = params
        csf_folder_path, pet_folder_path = params["csf_folder_path"], params["pet_folder_path"]
        label_list = LABEL_LIST  # [[0, 2, 3, 4]]  # skip the second nodes (SMC)
        self.class_num = len(label_list)
        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
        self.y = dict()
        self.x = dict()
        self.y_original = dict()
        self.x_original = dict()
        self.increase_rate = np.zeros(7)
        assert params["start"] in ["ranged", "fixed"] or "ranged" in params["start"]
        assert params["dataset"] in ["all", "chosen_0", "rebuild"]
        if "x" not in params:
            if params["start"] in ["ranged", "rebuild"] or "ranged" in params["start"]:
                self.x_all = np.asarray([3, 6, 9, 11, 12])
            else:
                self.x_all = np.asarray([0, 3, 6, 8, 9])
        else:
            self.x_all = np.asarray(params.get("x"))
        if params["dataset"] == "rebuild":
            with open("data/rebuild_truth.pkl", "rb") as f:
                rebuild_dic = pickle.load(f)
            for one_line in self.lines:
                if one_line in rebuild_dic:
                    self.y[one_line] = rebuild_dic[one_line]["y"]
                    self.x[one_line] = rebuild_dic[one_line]["x"]
                else:
                    self.y[one_line] = np.asarray([])
                    self.x[one_line] = np.asarray([])
            with open("data/plots.json", "r") as f:
                plot_json = json.load(f)
            for one_line in self.lines:
                self.y_original[one_line] = np.asarray(plot_json["truth_plot"][one_line]["y"])
                self.x_original[one_line] = np.asarray(plot_json["truth_plot"][one_line]["x"])
            return


        for one_line in self.lines:
            self.y[one_line] = []
            self.x[one_line] = self.x_all
        for i, class_name in enumerate(label_list):
            csf_data = np.load(os.path.join(csf_folder_path, "CSF_{}.npy".format(class_name)))
            pet_data_a = np.load(os.path.join(pet_folder_path, "PET-A_{}.npy".format(class_name)))
            pet_data_t = np.load(os.path.join(pet_folder_path, "PET-T_{}.npy".format(class_name)))
            pet_data_n = np.load(os.path.join(pet_folder_path, "PET-N_{}.npy".format(class_name)))
            self.y["APET"] = self.y["APET"] + [np.mean(pet_data_a)]
            self.y["TPET"] = self.y["TPET"] + [np.mean(pet_data_t)]
            self.y["NPET"] = self.y["NPET"] + [np.mean(pet_data_n)]

            self.y["ACSF"] = self.y["ACSF"] + [csf_data[0]] 
            self.y["TtCSF"] = self.y["TtCSF"] + [csf_data[1]]
            self.y["TpCSF"] = self.y["TpCSF"] + [csf_data[2]]
            self.y["TCSF"] = self.y["TCSF"] + [csf_data[1] - csf_data[2]]
        for i, one_key in enumerate(self.lines):
            self.y[one_key] = np.asarray(self.y[one_key])
            self.increase_rate[i] = (self.y[one_key][-1] - self.y[one_key][0]) / self.y[one_key][0]
        print("increse rate: {}".format(self.increase_rate))
        # self.y["NPET"] = 2.0 - self.y["NPET"]  # 1.0 - (self.y["NPET"] - np.min(self.y["NPET"])) / (np.max(self.y["NPET"]) - np.min(self.y["NPET"]))

        if params["dataset"] == "chosen_0":
            for one_key in ["NPET"]:
                self.y[one_key] = self.y[one_key]  # [[]]
                self.x[one_key] = self.x[one_key]  # [[]]
            for one_key in ["ACSF", "TCSF", "TtCSF"]:
                self.y[one_key] = self.y[one_key][[0, 2, 3, 4]]
                self.x[one_key] = self.x[one_key][[0, 2, 3, 4]]
            for one_key in ["TpCSF"]:
                self.y[one_key] = self.y[one_key][[0, 2, 4]]
                self.x[one_key] = self.x[one_key][[0, 2, 4]]
        else:
            for one_key in ["NPET"]:
                self.y[one_key] = self.y[one_key][[]]
                self.x[one_key] = self.x[one_key][[]]
            pass


class ADSolver:
    def __init__(self, class_name, const_truth=None):
        self.n = 1  # Config.N_dim
        self.L = Config.L
        #        self.t = np.linspace(0, 10 - 0.1, 100)
        self.T = 12.01
        self.T_unit = 0.01
        self.t = np.linspace(0.0, self.T - self.T_unit, int(self.T / self.T_unit))  # expand time
        self.class_name = class_name
        self.const_truth = const_truth
        self.y0 = Start(class_name, tcsf_scaler=self.const_truth.params["tcsf_scaler"]).all
        # print("ODE size: {}".format(self.y0.shape))

        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]

        # print("output has {} curves".format(len(self.output)))
        self.output_names = ["$A_{PET}$", "$T_{PET}$", "$N_{PET}$", "$A_{CSF}$", "$T_{pCSF}$", "$T_{CSF}$",
                             "$T_{tCSF}$"]
        self.output_names_rest = ["$A_{m} Avg$", "$T_{m} Avg$", "$A_{o} Avg$", "$T_{o} Avg$", "$T_{p} Avg$"]
        self.n_color = 12
        self.colors = ColorCandidate().get_color_list(self.n_color, light_rate=0.5)
        self.y = None
        self.output = None
        self.params = None
        self.starts_weights = None
        self.truth_ylim = dict()
        self.predict_ylim = dict()
        self.tol = 1e-4
        # print("atol = rtol = {}".format(self.tol))

    def step(self, _params=None, _starts_weights=None):
        if _params is not None:
            self.params = np.asarray(_params)
            self.starts_weights = np.asarray(_starts_weights)
        else:
            self.params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
            self.starts_weights = np.asarray([STARTS_WEIGHTS[i]["init"] for i in range(STARTS_NUM)])
            print("Params & starts_weights are not given. Using the initial value instead to simulate ...")
        self.y0 = self.y0 * self.starts_weights
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

        NPET = 0.9 - (NPET - np.min(NPET)) / (np.max(NPET) - np.min(NPET)) * 0.1  # 200.0 - NPET

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

        # k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2 \
        #     = iter(self.params)

        k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp, n_a2Tp, k_acsf, k_tcsf \
            = iter(self.params)
#        k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp, K_cA \
#            = iter(self.params)

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

        d_Am = 0.0  # 5.0
        d_Ao = 0.0
        d_Tm = 0.0  # 5.0
        d_Tp = 0.0  # 5.0
        d_To = 0.0

        sum_func = np.sum
        matmul_func = my_matmul  # np.matmul
        offset = 1e-18

        Am_ = k_p1Am + k_p2Am * 1.0 / (
                numpy_safe_pow(K_mTA, n_TA) / numpy_safe_pow(To, n_TA) + 1.0) - k_dAm * Am - n_a1A * k_a1A * (
                  numpy_safe_pow(Am, n_a1A)) - n_a2A * k_a2A * Af * numpy_safe_pow(Am, n_a2A) + (
                      n_a1A + n_a2A) * k_diA * Ao - n_cA * k_cA * (
                  numpy_safe_pow(Am, n_cA)) * Ao - k_sA * Am + d_Am * matmul_func(self.L, Am)

        Ao_ = - k_dAo * Ao + k_a1A * numpy_safe_pow(Am, n_a1A) + k_a2A * Af * numpy_safe_pow(Am,
                                                                                             n_a2A) - k_diA * Ao - k_cA * numpy_safe_pow(
            Am, n_cA) * Ao + d_Ao * matmul_func(self.L,
                                                Ao)

        Af_ = k_cA * numpy_safe_pow(Am, n_cA) * Ao

        ACSF_ = k_sA * sum_func(Am) - k_yA * ACSF
        assert self.const_truth.params["option"] in ["option1", "option2"] or "option1" in self.const_truth.params["option"]
        if "option1" in self.const_truth.params["option"]:
            #####0222######
            Tm_ = k_pTm * Am ** 2 - k_dTm * Tm - (
                    k_ph1 + k_ph2 * Ao
            ) * Tm + k_deph * Tp - n_a1T * k_a1T * numpy_safe_pow(
                Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
                          1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (
                          n_a1T + n_a2T) * k_diT * To - n_cT * k_cT * numpy_safe_pow(
                Tm, n_cT) * (numpy_safe_pow(Tp, n_cTp)) * To - k_sT * Tm + d_Tm * matmul_func(self.L, Tm)

            Tp_ = -k_dTp * Tp + (
                    k_ph1 + k_ph2 * Ao) * Tm - k_deph * Tp - n_a1Tp * k_a1T * (
                      numpy_safe_pow(Tm, n_a1T)) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
                          1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (
                          n_a1Tp + n_a2T) * k_diT * To - n_cTp * k_cT * numpy_safe_pow(
                Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * To - k_sTp * Tp + d_Tp * matmul_func(self.L, Tp)

            # Tp_ = -k_dTp * Tp + (
            #         k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao,
            #                                                                             n_AT) + 1.0)) * Tm - k_deph * Tp - n_a1Tp * k_a1T * (
            #           numpy_safe_pow(Tm, n_a1T)) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
            #               1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (
            #                   n_a1Tp + n_a2T) * k_diT * To - n_cTp * k_cT * numpy_safe_pow(
            #     Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * To - k_sTp * Tp + d_Tp * matmul_func(self.L, Tp)

            To_ = - k_dTo * To + k_a1T * numpy_safe_pow(Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) + k_a2T * Tf * 1.0 / (
                    1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp),
                                                                        n_a2T)) - k_diT * To - k_cT * numpy_safe_pow(Tm,
                                                                                                                     n_cT) * (
                      numpy_safe_pow(Tp, n_cTp)) * To + d_To * matmul_func(self.L, To)

        else:  # option2
            Tm_ = k_pTm - k_dTm * Tm - (k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao,
                                                                                                            n_AT) + 1.0)) * Tm + k_deph * Tp - n_a1T * k_a1T * numpy_safe_pow(
                Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * numpy_safe_pow(Tm,
                                                                                              n_a2T) * numpy_safe_pow(
                Tp, n_a2Tp) + (n_a1T + n_a2T) * k_diT * To - n_cT * k_cT * numpy_safe_pow(
                Tm, n_cT) * (numpy_safe_pow(Tp, n_cTp)) * To - k_sT * Tm + d_Tm * matmul_func(self.L, Tm)

            Tp_ = -k_dTp * Tp + (k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao,
                                                                                                     n_AT) + 1.0)) * Tm - k_deph * Tp - n_a1Tp * k_a1T * (
                      numpy_safe_pow(Tm, n_a1T)) * numpy_safe_pow(Tp, n_a1Tp) - n_a2Tp * k_a2T * Tf * numpy_safe_pow(Tm,
                                                                                                                     n_a2T) * numpy_safe_pow(
                Tp, n_a2Tp) + (n_a1Tp + n_a2Tp) * k_diT * To - n_cTp * k_cT * numpy_safe_pow(
                Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * To - k_sTp * Tp + d_Tp * matmul_func(self.L, Tp)

            To_ = - k_dTo * To + k_a1T * numpy_safe_pow(Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) + k_a2T * Tf * 1.0 / (
                    1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp),
                                                                        n_a2T)) - k_diT * To - k_cT * numpy_safe_pow(Tm,
                                                                                                                     n_cT) * (
                      numpy_safe_pow(Tp, n_cTp)) * To + d_To * matmul_func(self.L, To)

        #         Tf_ = k_cT * numpy_safe_pow(Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * numpy_safe_pow(To, n_cTo)
        #
        #         TCSF_ = k_sT * sum_func((Tm ** 2) / (Tm ** 2 + k_tcsf ** 2)) - k_yT * TCSF  # v0118 sqrt !
        # #        TCSF_ = k_sT * sum_func(Tm**2) - k_yT * TCSF
        #         TpCSF_ = k_sTp * sum_func((Tp ** 2) / (Tp ** 2 + k_tcsf ** 2)) - k_yTp * TpCSF  # v0118 sqrt !
        #
        #         N_ = k_AN * 1.0 / (numpy_safe_pow(K_mAN, n_AN) / numpy_safe_pow((Ao + Af), n_AN) + 1.0) + k_TN * 1.0 / (
        #                     numpy_safe_pow(K_mTN, n_TN) / numpy_safe_pow((To + Tf), n_TN) + 1.0)

        Tf_ = k_cT * numpy_safe_pow(Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * numpy_safe_pow(To, n_cTo)

        TCSF_ = k_sT * sum_func(Tm) - k_yT * TCSF

        TpCSF_ = k_sTp * sum_func(Tp) - k_yTp * TpCSF

        N_ = k_AN * 1.0 / (numpy_safe_pow(K_mAN, n_AN) / numpy_safe_pow((Ao + Af), n_AN) + 1.0) + k_TN * 1.0 / (
                numpy_safe_pow(K_mTN, n_TN) / numpy_safe_pow((To + Tf), n_TN) + 1.0)
        dy = np.concatenate([Am_, Ao_, Af_, ACSF_, Tm_, Tp_, To_, Tf_, TCSF_, TpCSF_, N_])
        # # print(dy.shape)
        # mt.time_end()
        return dy

    def draw(self, opt, save_flag=True, time_string="test", given_loss=None):

        # folder_path = "figure/{}/".format(time_string)
        folder_path = "figure/{0}{1}/".format(
            "{}_".format(opt.model_name) if opt.model_name != "none" else "",
            time_string)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path_target = os.path.join(folder_path, "figure_target_{}.png".format(time_string))
        save_path_rest = os.path.join(folder_path, "figure_rest_{}.png".format(time_string))
        m = MultiSubplotDraw(row=3, col=3, fig_size=(24, 18), tight_layout_flag=True, show_flag=False,
                             save_flag=save_flag,
                             save_path=save_path_target, save_dpi=200)
        for i, (name, data, color, line_string) in enumerate(zip(self.output_names, self.output[:len(self.output_names)], self.colors[:len(self.output_names)], self.lines)):
            y_lists = data[:, :-int(3.0 / self.T_unit)] if self.const_truth.params["start"] == "fixed" else data
            ax = m.add_subplot(
                y_lists=y_lists,
                x_list=self.t,
                color_list=[color],
                line_style_list=["solid"],
                fig_title="{}{}".format(name, " (loss={0:.3f}, rate={1:.3f}, tar_rate={2:.3f})".format(given_loss[i], (data[0][1200] - data[0][300]) / data[0][300], self.const_truth.increase_rate[i]) if given_loss is not None else ""),
                legend_list=[name],
                line_width=2,
            )
            # ax.scatter(x=self.const_truth.x[line_string], y=y_lists[0][(self.const_truth.x[line_string] / self.T_unit).astype(int)], s=100, facecolor=self.colors[i + self.n_color], alpha=0.8, marker="x", linewidths=3, zorder=10)  # x nodes
            # ax.set_ylim([np.min(data[0]), np.max(data[0])])
            self.predict_ylim[self.lines[i]] = list(ax.get_ylim())

            if self.const_truth:
                x = self.const_truth.x[line_string]
                y = self.const_truth.y[line_string]
                # print(len(x), len(y))
                ax2 = ax.twinx()
                ax2.set_ylabel("truth points val", fontsize=15)
                ax2.scatter(x=x, y=y, s=100, facecolor='black', alpha=0.8, marker="o", edgecolors='black', linewidths=1, zorder=10)
                if self.const_truth.params["dataset"] == "rebuild":
                    ax2.scatter(x=self.const_truth.x_original[line_string], y=self.const_truth.y_original[line_string], s=100, facecolor="none", marker="d", edgecolors='#00ff00', linewidths=3, zorder=10)
                # if line_string in ["NPET"]:
                #     ax2.scatter(x=x, y=y, s=100, facecolor="blue", alpha=0.5, marker="d",
                #                 edgecolors='black', linewidths=1,
                #                 zorder=10)
                # elif line_string in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:
                #     ax2.scatter(x=x[[0, 2, 3, 4]], y=y[[0, 2, 3, 4]], s=100, facecolor="red", alpha=0.5, marker="o", edgecolors='black', linewidths=1,
                #                 zorder=10)
                #     ax2.scatter(x=x[1], y=y[1], s=100, facecolor="blue", alpha=0.5, marker="d", edgecolors='black', linewidths=1,
                #                 zorder=10)
                # else:
                #     ax2.scatter(x=x, y=y, s=100, facecolor="red", alpha=0.5, marker="o",
                #                 edgecolors='black', linewidths=1,
                #                 zorder=10)
                ax2.tick_params(axis='y', labelcolor="red", labelsize=15)
                if "ranged" in self.const_truth.params["start"]:
                    ylim_bottom, ylim_top = ax2.get_ylim()
                    index_fixed = (x / self.T_unit).astype(int)
                    curve_data = data[0][index_fixed]
                    if line_string in ["ACSF", "NPET"]:
                        ax2.set_ylim([ylim_bottom, ylim_bottom + (ylim_top - ylim_bottom) / (np.max(curve_data) - np.min(curve_data)) * (np.max(data[0]) - np.min(curve_data))])
                        # ax2.set_ylim([ylim_bottom, ylim_bottom + (ylim_top - ylim_bottom) / (data[0][int(x[0] / self.T_unit)] - data[0][int(x[-1] / self.T_unit)]) * (data[0][0] - data[0][int(x[-1] / self.T_unit)])])
                    elif line_string in ["APET", "TPET", "TpCSF", "TCSF", "TtCSF"]:
                        ax2.set_ylim([ylim_top - (ylim_top - ylim_bottom) / (np.max(curve_data) - np.min(curve_data)) * (np.max(curve_data) - np.min(data[0])), ylim_top])
                        # ax2.set_ylim([ylim_top - (ylim_top - ylim_bottom) / (data[0][int(x[0] / self.T_unit)] - data[0][int(x[-1] / self.T_unit)]) * (data[0][0] - data[0][int(x[-1] / self.T_unit)]), ylim_top])
                self.truth_ylim[self.lines[i]] = list(ax2.get_ylim())


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

        m = MultiSubplotDraw(row=2, col=3, fig_size=(24, 12), tight_layout_flag=True, show_flag=False,
                             save_flag=save_flag,
                             save_path=save_path_rest, save_dpi=200)
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


def f_csf_rate(x, thr=1.7052845384621318, tol=0.2, p=1.0):
    return max((x - thr * (1 + tol)) * p, (thr * (1 - tol) - x) * p, 0)


def limit_rate(x, thr=0.05, tol=2.0, p=1.0):
    if thr > 0:
        ub = thr * tol
        lb = thr / tol
        return max((x - ub) * p, (lb - x) * p,  0)
    else:
        ub = thr / tol
        lb = thr * tol
        return max((x - ub) * p, (lb - x) * p, 0)


def loss_func(params, starts_weight, ct):
    # print("calling loss_func..")
    truth = ADSolver("CN", ct)
    truth.step(params, starts_weight)
    targets = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
    record = np.zeros(len(targets))

    gradient_check = []
    for i, one_target in enumerate(targets):
        target_points = np.asarray(ct.y[one_target])
        if len(target_points) == 0:
            record[i] = 0.0
            continue
        t_fixed = np.asarray(ct.x[one_target])
        # if one_target in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:  # skip the second points for all CSFs
        #     target_points = target_points[[0, 2, 3, 4]]
        #     t_fixed = t_fixed[[0, 2, 3, 4]]
        index_fixed = (t_fixed / truth.T_unit).astype(int)
        predict_points = np.asarray(truth.output[i][0][index_fixed])

        gradient_dy = np.gradient(truth.output[i][0], truth.t)
        if i not in [2, 3]:  # not NPET, ACSF
            gradient_penalty = np.mean(np.abs(gradient_dy) - gradient_dy)
        else:
            gradient_penalty = np.mean(np.abs(-gradient_dy) + gradient_dy)
        gradient_check.append(gradient_penalty * 1e4)
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
        if i == 2:
            record[i] /= 1.0
        elif i in [4, 5, 6]:
            record[i] *= 10.0
    # record = record[[0, 1, 3, 4, 5, 6]]
    # print("gradient_check: {} ({})".format(sum(gradient_check), gradient_check))
    csf_rate = \
        f_csf_rate(np.max(truth.output[3][0]) / np.max(truth.output[6][0]), thr=1.7052845384621318, tol=0.2, p=100.0) + \
        f_csf_rate(np.max(truth.output[4][0]) / np.max(truth.output[5][0]), thr=0.7142857142857143, tol=0.2, p=100.0) + \
        f_csf_rate(np.max(truth.output[0][0]) / np.max(truth.output[1][0]), thr=1.7, tol=0.2, p=100.0)
    csf_rate += \
        limit_rate((truth.output[0][0][1200] - truth.output[0][0][300]) / truth.output[0][0][300], thr=ct.increase_rate[0], tol=2.0, p=10.0) + \
        limit_rate((truth.output[1][0][1200] - truth.output[1][0][300]) / truth.output[1][0][300], thr=ct.increase_rate[1], tol=2.0, p=10.0) + \
        limit_rate((truth.output[3][0][1200] - truth.output[3][0][300]) / truth.output[3][0][300], thr=ct.increase_rate[3], tol=2.0, p=10.0) + \
        limit_rate((truth.output[4][0][1200] - truth.output[4][0][300]) / truth.output[4][0][300], thr=ct.increase_rate[4], tol=2.0, p=0.1) + \
        limit_rate((truth.output[5][0][1200] - truth.output[5][0][300]) / truth.output[5][0][300], thr=ct.increase_rate[5], tol=2.0, p=0.1) + \
        limit_rate((truth.output[6][0][1200] - truth.output[6][0][300]) / truth.output[6][0][300], thr=ct.increase_rate[6], tol=2.0, p=0.1)

    csf_rate += sum(gradient_check)
    # limit_rate((truth.output[2][0][1200] - truth.output[2][0][300]) / truth.output[2][0][300], thr=ct.increase_rate[2], tol=2.0, p=1.0) + \

    record *= 10.0

    return record, csf_rate  # remove NPET here


def transform_boundary(init_default_list, target_names, fix_values):
    output = []
    for i, item in enumerate(init_default_list):
        assert i == item.get("id")
        assert len(fix_values) == len(init_default_list)
        if item.get("name") in target_names:
            output.append(item)
        else:
            item["init"] = fix_values[i]
            item["ub"] = fix_values[i]
            item["lb"] = fix_values[i]
            output.append(item)
    return output


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


def run(params=None, starts=None, time_string=None, opt=None):
    if not time_string:
        time_string = get_now_string()
    print("Time String (as folder name): {}".format(time_string))

    class_name = "CN"
    if opt is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, choices=["chosen_0", "all", "rebuild"], help="dataset strategy")
        parser.add_argument("--start", type=str, choices=["ranged", "ranged*", "fixed"], help="start strategy")
        parser.add_argument("--generation", type=int, default=1000, help="generation, default: 1000")
        parser.add_argument("--pop_size", type=int, default=50, help="pop_size, default: 50")
        parser.add_argument("--model_name", default="none", type=str, help="model_name, can be any string")
        parser.add_argument("--option", type=str, default="option1", choices=["option1", "option1*", "option2"], help="option")
        parser.add_argument("--tcsf_scaler", type=float, default=0.3, help="tcsf_scaler, e.g., 0.3, 0.4, 0.5")
        parser.add_argument("--init_path", type=str, default=None, help="init_path")
        opt = parser.parse_args()
    ct = ConstTruth(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/",
        dataset=opt.dataset,
        start=opt.start,
        option=opt.option,
        tcsf_scaler=opt.tcsf_scaler,
    )
    truth = ADSolver(class_name, ct)
    truth.step(params, starts)
    loss, csf_rate_loss = loss_func(params, starts, ct)
    print("loss: {}".format(sum(loss) + csf_rate_loss))
    print("loss parts: {} csf match loss: {}".format(list(loss), csf_rate_loss))
    truth.draw(opt, time_string=time_string, given_loss=loss)
    return truth


if __name__ == "__main__":
    # ct = ConstTruth(
    #     csf_folder_path="data/CSF/",
    #     pet_folder_path="data/PET/",
    #     dataset="all"
    # )

    # full_params = np.load("saves/params_20230105_031544_255565.npy")
    # params = full_params[:PARAM_NUM]
    # starts = full_params[-STARTS_NUM:]
    # run(params, starts)

    # ct = ConstTruth(
    #     csf_folder_path="data/CSF/",
    #     pet_folder_path="data/PET/",
    #     dataset="chosen_0",
    #     start="ranged",
    #     option="option1",
    #     tcsf_scaler=0.3,
    # )


    parameters_full = np.load("saves/params_20230310_192938_379363.npy")
    print(len(parameters_full))
    PARAMS = transform_boundary(PARAMS, ["K_AN", "K_mAN", "n_AN", "K_TN", "K_mTN", "n_TN"], parameters_full[:49])
    print(json.dumps(PARAMS, indent=4))
    STARTS_WEIGHTS = transform_boundary(STARTS_WEIGHTS, ["N"], parameters_full[-11:])
    print(json.dumps(STARTS_WEIGHTS, indent=4))


