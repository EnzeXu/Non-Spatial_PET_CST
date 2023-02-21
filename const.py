import argparse
LABEL_LIST = ["CN", "SMC", "EMCI", "LMCI", "AD"]
LABEL_ID = {
    "CN": 0,
    "SMC": 1,
    "EMCI": 2,
    "LMCI": 3,
    "AD": 4
}
PARAM_NUM = 51 - 5 + 1 + 2  # n_a2Tp + ktcsf & tacsf
#PARAM_NUM = 51 - 4
UPFOLD = 10
LOWFOLD = 0.1
STARTS_NUM = 11
STARTS_NAME_LIST = ["Am", "Ao", "Af", "ACSF", "Tm", "Tp", "To", "Tf", "TCSF", "TpCSF", "N"]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["chosen_0", "all", "rebuild"], help="dataset strategy")
parser.add_argument("--start", type=str, choices=["ranged", "fixed"], help="start strategy")
parser.add_argument("--generation", type=int, default=1000, help="generation, default: 1000")
parser.add_argument("--pop_size", type=int, default=50, help="pop_size, default: 50")
parser.add_argument("--model_name", default="none", type=str, help="model_name, can be any string")
parser.add_argument("--option", type=str, choices=["option1", "option2"], help="option")
parser.add_argument("--tcsf_scaler", type=float, help="tcsf_scaler, e.g., 0.3, 0.4, 0.5")
opt = parser.parse_args()

assert opt.start in ["fixed", "ranged"]
if opt.start == "fixed":
    STARTS_WEIGHTS = [
        {
            "id": 0,
            "name": "Am",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 1,
            "name": "Ao",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 2,
            "name": "Af",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 3,
            "name": "ACSF",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 4,
            "name": "Tm",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 5,
            "name": "Tp",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 6,
            "name": "To",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 7,
            "name": "Tf",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 8,
            "name": "TCSF",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 9,
            "name": "TpCSF",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
        {
            "id": 10,
            "name": "N",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000,
        },
    ]
else:
    STARTS_WEIGHTS = [
        {
            "id": 0,
            "name": "Am",
            "init": 1.000,
            "lb": 1.000 - 0.3,
            "ub": 1.000 + 0.3,
        },
        {
            "id": 1,
            "name": "Ao",
            "init": 1.000,
            "lb": 1.000 - 0.3,
            "ub": 1.000 + 0.3,
        },
        {
            "id": 2,
            "name": "Af",
            "init": 1.000,
            "lb": 1.000 - 0.3,  # - 0.058705655763439606,
            "ub": 1.000,
        },
        {
            "id": 3,
            "name": "ACSF",
            "init": 1.000,
            "lb": 1.000,
            "ub": 1.000 + 0.3,  # + 0.14633390295437004,
        },
        {
            "id": 4,
            "name": "Tm",
            "init": 1.000,
            "lb": 1.000 - 0.3,
            "ub": 1.000 + 0.3,
        },
        {
            "id": 5,
            "name": "Tp",
            "init": 1.000,
            "lb": 1.000 - 0.3,
            "ub": 1.000 + 0.3,
        },
        {
            "id": 6,
            "name": "To",
            "init": 1.000,
            "lb": 1.000 - 0.3,
            "ub": 1.000 + 0.3,
        },
        {
            "id": 7,
            "name": "Tf",
            "init": 1.000,
            "lb": 1.000 - 0.3,  # - 0.1106672083840519,
            "ub": 1.000,
        },
        {
            "id": 8,
            "name": "TCSF",
            "init": 1.000,
            "lb": 1.000 - 0.3,  # - 0.3034193899069153,
            "ub": 1.000,
        },
        {
            "id": 9,
            "name": "TpCSF",
            "init": 1.000,
            "lb": 1.000 - 0.3,  # - 0.20798415143947105,
            "ub": 1.000,
        },
        {
            "id": 10,
            "name": "N",
            "init": 1.000,
            "lb": 1.000 - 0.3,  # - 0.01991284446769561,
            "ub": 1.000,
        },
    ]

# STARTS_WEIGHTS = [
#     {
#         "id": 0,
#         "name": "Am",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 1,
#         "name": "Ao",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 2,
#         "name": "Af",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 3,
#         "name": "ACSF",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 4,
#         "name": "Tm",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 5,
#         "name": "Tp",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 6,
#         "name": "To",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 7,
#         "name": "Tf",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 8,
#         "name": "TCSF",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 9,
#         "name": "TpCSF",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
#     {
#         "id": 10,
#         "name": "N",
#         "init": 1.000,
#         "lb": 0.70,
#         "ub": 1.30,
#     },
# ]

PARAM_NAME_LIST = ["k_p1Am", "k_p2Am", "k_dAm", "k_diA", "k_cA", "k_sA", "k_dAo", "k_yA", "k_pTm", "k_dTm", "k_ph1", "k_ph2", "k_deph", "k_diT", "k_cT", "k_sT", "k_dTp", "k_sTp", "k_dTo", "k_yT", "k_yTp", "k_AN", "k_TN", "k_a1A", "k_a2A", "k_a1T", "k_a2T", "K_mTA", "K_mAT", "K_mAN", "K_mTN", "K_mT2", "K_mA2", "n_TA", "n_cA", "n_AT", "n_cT", "n_cTp", "n_cTo", "n_AN", "n_TN", "n_a1A", "n_a2A", "n_a1T", "n_a2T", "n_a1Tp", "d_Am", "d_Ao", "d_Tm", "d_Tp", "d_To", ]
#PARAM_NAME_LIST = ["k_p1Am", "k_p2Am", "k_dAm", "k_diA", "k_cA", "k_sA", "k_dAo", "k_yA", "k_pTm", "k_dTm", "k_ph1", "k_ph2", "k_deph", "k_diT", "k_cT", "k_sT", "k_dTp", "k_sTp", "k_dTo", "k_yT", "k_yTp", "k_AN", "k_TN", "k_a1A", "k_a2A", "k_a1T", "k_a2T", "K_mTA", "K_mAT", "K_mAN", "K_mTN", "K_mT2", "K_mA2", "n_TA", "n_cA", "n_AT", "n_cT", "n_cTp", "n_cTo", "n_AN", "n_TN", "n_a1A", "n_a2A", "n_a1T", "n_a2T", "n_a1Tp", "K_cA", "d_Ao", "d_Tm", "d_Tp", "d_To", ]

assert opt.option in ["option1", "option2"]
if opt.option == "option1":
    PARAMS = [
        {
            "id": 0,
            "name": "k_p1Am",
            "init": 1.89870894e-02,  # 0.002,
            "lb": 0.01 * LOWFOLD,  # 0.5,
            "ub": 0.02 * UPFOLD  # 2.0
        },
        {
            "id": 1,
            "name": "k_p2Am",
            "init": 1.28389723e-03,  # 0.001,#0.0,
            "lb": 0.0,
            "ub": 0.1
        },
        {
            "id": 2,
            "name": "k_dAm",
            "init": 5.51210297e-03,  # 0.15,#0.08,
            "lb": 0.005 * LOWFOLD,
            "ub": 0.5 * UPFOLD
        },
        {
            "id": 3,
            "name": "k_diA",
            "init": 2.15326083e-02,  # 0.2,#0.2,#1.0,
            "lb": 0.02 * LOWFOLD,
            "ub": 5.0 * UPFOLD
        },
        {
            "id": 4,
            "name": "k_cA",
            "init": 3.56228549e+00,  # 0.1,#10,#15.0,
            "lb": 0.1 * LOWFOLD,
            "ub": 3.5 * UPFOLD
        },
        {
            "id": 5,
            "name": "k_sA",
            "init": 7e-5,  # 6.86585495e-04,#0.003,#0.025,
            "lb": 7e-5 * LOWFOLD,
            "ub": 7e-4 * UPFOLD
        },
        {
            "id": 6,
            "name": "k_dAo",
            "init": 2.13832924e-02,  # 0.1,
            "lb": 0.0,
            "ub": 0.2 * UPFOLD
        },
        {
            "id": 7,
            "name": "k_yA",
            "init": 8.02786625e-04,  # 0.02,#0.04,
            "lb": 0.0008 * LOWFOLD,
            "ub": 0.2 * UPFOLD
        },
        {
            "id": 8,
            "name": "k_pTm",
            "init": 10,#15,  # 1.02346459e-03,#0.001,
            "lb": 1 * LOWFOLD,
            "ub": 10 * UPFOLD
        },
        {
            "id": 9,
            "name": "k_dTm",
            "init": 0.5,  # 4.51742661e-01,#0.5,
            "lb": 0.1 * LOWFOLD,
            "ub": 2.5 * UPFOLD
        },
        {
            "id": 10,
            "name": "k_ph1",
            "init": 2.5*2,#2.5*5,  # 1.32419387e+00,#0.5,
            "lb": 0.5 * LOWFOLD,
            "ub": 5.0 * UPFOLD
        },
        {
            "id": 11,
            "name": "k_ph2",
            "init": 3*2,#3*5,  # 2.05606939e+00,#2.0,
            "lb": 6 * LOWFOLD,
            "ub": 6  * UPFOLD
        },
        {
            "id": 12,
            "name": "k_deph",
            "init": 9.97622454e-01,  # 1.0,#6.0,
            "lb": 1.0 * LOWFOLD,
            "ub": 30.0 * UPFOLD
        },
        {
            "id": 13,
            "name": "k_diT",
            "init": 0.68,  # 6.80829297e-01,#0.5,
            "lb": 0.1 * LOWFOLD,
            "ub": 2.5 * UPFOLD
        },
        {
            "id": 14,
            "name": "k_cT",
            "init": 0.1,  # 0.04,#3.93543647e-02,#0.045,#0.045,
            "lb": 0.01 * LOWFOLD,
            "ub": 0.225 * UPFOLD
        },
        {
            "id": 15,
            "name": "k_sT",
            "init": 5,  # 1,#1.46860174e-01,#1,#4.0,
            "lb": 0.5 * LOWFOLD,
            "ub": 20.0 * UPFOLD
        },
        {
            "id": 16,
            "name": "k_dTp",
            "init": 1.60981382e-01,  # 0.1,
            "lb": 0.1 * LOWFOLD,
            "ub": 5 * UPFOLD
        },
        {
            "id": 17,
            "name": "k_sTp",
            "init": 5,  # 3.00026704e+00,#3.0,
            "lb": 0.5 * LOWFOLD,
            "ub": 15.0 * UPFOLD
        },
        {
            "id": 18,
            "name": "k_dTo",
            "init": 0.003,  # 3.00776991e-03,# 0.005,#0.01,
            "lb": 0.003 * LOWFOLD,
            "ub": 0.03 * UPFOLD
        },
        {
            "id": 19,
            "name": "k_yT",
            "init": 2.44026659e+00,  # 2.0,#10.0,
            "lb": 2.0 * LOWFOLD,
            "ub": 50.0 * UPFOLD
        },
        {
            "id": 20,
            "name": "k_yTp",
            "init": 9.98884700e+00,  # 10.0,
            "lb": 2.0 * LOWFOLD,
            "ub": 50.0 * UPFOLD
        },
        {
            "id": 21,
            "name": "k_AN",
            "init": 7.52832341e+00,  # 2.0,
            "lb": 5 * LOWFOLD,
            "ub": 10.0 * UPFOLD
        },
        {
            "id": 22,
            "name": "k_TN",
            "init": 1.11001511e+00,  # 4.0,
            "lb": 1 * LOWFOLD,
            "ub": 20.0 * UPFOLD
        },
        {
            "id": 23,
            "name": "k_a1A",
            "init": 0.001,  # 4.12863979e-04,#0.0007,
            "lb": 0.001 * LOWFOLD,
            "ub": 0.001 * UPFOLD
        },
        {
            "id": 24,
            "name": "k_a2A",
            "init": 100,  # 7.11387924e+00,#24.0,
            "lb": 20 * LOWFOLD,
            "ub": 500.0 * UPFOLD
        },
        {
            "id": 25,
            "name": "k_a1T",
            "init": 1,  # 4.91485347e-04,#0.0006,
            "lb": 0.1 * LOWFOLD,
            "ub": 10 * UPFOLD
        },
        {
            "id": 26,
            "name": "k_a2T",
            "init": 500,  # 2.00000000e+01,#20.0,#60.0,
            "lb": 50.0 * LOWFOLD,
            "ub": 300.0 * UPFOLD
        },
        {
            "id": 27,
            "name": "K_mTA",
            "init": 5.17340221e-12,  # 1e-10,
            "lb": 2e-12 * LOWFOLD,
            "ub": 5 * UPFOLD
        },
        {
            "id": 28,
            "name": "K_mAT",
            "init": 1.78187831e-04,  # 0.0005,
            "lb": 0.0001 * LOWFOLD,
            "ub": 0.0025 * UPFOLD
        },
        {
            "id": 29,
            "name": "K_mAN",
            "init": 3.22245152e-08,  # 1e-08,
            "lb": 2e-09 * LOWFOLD,
            "ub": 5e-08 * UPFOLD
        },
        {
            "id": 30,
            "name": "K_mTN",
            "init": 2.92220667e-14,  # 1e-12,
            "lb": 2e-14 * LOWFOLD,
            "ub": 5e-12 * UPFOLD
        },
        {
            "id": 31,
            "name": "K_mT2",
            "init": 0.1,  # 4.69825956e-14,#1.6e-12,
            "lb": 0.1 * LOWFOLD,
            "ub": 0.1 * UPFOLD
        },
        {
            "id": 32,
            "name": "K_mA2",
            "init": 1.00645539e-08,  # 1e-08,
            "lb": 2e-08 * LOWFOLD,
            "ub": 5e-06 * UPFOLD
        },
        {
            "id": 33,
            "name": "n_TA",
            "init": 2.03963910e+00,  # 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 34,
            "name": "n_cA",
            "init": 3.90702369e+00,  # 4.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 35,
            "name": "n_AT",
            "init": 1.01109972e+00,  # 1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 36,
            "name": "n_cT",
            "init": 1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 37,
            "name": "n_cTp",
            "init": 1.01336043e+00,  # 4.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 38,
            "name": "n_cTo",
            "init": 1.00000350e+00,  # 1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 39,
            "name": "n_AN",
            "init": 2.62861359e+00,  # 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 40,
            "name": "n_TN",
            "init": 1.87556484e+00,  # 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 41,  # 2;[1,12]
            "name": "n_a1A",
            "init": 1.32987400e+00,  # 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 42,  # 8;[1,12]
            "name": "n_a2A",
            "init": 7.63672859e+00,  # 8.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 43,
            "name": "n_a1T",
            "init": 4,  # 1.88070473e+00,#1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 44,
            "name": "n_a2T",
            "init": 4,  # 1+1.00141166e+00,#2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 45,
            "name": "n_a1Tp",
            "init": 2.23999459e+00,  # 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 46,
            "name": "n_a2Tp",
            "init": 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 47,
            "name": "k_acsf",
            "init": 2e-2,
            "lb": 2e-2 * LOWFOLD,
            "ub": 2e-2 * UPFOLD
        },
        {
            "id": 48,
            "name": "k_tcsf",
            "init": 2e-2,
            "lb": 2e-2 * LOWFOLD,
            "ub": 2e-2 * UPFOLD
        },
        # {
        #     "id": 46,
        #     "name": "d_Am",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 12.0,
        # },
        # {
        #     "id": 47,
        #     "name": "d_Ao",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # },
        # {
        #     "id": 48,
        #     "name": "d_Tm",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # },
        # {
        #     "id": 49,
        #     "name": "d_Tp",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # },
        # {
        #     "id": 50,
        #     "name": "d_To",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # }
    ]
else:  # for option2
    PARAMS = [
        {
            "id": 0,
            "name": "k_p1Am",
            "init": 0.2,  # (new update) 0.1,
            "lb": 0.02 * LOWFOLD,#0.5,
            "ub": 0.1 * UPFOLD#2.0
        },
        {
            "id": 1,
            "name": "k_p2Am",
            "init": 0.2,  # (new update) 0.1,#0.0,
            "lb": 0.0,
            "ub": 1
        },
        {
            "id": 2,
            "name": "k_dAm",
            "init": 0.1,#0.08,
            "lb": 0.016 * LOWFOLD,
            "ub": 0.4 * UPFOLD
        },
        {
            "id": 3,
            "name": "k_diA",
            "init": 0.2,#0.2,#1.0,
            "lb": 0.2 * LOWFOLD,
            "ub": 5.0 * UPFOLD
        },
        {
            "id": 4,
            "name": "k_cA",
            "init": 0.5,#10,#15.0,
            "lb": 0.1 * LOWFOLD,
            "ub": 3.0 * UPFOLD
        },
        {
            "id": 5,
            "name": "k_sA",
            "init": 0.003,#0.025,
            "lb": 0.001 * LOWFOLD,
            "ub": 0.125 * UPFOLD
        },
        {
            "id": 6,
            "name": "k_dAo",
            "init": 0.1,
            "lb": 0.0,
            "ub": 0.2 * UPFOLD
        },
        {
            "id": 7,
            "name": "k_yA",
            "init": 0.02,#0.04,
            "lb": 0.008 * LOWFOLD,
            "ub": 0.2 * UPFOLD
        },
        {
            "id": 8,
            "name": "k_pTm",
            "init": 0.2,  # (new update) 0.3,
            "lb": 0.0002 * LOWFOLD,
            "ub": 0.5 * UPFOLD
        },
        {
            "id": 9,
            "name": "k_dTm",
            "init": 0.05,  # (new update) 0.1,
            "lb": 0.1 * LOWFOLD,
            "ub": 2.5 * UPFOLD
        },
        {
            "id": 10,
            "name": "k_ph1",
            "init": 2.0,  # (new update) 4,
            "lb": 0.1 * LOWFOLD,
            "ub": 4 * UPFOLD
        },
        {
            "id": 11,
            "name": "k_ph2",
            "init": 5.0,  # (new update) 4.0,
            "lb": 0.4 * LOWFOLD,
            "ub": 10.0 * UPFOLD
        },
        {
            "id": 12,
            "name": "k_deph",
            "init": 1.0,#6.0,
            "lb": 1.0 * LOWFOLD,
            "ub": 10.0 * UPFOLD
        },
        {
            "id": 13,
            "name": "k_diT",
            "init": 0.5,
            "lb": 0.1 * LOWFOLD,
            "ub": 2.5 * UPFOLD
        },
        {
            "id": 14,
            "name": "k_cT",
            "init": 0.4,#0.045,
            "lb": 0.009 * LOWFOLD,
            "ub": 0.4 * UPFOLD
        },
        {
            "id": 15,
            "name": "k_sT",
            "init": 4.0,  # (new update) 1,#4.0,
            "lb": 1.0 * LOWFOLD,
            "ub": 20.0 * UPFOLD
        },
        {
            "id": 16,
            "name": "k_dTp",
            "init": 0.05,
            "lb": 0.02 * LOWFOLD,
            "ub": 0.5 * UPFOLD
        },
        {
            "id": 17,
            "name": "k_sTp",
            "init": 3.0,
            "lb": 0.6 * LOWFOLD,
            "ub": 15.0 * UPFOLD
        },
        {
            "id": 18,
            "name": "k_dTo",
            "init": 0.005,#0.01,
            "lb": 0.002 * LOWFOLD,
            "ub": 0.05 * UPFOLD
        },
        {
            "id": 19,
            "name": "k_yT",
            "init": 2.0,#10.0,
            "lb": 2.0 * LOWFOLD,
            "ub": 50.0
        },
        {
            "id": 20,
            "name": "k_yTp",
            "init": 10.0,
            "lb": 2.0 * LOWFOLD,
            "ub": 50.0 * UPFOLD
        },
        {
            "id": 21,
            "name": "k_AN",
            "init": 2.0,
            "lb": 0.4 * LOWFOLD,
            "ub": 10.0 * UPFOLD
        },
        {
            "id": 22,
            "name": "k_TN",
            "init": 4.0,
            "lb": 0.8 * LOWFOLD,
            "ub": 20.0 * UPFOLD
        },
        {
            "id": 23,
            "name": "k_a1A",
            "init": 0.0007,
            "lb": 0.00014 * LOWFOLD,
            "ub": 0.007
        },
        {
            "id": 24,
            "name": "k_a2A",
            "init": 24.0,
            "lb": 4.8 * LOWFOLD,
            "ub": 120.0 * UPFOLD
        },
        {
            "id": 25,
            "name": "k_a1T",
            "init": 0.0006,
            "lb": 0.00012 * LOWFOLD,
            "ub": 0.003 * UPFOLD
        },
        {
            "id": 26,
            "name": "k_a2T",
            "init": 20.0,#60.0,
            "lb": 12.0 * LOWFOLD,
            "ub": 300.0 * UPFOLD
        },
        {
            "id": 27,
            "name": "K_mTA",
            "init": 1e-8,
            "lb": 2e-11 * LOWFOLD,
            "ub": 5e-8 * UPFOLD
        },
        {
            "id": 28,
            "name": "K_mAT",
            "init": 0.001,  # (new update) 0.0005,
            "lb": 0.0001 * LOWFOLD,
            "ub": 0.0025 * UPFOLD
        },
        {
            "id": 29,
            "name": "K_mAN",
            "init": 1e-08,
            "lb": 1e-09 * LOWFOLD,
            "ub": 1e-07 * UPFOLD
        },
        {
            "id": 30,
            "name": "K_mTN",
            "init": 1e-12,
            "lb": 2e-13 * LOWFOLD,
            "ub": 5e-11 * UPFOLD
        },
        {
            "id": 31,
            "name": "K_mT2",
            "init": 1.6e-12,
            "lb": 3e-13 * LOWFOLD,
            "ub": 8e-11 * UPFOLD
        },
        {
            "id": 32,
            "name": "K_mA2",
            "init": 1e-08,
            "lb": 2e-09 * LOWFOLD,
            "ub": 5e-07 * UPFOLD
        },
        {
            "id": 33,
            "name": "n_TA",
            "init": 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 34,
            "name": "n_cA",
            "init": 4.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 35,
            "name": "n_AT",
            "init": 1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 36,
            "name": "n_cT",
            "init": 1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 37,
            "name": "n_cTp",
            "init": 4.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 38,
            "name": "n_cTo",
            "init": 1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 39,
            "name": "n_AN",
            "init": 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 40,
            "name": "n_TN",
            "init": 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 41,  #2;[1,12]
            "name": "n_a1A",
            "init": 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 42,  #8;[1,12]
            "name": "n_a2A",
            "init": 8.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 43,
            "name": "n_a1T",
            "init": 1.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 44,
            "name": "n_a2T",
            "init": 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 45,
            "name": "n_a1Tp",
            "init": 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        {
            "id": 46,
            "name": "n_a2Tp",
            "init": 4.0,  # (new update) 2.0,
            "lb": 1.0,
            "ub": 12.0
        },
        # {
        #     "id": 46,
        #     "name": "d_Am",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 12.0,
        # },
        # {
        #     "id": 47,
        #     "name": "d_Ao",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # },
        # {
        #     "id": 48,
        #     "name": "d_Tm",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # },
        # {
        #     "id": 49,
        #     "name": "d_Tp",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # },
        # {
        #     "id": 50,
        #     "name": "d_To",
        #     "init": 1.0,
        #     "lb": 1.0,
        #     "ub": 1.0,
        # }
    ]

"""
PARAMS = [
    {
        "id": 0,
        "name": "k_p1Am",
        "init": 0.001,
        "lb": 0.0002,
        "ub": 0.005
    },
    {
        "id": 1,
        "name": "k_p2Am",
        "init": 0.0,
        "lb": 0.0,
        "ub": 0.001
    },
    {
        "id": 2,
        "name": "k_dAm",
        "init": 0.08,
        "lb": 0.016,
        "ub": 0.4
    },
    {
        "id": 3,
        "name": "k_diA",
        "init": 1.0,
        "lb": 0.2,
        "ub": 5.0
    },
    {
        "id": 4,
        "name": "k_cA",
        "init": 15.0,
        "lb": 3.0,
        "ub": 30.0
    },
    {
        "id": 5,
        "name": "k_sA",
        "init": 0.025,
        "lb": 0.005,
        "ub": 0.125
    },
    {
        "id": 6,
        "name": "k_dAo",
        "init": 0.1,
        "lb": 0.0,
        "ub": 0.2
    },
    {
        "id": 7,
        "name": "k_yA",
        "init": 0.04,
        "lb": 0.008,
        "ub": 0.2
    },
    {
        "id": 8,
        "name": "k_pTm",
        "init": 0.001,
        "lb": 0.0002,
        "ub": 0.005
    },
    {
        "id": 9,
        "name": "k_dTm",
        "init": 0.5,
        "lb": 0.1,
        "ub": 2.5
    },
    {
        "id": 10,
        "name": "k_ph1",
        "init": 0.5,
        "lb": 0.1,
        "ub": 2.5
    },
    {
        "id": 11,
        "name": "k_ph2",
        "init": 2.0,
        "lb": 0.4,
        "ub": 10.0
    },
    {
        "id": 12,
        "name": "k_deph",
        "init": 6.0,
        "lb": 1.2,
        "ub": 30.0
    },
    {
        "id": 13,
        "name": "k_diT",
        "init": 0.5,
        "lb": 0.1,
        "ub": 2.5
    },
    {
        "id": 14,
        "name": "k_cT",
        "init": 0.045,
        "lb": 0.009,
        "ub": 0.225
    },
    {
        "id": 15,
        "name": "k_sT",
        "init": 4.0,
        "lb": 0.8,
        "ub": 20.0
    },
    {
        "id": 16,
        "name": "k_dTp",
        "init": 0.1,
        "lb": 0.02,
        "ub": 0.5
    },
    {
        "id": 17,
        "name": "k_sTp",
        "init": 3.0,
        "lb": 0.6,
        "ub": 15.0
    },
    {
        "id": 18,
        "name": "k_dTo",
        "init": 0.01,
        "lb": 0.002,
        "ub": 0.05
    },
    {
        "id": 19,
        "name": "k_yT",
        "init": 10.0,
        "lb": 2.0,
        "ub": 50.0
    },
    {
        "id": 20,
        "name": "k_yTp",
        "init": 10.0,
        "lb": 2.0,
        "ub": 50.0
    },
    {
        "id": 21,
        "name": "k_AN",
        "init": 2.0,
        "lb": 0.4,
        "ub": 10.0
    },
    {
        "id": 22,
        "name": "k_TN",
        "init": 4.0,
        "lb": 0.8,
        "ub": 20.0
    },
    {
        "id": 23,
        "name": "k_a1A",
        "init": 0.0007,
        "lb": 0.00014,
        "ub": 0.0035
    },
    {
        "id": 24,
        "name": "k_a2A",
        "init": 24.0,
        "lb": 4.8,
        "ub": 120.0
    },
    {
        "id": 25,
        "name": "k_a1T",
        "init": 0.0006,
        "lb": 0.00012,
        "ub": 0.003
    },
    {
        "id": 26,
        "name": "k_a2T",
        "init": 60.0,
        "lb": 12.0,
        "ub": 300.0
    },
    {
        "id": 27,
        "name": "K_mTA",
        "init": 1e-10,
        "lb": 2e-11,
        "ub": 5e-10
    },
    {
        "id": 28,
        "name": "K_mAT",
        "init": 0.0005,
        "lb": 0.0001,
        "ub": 0.0025
    },
    {
        "id": 29,
        "name": "K_mAN",
        "init": 1e-08,
        "lb": 2e-09,
        "ub": 5e-08
    },
    {
        "id": 30,
        "name": "K_mTN",
        "init": 1e-12,
        "lb": 2e-13,
        "ub": 5e-12
    },
    {
        "id": 31,
        "name": "K_mT2",
        "init": 1.6e-12,
        "lb": 3e-13,
        "ub": 8e-12
    },
    {
        "id": 32,
        "name": "K_mA2",
        "init": 1e-08,
        "lb": 2e-09,
        "ub": 5e-08
    },
    {
        "id": 33,
        "name": "n_TA",
        "init": 2.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 34,
        "name": "n_cA",
        "init": 4.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 35,
        "name": "n_AT",
        "init": 1.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 36,
        "name": "n_cT",
        "init": 1.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 37,
        "name": "n_cTp",
        "init": 4.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 38,
        "name": "n_cTo",
        "init": 1.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 39,
        "name": "n_AN",
        "init": 2.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 40,
        "name": "n_TN",
        "init": 2.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 41,
        "name": "n_a1A",
        "init": 2.0,
        "lb": 0.0,
        "ub": 12.0
    },
    {
        "id": 42,
        "name": "n_a2A",
        "init": 8.0,
        "lb": 0.0,
        "ub": 12.0
    },
    {
        "id": 43,
        "name": "n_a1T",
        "init": 1.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 44,
        "name": "n_a2T",
        "init": 2.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 45,
        "name": "n_a1Tp",
        "init": 2.0,
        "lb": 1.0,
        "ub": 12.0
    },
    {
        "id": 46,
        "name": "d_Am",
        "init": 1.0,
        "lb": 1.0,
        "ub": 1.0,
    },
    {
        "id": 47,
        "name": "d_Ao",
        "init": 1.0,
        "lb": 1.0,
        "ub": 1.0,
    },
    {
        "id": 48,
        "name": "d_Tm",
        "init": 1.0,
        "lb": 1.0,
        "ub": 1.0,
    },
    {
        "id": 49,
        "name": "d_Tp",
        "init": 1.0,
        "lb": 1.0,
        "ub": 1.0,
    },
    {
        "id": 50,
        "name": "d_To",
        "init": 1.0,
        "lb": 1.0,
        "ub": 1.0,
    }
]
"""



# string = """
# #M/year
# #####uppder and lower bound. 0.2~5fold if not indicated
# k_p1Am = 0.001
# k_p2Am = 0 #[0 0.001]
# k_dAm = 0.08
# k_diA = 1
# k_cA = 15 #[3 30]
# k_sA = 0.025
# k_dAo = 0.1 #[0 0.2]
# k_yA = 0.04
#
#
# k_pTm = 0.001
# k_dTm = 0.5
# k_ph1 = 0.5
# k_ph2 = 2
# k_deph = 6
#
# k_diT = 0.5
# k_cT = 0.045
# k_sT = 4
# k_dTp = 0.1
# k_sTp = 3
# k_dTo = 0.01
# k_yT = 10
# k_yTp = 10
# k_AN = 2
# k_TN = 4
# # new added
# k_a1A = 7e-4
# k_a2A = 24
#
# k_a1T = 6e-4
# k_a2T = 60
#
# K_mTA = 1.0e-10
# K_mAT = 5.0e-4
# K_mAN = 1.0e-8
# K_mTN = 1.0e-12
#
# #1028_RX
# K_mT2 = 1.6e-12
# K_mA2 = 1e-8
#
#
# ##The following parameters are integers between [1 12]
# n_TA = 2.0
# n_cA = 4.0
# n_AT = 1.0
# n_cT = 1.0
# n_cTp = 4.0
# #1028_RX
# n_cTo = 1.0
# n_AN = 2.0
# n_TN = 2.0
# # new added
# n_a1A = 2.0
# n_a2A = 8
# n_a1T = 1.0
# n_a2T = 2.0
# n_a1Tp = 2.0
#
# ##following parameters are fixed
# d_Am = 1.0
# d_Ao = 1.0
# d_Tm = 1.0
# d_Tp = 1.0
# d_To = 1.0
# """
# import json
# lines = string.split("\n")
# lines = [item for item in lines if len(item) > 2 and item[0] != "#"]
# print("num_params:", len(lines))
#
# name_list = []
# # init_list = []
# # min_list = []
# # max_list = []
#
# dic = []
# for i, line in enumerate(lines):
#     items = line.split()
#     dic.append({
#         "id": i,
#         "name": items[0],
#         "init": round(float(items[2]), 13),
#         "lb": round(0.2 * float(items[2]), 13),
#         "ub": round(5.0 * float(items[2]), 13)
#     })
#     name_list.append(items[0])
#     # init_list.append(float(items[2]))
#     # min_list.append(float(items[2]) * 0.2)
#     # max_list.append(float(items[2]) * 5.0)
# print("PARAM_NAME_LIST = [", end="")
# for item in name_list:
#     print("\"{}\", ".format(item), end="")
# print("]")
#
# print(json.dumps(dic, indent=4, ensure_ascii=False))


# for i in range(PARAM_NUM):
#     print("{}, ".format(PARAM_NAME_LIST[i]), end="")
#
# k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp, d_Am, d_Ao, d_Tm, d_Tp, d_To = iter(params)
