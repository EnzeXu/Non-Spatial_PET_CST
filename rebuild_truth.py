
import random
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt


def build_one_truth(curve_name, x, f, params):
    x = np.asarray(x)
    y = f(x, params)
    print(curve_name)
    print(x)
    print(y)
    return {
        "curve_name": curve_name,
        "x": x,
        "y": y
    }


def one_time_build_new_truth():
    x = np.linspace(0.0, 12.0, 121)
    # x = np.asarray([3.0,5.0,9.0,11.0,12.0])
    dic = dict()
    f_APET = lambda x, params: params[0] * (params[1] ** (params[2] * (x-3) + params[3])) + params[4] * (x-3) + params[5] + params[6] * (x-3) ** 2 + params[7] * (x-3) ** 3 + 1.7
    p_APET = [5.727711, 1.0901781, 0.48869765, -2.4110289, -0.19570374, -4.6469927, -0.0035516794, 0.00017240157]
    dic["APET"] = build_one_truth("APET", x, f_APET, p_APET)
    f_TPET = lambda x, params: params[0] * (params[1] ** (params[2] * (x - 3) + params[3])) + params[4] * (x - 3) + params[5] + params[6] * (x - 3) ** 2 + params[7] * (x - 3) ** 3 + 1.28
    p_TPET = [0.41971564, 10.686993, 0.95063734, -8.744956, -3.3821703e-05, 0.00020107726, 0.00163968, -4.984618e-05]
    dic["TPET"] = build_one_truth("TPET", x, f_TPET, p_TPET)
    f_ACSF = lambda x, params: -params[0] * (params[1] ** (params[2] * x + params[3])) + params[4] * x + params[5] + 144
    p_ACSF = [0.00028278123, 2.6438816, 1.0435379, -0.04161641, -0.62772256, 60.974365]
    dic["ACSF"] = build_one_truth("ACSF", x, f_ACSF, p_ACSF)
    f_TpCSF = lambda x, params: params[0] * (params[1] ** (params[2] * (x-3) + params[3])) + params[4] * (x-3) + params[5] + 27.99
    p_TpCSF = [0.0010828931, 1.666073, 2.1516688, -0.7176372, 0.5728904, 0.016868832]
    dic["TpCSF"] = build_one_truth("TpCSF", x, f_TpCSF, p_TpCSF)
    f_TCSF = lambda x, params: params[0] * (params[1] ** (params[2] * (x - 3) + params[3])) + params[4] * (x - 3) + params[5] + 36.9
    p_TCSF = [0.00087249884, 1.8214605, 2.1398628, -2.300079, 1.2773108, 0.04649601]
    dic["TCSF"] = build_one_truth("TCSF", x, f_TCSF, p_TCSF)
    f_TtCSF = lambda x, params: params[0] * (params[1] ** (params[2] * (x - 3) + params[3])) + params[4] * (x - 3) + params[5] + params[6] * (x - 3) ** 2 + params[7] * (x - 3) ** 3 + 64.9
    p_TtCSF = [0.16684374, 13.285673, 1.2148288, -8.928572, 0.95362085, 0.03791046, 0.387468, -0.021391869]
    dic["TtCSF"] = build_one_truth("TtCSF", x, f_TtCSF, p_TtCSF)
    with open("data/rebuild_truth.pkl", "wb") as f:
        pickle.dump(dic, f)



if __name__ == "__main__":
    one_time_build_new_truth()
