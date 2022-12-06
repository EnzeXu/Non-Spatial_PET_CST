import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX

from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF

from const import *
from ode_truth import loss_func, get_now_string, ConstTruth, run


def toy_loss_func(x):
    true = [(PARAMS[i]["ub"] + PARAMS[i]["lb"]) / 2.0 for i in range(PARAM_NUM)]
    return np.sum(np.abs(np.asarray(true) - np.asarray(x)))


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=PARAM_NUM,
                         n_obj=1,
                         # n_eq_constr=1,
                         #n_ieq_constr=0,
                         xl=np.array([PARAMS[i]["lb"] for i in range(PARAM_NUM)]),
                         xu=np.array([PARAMS[i]["ub"] for i in range(PARAM_NUM)]),
                         )
        self.ct = ConstTruth(
            csf_folder_path="data/CSF/",
            pet_folder_path="data/PET/"
        )

    def _evaluate(self, x, out, *args, **kwargs):
        loss_all = loss_func(x, self.ct)
        # loss1, loss2, loss3, loss4, loss5, loss6, loss7 = iter(loss_all)
        loss1 = np.sum(loss_all)
        # loss1 = np.sum(np.abs(x))
        # eq1 = np.sum(x[33: 46] - np.floor(x[33: 46]))


        out["F"] = [loss1]  # [loss1, loss2]
        # out["H"] = [eq1]


def simulate(pop_size=50, generation=100, method="GA"):
    time_string_start = get_now_string()
    t0 = time.time()
    print("[run - multi_obj] Start at {}".format(time_string_start))
    problem = MyProblem()

    initial_x = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])  # default
#    initial_x = np.load("saves/params_20221203_113822.npy")
    assert method in ["GA", "DE", "ES", "PSO", "BRKGA", "G3PCX"]
    print("[run - multi_obj] Method: {}".format(method))
    if method == "DE":
        algorithm = DE(
            pop_size=pop_size,
            n_offsprings=2,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False
        )
    elif method == "ES":
        algorithm = ES(
            pop_size=pop_size,
            n_offsprings=100,
            rule=1.0 / 7.0
        )
    elif method == "PSO":
        algorithm = PSO(
            pop_size=pop_size,
        )
    elif method == "BRKGA":
        algorithm = BRKGA(
            n_elites=10,
            sampling=initial_x,
            n_offsprings=2,
            n_mutants=10,
            bias=0.7
        )
    elif method == "G3PCX":
        algorithm = G3PCX(
            pop_size=pop_size,
            n_offsprings=2
        )
    else:
        algorithm = NSGA2(
            pop_size=pop_size,  # [highlight] population size
            n_offsprings=2,
            sampling=initial_x,  # sampling=initial_x
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

    termination = get_termination("n_gen", generation)  # [highlight] step number
    t00 = time.time()
    print("[run - multi_obj] Starting ...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=0,
                   save_history=True,
                   verbose=True)
    print("[run - multi_obj] Finished GA. Time cost: {0:.2f} s".format(time.time() - t00))
    X = res.X
    F = res.F
    print("[run - multi_obj] X:", X)
    print("[run - multi_obj] F:", F)

    if np.ndim(F) == 1:
        best_x = X
        best_f = F
    else:
        approx_min = F.min(axis=0)
        approx_max = F.max(axis=0)
        print("[run - multi_obj] (before normalization) min:", approx_min)
        print("[run - multi_obj] (before normalization) max:", approx_max)
        # print(approx_max - approx_min)
        nF = (F - approx_min) / (approx_max - approx_min) if len(F) > 1 else F

        print("[run - multi_obj] (after normalization) min:", nF.min(axis=0))
        print("[run - multi_obj] (after normalization) max:", nF.max(axis=0))
        average_n = 1
        weights = np.array([1.0 / average_n] * average_n)
        decomp = ASF()
        i = decomp.do(nF, 1.0 / weights).argmin()

        # print("Best regarding ASF: Point i = %s\nF = %s\nX = %s" % (i, F[i], X[i]))
        best_x = X[i]
        best_f = F[i:i+1]


    folder_path = "saves/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path_params_x = os.path.join(folder_path, "params_{}.npy".format(time_string_start))
    save_path_params_f = os.path.join(folder_path, "val_{}.npy".format(time_string_start))
    print("[run - multi_obj] Params shape: ", best_x.shape)
    print(best_x)
    print("[run - multi_obj] The optimal params achieved loss = {}".format(best_f[0]))
    np.save(save_path_params_x, best_x)
    np.save(save_path_params_f, best_f)
    print("[run - multi_obj] The optimal params are saved to \"{}\" and the optimal value is saved to \"{}\".".format(save_path_params_x, save_path_params_f))
    t1 = time.time()
    time_string_end = get_now_string()
    print("[run - multi_obj] End at {0} ({1:.2f} min)".format(time_string_end, (t1 - t0) / 60.0))

    original_params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
    original_loss = np.sum(loss_func(original_params, problem.ct))
    print("[run - multi_obj] Note that using the initial params loss = {}".format(original_loss))

    # plt.figure(figsize=(7, 5))
    # plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    # plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
    # plt.title("Objective Space")
    # plt.show()

    """
    ct = ConstTruth(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/"
    )

    p = np.load("saves/params_20221102_214224.npy") # load saved file (33*1)
    record2 = loss_func(p, ct) # calcu loss
    print(record2)

    run(p) # plot
    """


def test_params(params_path="saves/params_default_46.npy"):
    params = np.load(params_path)
    params_dic = {PARAM_NAME_LIST[i]: "{} [{}, {}, {}]".format(params[i], PARAMS[i]["lb"], PARAMS[i]["init"], PARAMS[i]["ub"]) for i in range(PARAM_NUM)}
    print("Params = ")
    print(json.dumps(params_dic, indent=4))
    run(params)
    return params


if __name__ == "__main__":
    simulate(pop_size=30, generation=1000, method="DE")
    # test_params("saves/params_20221112_205713.npy")
    # test_params("saves/params_20221112_225529.npy")
    # test_params()
