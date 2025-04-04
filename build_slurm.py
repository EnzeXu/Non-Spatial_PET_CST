draft = """#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --partition=medium
#SBATCH --nodes=2
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --account=chenGrp
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/{0}-%j.o"
#SBATCH --error="jobs_oe/{0}-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python {1} {2}
"""


def one_slurm(job_name, python_name, kwargs, draft=draft):
    path = "jobs/{}.slurm".format(job_name)
    print("building {}".format(path))
    with open(path, "w") as f:
        f.write(draft.format(
            job_name,
            python_name,
            " ".join(["--{0} {1}".format(one_key, kwargs[one_key]) for one_key in kwargs]),
        ))



def one_time_build_A():
    plans = [
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        # ["A3", 1000, "chosen_0", "fixed", 100],
        # ["A3", 2000, "chosen_0", "fixed", 100],
        # ["A4", 1000, "chosen_0", "ranged", 100],
        # ["A4", 2000, "chosen_0", "ranged", 100],
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 3000, "all", "fixed", 100],
        # ["A1", 5000, "all", "fixed", 100],
        # ["A1", 7000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 3000, "all", "ranged", 100],
        # ["A2", 5000, "all", "ranged", 100],
        # ["A2", 7000, "all", "ranged", 100],
        # ["A1", 1250, "all", "fixed", 100],
        # ["A1", 1500, "all", "fixed", 100],
        # ["A1", 1750, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1250, "all", "ranged", 100],
        # ["A2", 1500, "all", "ranged", 100],
        # ["A2", 1750, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        # ["A3-1", 500, "chosen_0", "fixed", 100, "option1"],
        # ["A3-1", 1000, "chosen_0", "fixed", 100, "option1"],
        # ["A3-1", 1500, "chosen_0", "fixed", 100, "option1"],
        # ["A3-1", 2000, "chosen_0", "fixed", 100, "option1"],
        # ["A3-2", 500, "chosen_0", "fixed", 100, "option2"],
        # ["A3-2", 1000, "chosen_0", "fixed", 100, "option2"],
        # ["A3-2", 1500, "chosen_0", "fixed", 100, "option2"],
        # ["A3-2", 2000, "chosen_0", "fixed", 100, "option2"],
        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.1],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.1],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.1],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.1],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.1],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.1],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.2],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.2],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.2],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.2],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.2],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.2],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.3],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.3],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.3],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.3],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.3],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.3],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.4],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.4],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.4],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.4],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.4],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.4],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.5],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.5],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.5],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.5],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.5],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.5],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.6],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.6],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.6],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.6],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.6],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.6],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.7],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.7],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.7],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.7],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.7],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.7],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.8],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.8],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.8],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.8],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.8],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.8],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 0.9],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 0.9],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 0.9],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 0.9],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 0.9],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 0.9],

        ["A4-1", 500, "chosen_0", "ranged", 100, "option1", 1.0],
        ["A4-1", 600, "chosen_0", "ranged", 100, "option1", 1.0],
        ["A4-1", 700, "chosen_0", "ranged", 100, "option1", 1.0],
        ["A4-1", 800, "chosen_0", "ranged", 100, "option1", 1.0],
        ["A4-1", 900, "chosen_0", "ranged", 100, "option1", 1.0],
        ["A4-1", 1000, "chosen_0", "ranged", 100, "option1", 1.0],
        # ["A4-1", 1500, "chosen_0", "ranged", 100, "option1"],
        # ["A4-1", 2000, "chosen_0", "ranged", 100, "option1"],
        # ["A4-2", 500, "chosen_0", "ranged", 100, "option2"],
        # ["A4-2", 1000, "chosen_0", "ranged", 100, "option2"],
        # ["A4-2", 1500, "chosen_0", "ranged", 100, "option2"],
        # ["A4-2", 2000, "chosen_0", "ranged", 100, "option2"],
    ]
    dic = dict()
    for one_plan in plans:
        dic["model_name"] = "{0}_{1:.1f}".format(one_plan[0], one_plan[6])
        dic["generation"] = one_plan[1]
        dic["dataset"] = one_plan[2]
        dic["start"] = one_plan[3]
        dic["pop_size"] = one_plan[4]
        dic["option"] = one_plan[5]
        dic["tcsf_scaler"] = one_plan[6]

        one_slurm(
            "GA_{}_{}".format(dic["model_name"], one_plan[1]),
            "test_nsga.py",
            dic)

def one_time_build_S():
    plans = [
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        # ["A3", 1000, "chosen_0", "fixed", 100],
        # ["A3", 2000, "chosen_0", "fixed", 100],
        # ["A4", 1000, "chosen_0", "ranged", 100],
        # ["A4", 2000, "chosen_0", "ranged", 100],
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 3000, "all", "fixed", 100],
        # ["A1", 5000, "all", "fixed", 100],
        # ["A1", 7000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 3000, "all", "ranged", 100],
        # ["A2", 5000, "all", "ranged", 100],
        # ["A2", 7000, "all", "ranged", 100],
        # ["A1", 1250, "all", "fixed", 100],
        # ["A1", 1500, "all", "fixed", 100],
        # ["A1", 1750, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1250, "all", "ranged", 100],
        # ["A2", 1500, "all", "ranged", 100],
        # ["A2", 1750, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        # ["A3-1", 500, "chosen_0", "fixed", 100, "option1"],
        # ["A3-1", 1000, "chosen_0", "fixed", 100, "option1"],
        # ["A3-1", 1500, "chosen_0", "fixed", 100, "option1"],
        # ["A3-1", 2000, "chosen_0", "fixed", 100, "option1"],
        # ["A3-2", 500, "chosen_0", "fixed", 100, "option2"],
        # ["A3-2", 1000, "chosen_0", "fixed", 100, "option2"],
        # ["A3-2", 1500, "chosen_0", "fixed", 100, "option2"],
        # ["A3-2", 2000, "chosen_0", "fixed", 100, "option2"],
        ["S1-500", 500, "chosen_0", "ranged", 100, "option1", 0.3],
        ["S1-1000", 1000, "chosen_0", "ranged", 100, "option1", 0.3],
    ]
    dic = dict()
    for one_plan in plans:
        dic["model_name"] = "{0}_{1:.1f}".format(one_plan[0], one_plan[6])
        dic["generation"] = one_plan[1]
        dic["dataset"] = one_plan[2]
        dic["start"] = one_plan[3]
        dic["pop_size"] = one_plan[4]
        dic["option"] = one_plan[5]
        dic["tcsf_scaler"] = one_plan[6]

        one_slurm(
            "{}_{}".format(dic["model_name"], one_plan[1]),
            "test_nsga.py",
            dic)

if __name__ == "__main__":
    one_time_build_S()
    pass
