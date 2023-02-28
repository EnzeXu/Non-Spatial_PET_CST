from ode_truth import get_now_string


def pass_omega():
    with open("simulation_record.txt", "a") as f:
        f.write("----------,pass,{}\n".format(get_now_string()))


if __name__ == "__main__":
    pass_omega()
