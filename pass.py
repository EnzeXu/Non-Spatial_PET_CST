from datetime import datetime


def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def pass_omega():
    with open("simulation_record.txt", "a") as f:
        f.write("----------,pass,{}\n".format(get_now_string()))


if __name__ == "__main__":
    pass_omega()
