from simulation import simulate
# from pymoo.operators.sampling.lhs import LHS
if __name__ == "__main__":
    simulate(pop_size=30, generation=100, method="DE")
