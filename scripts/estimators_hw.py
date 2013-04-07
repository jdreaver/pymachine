import numpy as np

def answer():
    num_trials = 10000
    e = np.array([np.min(np.random.rand(2)) for i in range(num_trials)]).mean()
    print("Expected value of e:", e)
    return

if __name__ == '__main__':
    answer()
