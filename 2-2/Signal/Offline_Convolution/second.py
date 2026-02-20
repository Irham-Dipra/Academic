from first import Signal, LTI_System
import numpy as np

def run_smoothing():
    with open('input_signal.txt', 'r') as f:
        lines = f.readlines()
        n_start, n_end = map(int, lines[0].split())
        values = list(map(float, lines[1].split()))

    INF = max(abs(n_start), abs(n_end)) + 10
    
    x = Signal(INF)
    for i, val in enumerate(values):
        x.set_value_at_time(n_start + i, val)
    
    x.plot("Noisy Input Signal")

    h = Signal(INF)
    for n in [-2, -1, 0, 1, 2]:
        h.set_value_at_time(n, 1/5)

    system = LTI_System(h)
    y = system.output(x)

    y.plot("Smoothed Output Signal")

if __name__ == "__main__":
    run_smoothing()