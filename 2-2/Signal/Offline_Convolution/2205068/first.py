import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, INF):
        self.INF = INF
        self.values = np.zeros(2 * self.INF + 1)

    def set_value_at_time(self, t, value):
        if -self.INF <= t <= self.INF:
            self.values[t + self.INF] = value

    def shift(self, k):
        new_signal = Signal(self.INF)
        # y(n) = x(n-k). Value at index i moves to i+k
        for i in range(len(self.values)):
            new_idx = i + k
            if 0 <= new_idx < len(self.values):
                new_signal.values[new_idx] = self.values[i]
        return new_signal

    def add(self, other):
        new_signal = Signal(self.INF)
        new_signal.values = self.values + other.values
        return new_signal

    def multiply(self, scalar):
        new_signal = Signal(self.INF)
        new_signal.values = self.values * scalar
        return new_signal

    def plot(self, title="Discrete Signal"):
        t = np.arange(-self.INF, self.INF + 1)
        plt.figure(figsize=(10, 4))
        plt.stem(t, self.values)
        plt.title(title)
        plt.xlabel("n")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

class LTI_System:
    def __init__(self, impulse_response: Signal):
        self.h = impulse_response

    def linear_combination_of_impulses(self, input_signal: Signal):
        impulses = []
        coefficients = []
        for i in range(len(input_signal.values)):
            val = input_signal.values[i]
            if val != 0:
                time_k = i - input_signal.INF
                impulse = Signal(input_signal.INF)
                impulse.set_value_at_time(time_k, 1.0)
                
                impulses.append(impulse)
                coefficients.append(val)
        return impulses, coefficients

    def output(self, input_signal: Signal):
        _, coefficients = self.linear_combination_of_impulses(input_signal)
        
        result_signal = Signal(input_signal.INF)
        
        for i in range(len(input_signal.values)):
            x_k = input_signal.values[i]
            if x_k != 0:
                k = i - input_signal.INF
                shifted_h = self.h.shift(k)
                scaled_h = shifted_h.multiply(x_k)
                result_signal = result_signal.add(scaled_h)
        
        return result_signal

if __name__ == "__main__":
    INF = 10
    x = Signal(INF)
    x.set_value_at_time(-2, 1)
    x.set_value_at_time(0, 2)
    x.set_value_at_time(3, -1)
    
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    system = LTI_System(h)
    y = system.output(x)
    y.plot("Output Signal y(n)")