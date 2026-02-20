import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, INF):
        self.INF = INF
        self.values = np.zeros(2*INF + 1)
    
    def set_values_at_time(self, t, value):
        if t >= -self.INF and t <= self.INF:
            self.values[t + self.INF] = value
    
    def right_shift(self, k):
        new_signal = Signal(self.INF)
        for i in range(len(self.values)):
            new_idx = i + k
            if(new_idx <= 2*self.INF):
                new_signal.values[new_idx] = self.values[i]
        return new_signal
    
    def add(self, other_signal):
        new_signal = Signal(self.INF)
        new_signal.values = self.values + other_signal.values
        return new_signal
    
    def multiply(self, k):
        new_signal = Signal(self.INF)
        new_signal.values = self.values * k
        return new_signal

    def plot(self, title = "Discrete Signal"):
        t = np.arange(-self.INF, self.INF + 1)
        plt.figure(figsize=(10, 4))
        plt.stem(t, self.values)
        plt.title(title)
        plt.xticks(np.arange(-self.INF, self.INF + 1))
        plt.xlabel("n")
        plt.ylabel("amplitude")
        plt.grid(True)
        plt.show()

class LTI_system:
    def __init__(self, impulse_response:Signal):
        self.h = impulse_response
    
    def linear_combination_of_impulses(self, input:Signal):
        impulses = []
        coeffiecients = []
        for i in range(len(input.values)):
            value = input.values[i]
            if value!= 0:
                coeffiecients.append(value)
                new_signal = Signal(input.INF)
                new_signal.set_values_at_time(i - input.INF, 1.0)
                impulses.append(new_signal)
        return impulses, coeffiecients
    
    def output(self, input:Signal):
        output_signal = Signal(input.INF)
        for i in range(len(input.values)):
            x_k = input.values[i]
            k = i - input.INF
            new_signal = Signal(input.INF)
            new_signal = self.h.right_shift(k)
            new_signal = new_signal.multiply(x_k)
            output_signal = output_signal.add(new_signal)
        return output_signal
    
    def output2(self, intput:Signal):
        output_signal = Signal(input.INF)
        for i in range(-input.INF, input.INF + 1):
            flipped_impulse = Signal(input.INF)
            flipped_impulse.values = np.flip(self.h.values)
            flipped_impulse = flipped_impulse.right_shift(i)
            x = input.values * flipped_impulse.values
            value = np.sum(x)
            output_signal.values[i] = value
        return output_signal




if __name__ == "__main__":
    INF = 10
    x = Signal(INF)
    x.set_values_at_time(-2, 1)
    x.set_values_at_time(0, 2)
    x.set_values_at_time(3, -1)
    x.plot()

    h = Signal(INF)
    h.set_values_at_time(0, 1)
    h.set_values_at_time(1, 0.5)

    system = LTI_system(h)
    y = system.output(x)
    y.plot("Output Signal y(n)")

    y2 = system.output(x)
    y2.plot("2nd output")