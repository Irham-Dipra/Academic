import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

class FourierSeries:
    def __init__(self, func, L, terms=10):

        self.func = func
        self.L = L
        self.terms = terms

    def calculate_a0(self, N=1000):

        x = np.linspace(-self.L, self.L, N) 
        y = self.func(x)

        try:
            integral = np.trapz(y, x)
        except AttributeError:
            integral = np.trapezoid(y, x)
        return (1 / (2 * self.L)) * integral

    def calculate_an(self, n, N=1000):

        x = np.linspace(-self.L, self.L, N)
        y = self.func(x) * np.cos(n * np.pi * x / self.L)
        try:
            integral = np.trapz(y, x)
        except AttributeError:
            integral = np.trapezoid(y, x)
        return (1 / self.L) * integral

    def calculate_bn(self, n, N=1000):

        x = np.linspace(-self.L, self.L, N)
        y = self.func(x) * np.sin(n * np.pi * x / self.L)
        try:
            integral = np.trapz(y, x)
        except AttributeError:
            integral = np.trapezoid(y, x)
        return (1 / self.L) * integral

    def approximate(self, x):

        a0 = self.calculate_a0()
        result = (a0 / 2) * np.ones_like(x)
        
        for n in range(1, self.terms + 1):
            an = self.calculate_an(n)
            bn = self.calculate_bn(n)
            result += an * np.cos(n * np.pi * x / self.L) + bn * np.sin(n * np.pi * x / self.L)
        
        return result

    def plot(self, ax, wave_type="square"):


        if wave_type == "cubic":
            x = np.linspace(-6, 6, 1000)
        else:
            x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
        

        original = self.func(x)
        

        approximation = self.approximate(x)


        ax.clear()
        ax.plot(x, original, label="Original Function", color="blue", alpha=0.5)
        ax.plot(x, approximation, label=f"Fourier Series (N={self.terms})", color="red", linestyle="--")
        

        if wave_type == "sawtooth":
            ax.set_ylim(-3.5, 3.5) 
        elif wave_type == "cubic":
            ax.set_ylim(-1.5, 1.5)
        elif wave_type == "pulse":
            ax.set_ylim(-0.5, 1.5)
        else:
            ax.set_ylim(-1.5, 1.5)
            

        if wave_type == "cubic":
            ax.set_xlim(-6, 6)
        else:
            ax.set_xlim(-4 * np.pi, 4 * np.pi)
        
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title(f"Fourier Series Approximation: {wave_type.replace('_', ' ').title()}")


def target_function(x, function_type="square"):

    if function_type == "square":
        return np.where(np.sin(x) > 0, 1, -1)

    elif function_type == "sawtooth":
        period = 2 * np.pi
        x_mod = (x + np.pi) % period - np.pi
        return x_mod

    elif function_type == "triangle":
        return (2 / np.pi) * np.arcsin(np.sin(x))

    elif function_type == "cubic":
        period = 2
        x_mod = (x + 1) % period - 1
        return x_mod ** 3
    
    elif function_type == "pulse":
        period = 2 * np.pi
        x_mod = np.abs((x + np.pi) % period - np.pi)
        return np.where(x_mod < 0.01, 1.0, 0.0)

    else:
        raise ValueError("Invalid function_type.")


def get_half_period(wave_type):

    if wave_type == "cubic":
        return 1.0
    else:
        return np.pi



if __name__ == "__main__":
    initial_terms = 1
    initial_wave = "square"
    L = get_half_period(initial_wave)


    fig_plot, ax_plot = plt.subplots(figsize=(10, 6))


    fig_widgets = plt.figure(figsize=(8, 4))
    current_func = lambda x: target_function(x, initial_wave)
    fs = FourierSeries(current_func, L, initial_terms)

    fs.plot(ax_plot, initial_wave)



    ax_radio = fig_widgets.add_axes([0.05, 0.1, 0.15, 0.3], facecolor='#f0f0f0')
    radio = RadioButtons(ax_radio, ('square', 'sawtooth', 'triangle', 'cubic', 'pulse'))

    ax_n = fig_widgets.add_axes([0.25, 0.6, 0.6, 0.1])

    slider_n = Slider(
        ax=ax_n,
        label='Harmonics (N)',
        valmin=1,
        valmax=500,
        valinit=initial_terms,
        valstep=1
    )



    def update(val):

        n = int(slider_n.val)
        wave_type = radio.value_selected
        

        fs.terms = n
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)
        

        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()

    def change_wave(label):

        wave_type = label
        

        fs.terms = int(slider_n.val)
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)
        

        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()


    slider_n.on_changed(update)
    radio.on_clicked(change_wave)

    plt.show()