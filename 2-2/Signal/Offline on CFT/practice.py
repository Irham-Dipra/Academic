import numpy as np
import matplotlib.pyplot as plt


class ContinuousSignal:

    def __init__(self, t):
        self.t = t

    def values(self):
        """
        Returns the signal values evaluated over time axis t.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def plot(self, title="signal"):
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.values())
        plt.xlabel("time(t)")
        plt.ylabel("amplitude")
        plt.title(title)
        plt.grid(True)
        plt.show()


class SignalGenerator(ContinuousSignal):
    def __init__(self, t):
        super().__init__(t)
    
    def sine(self, amplitude, frequency):
        """Generate a sine wave: A sin(2*pi*f*t)"""
        return amplitude * np.sin(2 * np.pi * frequency * self.t)

    def cosine(self, amplitude, frequency):
        """Generate a cosine wave: A cos(2*pi*f*t)"""
        return amplitude * np.cos(2 * np.pi * frequency * self.t)

    def square(self, amplitude, frequency):
        """Generate a square wave: A * sign(sin(2*pi*f*t))"""
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * self.t))

    def sawtooth(self, amplitude, frequency):
        """Generate a sawtooth wave."""
        ft = frequency * self.t
        return amplitude * (2 * (ft - np.floor(0.5 + ft)))

    def triangle(self, amplitude, frequency):
        """Generate a triangle wave."""
        return (2 * amplitude / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * self.t))

    def cubic(self, coefficient):
        """Generate a cubic polynomial signal: c * t^3"""
        return coefficient * (self.t ** 3)

    def parabolic(self, coefficient):
        """Generate a parabolic signal: c * t^2"""
        return coefficient * (self.t ** 2)

    def rectangular(self, width):
        """Generate a rectangular window centered at t=0."""
        return np.where(np.abs(self.t) <= width / 2, 1.0, 0.0)

    def pulse(self, start, end):
        """Generate a finite pulse active between start and end."""
        # 1 if start <= t <= end, else 0
        return np.where((self.t >= start) & (self.t <= end), 1.0, 0.0)
    
class CompositeSignal(ContinuousSignal):
    def __init__(self, t):
        super().__init__(t)
        self.components = []

    def add_component(self, signal_array):    
        self.components.append(signal_array)

    def values(self):
        if not self.components:
            return np.zeros_like(self.t)
        
        return sum(self.components)
    
class CFTAnalyzer:

    def __init__(self, signal_obj, t, frequencies):
        self.signal_obj = signal_obj
        self.t = t
        self.frequencies = frequencies
        self.x_t = self.signal_obj.values()

    def compute_cft(self):
        real_part = []
        imag_part = []

        for f in self.frequencies:
            integrand_real = self.x_t * np.cos(2*np.pi*f*self.t)
            integrand_imag = self.x_t * -np.sin(2 * np.pi * f * self.t)

            val_real = np.trapezoid(integrand_real, self.t)
            val_imag = np.trapezoid(integrand_imag, self.t)

            real_part.append(val_real)
            imag_part.append(val_imag)
        
        return np.array(real_part), np.array(imag_part)
    
    def plot_spectrum(self):
        """
        Plot magnitude spectrum of the signal.
        Magnitude = sqrt(Real^2 + Imag^2)
        """
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)

        plt.figure(figsize=(10, 4))
        plt.plot(self.frequencies, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Frequency Spectrum (Magnitude)")
        plt.grid(True)
        plt.show()

class InverseCFT:
    """
    Reconstructs time-domain signal using ICFT.
    """
    def __init__(self, spectrum, frequencies, t):
        self.real_spec, self.imag_spec = spectrum # Tuple unpacking
        self.frequencies = frequencies
        self.t = t

    def reconstruct(self):
        """
        Perform inverse CFT using numerical integration.
        """
        reconstructed_signal = []

        # We need to integrate over Frequency (df) for EACH time point t
        for time_point in self.t:
            # Real part of integrand: Re(f)*cos(wt) - Im(f)*sin(wt)
            integrand = (self.real_spec * np.cos(2 * np.pi * self.frequencies * time_point) - 
                         self.imag_spec * np.sin(2 * np.pi * self.frequencies * time_point))
            
            # Integrate over frequency
            val = np.trapezoid(integrand, self.frequencies)
            reconstructed_signal.append(val)

        return np.array(reconstructed_signal)
    
if __name__ == "__main__":
    t =  np.linspace(-4, 4, 3000)
    gen = SignalGenerator(t)

    composite = CompositeSignal(t)

    composite.add_component(gen.square(1, 1))        # Square wave 1 Hz
    composite.add_component(gen.triangle(1,1))

    composite.plot("Composite Signal") 

    frequencies = np.linspace(-10, 10, 1000) # Range matches the 3rd image filename
    cft = CFTAnalyzer(composite, t, frequencies)

    cft.plot_spectrum()

    icft = InverseCFT(cft.compute_cft(), frequencies, t)
    x_rec = icft.reconstruct()

    plt.figure(figsize=(10, 6))
    plt.plot(t, composite.values(), label="Original")
    plt.plot(t, x_rec, '--', label="Reconstructed")
    plt.legend()
    plt.title("Reconstruction using ICFT")
    plt.grid(True)
    plt.show()
    