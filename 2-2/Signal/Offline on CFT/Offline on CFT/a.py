import numpy as np
import matplotlib.pyplot as plt

# ... [Classes ContinuousSignal, SignalGenerator, CompositeSignal remain mostly unchanged] ...

# CHANGE 1: Helper Class for arbitrary data
class CustomSignal(ContinuousSignal):
    """
    New Class: Allows us to treat a raw numpy array as a 'Signal' object.
    Why: We calculate y(t) as a simple array multiplication (x * e^j...). 
    We need to wrap that array back into an object so CFTAnalyzer can read it.
    """
    def __init__(self, t, signal_array):
        super().__init__(t)
        self.signal_array = signal_array
        
    def values(self):
        return self.signal_array

class CFTAnalyzer:
    def __init__(self, signal_obj, t, frequencies):
        self.signal_obj = signal_obj
        self.t = t
        self.frequencies = frequencies
        self.x_t = self.signal_obj.values()

    def compute_cft(self):
        """
        CHANGE 2: Updated Math to handle Complex Signals
        Old way: Separated cos (Real) and sin (Imag).
        New way: Uses Euler's formula e^(-j*theta) directly.
        Why: The assignment creates a complex signal y(t). The old logic might 
        drop the imaginary interaction terms. This is the general definition of CFT.
        """
        spectrum = []
        
        # Helper to support both old and new numpy versions
        integrate_func = getattr(np, 'trapezoid', np.trapz)

        for f in self.frequencies:
            # Formula: x(t) * e^(-j 2 pi f t)
            # This handles both magnitude and phase correctly for ANY signal.
            complex_exponential = np.exp(-1j * 2 * np.pi * f * self.t)
            integrand = self.x_t * complex_exponential
            
            val = integrate_func(integrand, self.t)
            spectrum.append(val)
        
        spectrum = np.array(spectrum)
        # We still return separate Real/Imag parts to match your template structure
        return np.real(spectrum), np.imag(spectrum)
    
    def get_spectrum_data(self):
        """
        CHANGE 3: Helper to get Magnitude and Phase
        Why: The assignment specifically asks for Magnitude |Y(f)| and Angle /_Y(f).
        """
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)
        return magnitude, phase

# ... [InverseCFT remains unchanged] ...

if __name__ == "__main__":
    # --- CHANGE 4: Assignment Parameters ---
    # [cite_start]The pdf specifies t in [-5, 5] with 2000 samples [cite: 11]
    t = np.linspace(-5, 5, 2000)
    # [cite_start]The pdf specifies f in [-10, 10] with 1000 samples [cite: 16]
    frequencies = np.linspace(-10, 10, 1000)
    
    # [cite_start]Given constants from PDF [cite: 7, 8]
    f0 = 10  # Shift amount
    a = 10   # Compression factor

    gen = SignalGenerator(t)

    # --- CHANGE 5: Defining x(t) ---
    # [cite_start]"x(t) = Square(t) + Triangle(t)" [cite: 6]
    x_sig = CompositeSignal(t)
    x_sig.add_component(gen.square(1, 1))
    x_sig.add_component(gen.triangle(1, 1))

    # --- CHANGE 6: Creating y(t) ---
    # The prompt asks for two effects:
    # [cite_start]1. Compress time by a=10 -> x(at) [cite: 8]
    # [cite_start]2. Shift phase by 2*pi*f0*t -> Multiply by e^(j*2*pi*f0*t) [cite: 7]
    
    # Step A: Create x(at)
    # We simulate "time compression" by increasing the frequency of the components by 'a'.
    # If a wave was 1Hz, it is now 10Hz (happens 10x faster).
    comp_square = gen.square(1, 1 * a) 
    comp_triangle = gen.triangle(1, 1 * a)
    x_compressed = comp_square + comp_triangle
    
    # Step B: Apply Phase Shift
    # This creates the complex signal.
    phase_shift_term = np.exp(1j * 2 * np.pi * f0 * t)
    y_values = x_compressed * phase_shift_term
    
    # Wrap it so we can calculate its CFT
    y_sig = CustomSignal(t, y_values)

    # --- CHANGE 7: Numerical Verification ---
    # The PDF asks to verify: |Y(f)| = 1/|a| * [cite_start]|X((f-f0)/a)| [cite: 13, 15]
    
    # Calculate LHS (Left Hand Side): The actual CFT of y(t)
    analyzer_y = CFTAnalyzer(y_sig, t, frequencies)
    mag_y, phase_y = analyzer_y.get_spectrum_data()

    # Calculate RHS (Right Hand Side): The theoretical prediction
    # Strategy: Instead of shifting the array indices (which is messy), 
    # we just ask the CFTAnalyzer to compute x(t) at the "shifted frequencies".
    mapped_frequencies = (frequencies - f0) / a
    
    analyzer_x_mapped = CFTAnalyzer(x_sig, t, mapped_frequencies)
    mag_x_mapped, phase_x_mapped = analyzer_x_mapped.get_spectrum_data()
    
    # Apply the amplitude scaling 1/|a|
    mag_y_theoretical = (1 / np.abs(a)) * mag_x_mapped
    phase_y_theoretical = phase_x_mapped

    # ... [Plotting code follows standard matplotlib pattern] ...

    # --- CHANGE 8: Error Analysis ---
    # [cite_start]Implementing the specific MSE formula provided in the PDF [cite: 18]
    N = len(frequencies)
    ise_mag = (1/N) * np.sum((mag_y - mag_y_theoretical)**2)
    
    print(f"ISE Magnitude: {ise_mag}")