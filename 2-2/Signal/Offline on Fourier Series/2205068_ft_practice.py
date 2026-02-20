import numpy as np
import matplotlib.pyplot as plt

# TODO: Define the functions for the different signals
def parabolic_function(x):
    """
    Implement a parabolic function that returns x^2 for -2 <= x <= 2, and 0 elsewhere
    """
    return np.where(np.abs(x) <= 2, x**2, 0)

def triangular_function(x):
    """
    Implement a triangular function that returns 1 - |x|/2 for -2 <= x <= 2, and 0 elsewhere
    """
    return np.where(np.abs(x) <= 2, 1 - np.abs(x)/2, 0)

def sawtooth_function(x):
    """
    Implement a sawtooth function that returns x + 2 for -2 <= x <= 2, and 0 elsewhere
    """
    return np.where(np.abs(x) <= 2, x + 2, 0)

def rectangular_function(x):
    """
    Implement a rectangular (step) function that returns 1 for -2 <= x <= 2, and 0 elsewhere
    """
    return np.where(np.abs(x) <= 2, 1, 0)

# TODO: Implement Fourier Transform using trapezoidal integration
def fourier_transform(signal, frequencies, sampled_times):
    """
    Compute the Fourier Transform of a signal using trapezoidal integration.
    
    Parameters:
    - signal: the input signal values
    - frequencies: array of frequencies to compute the transform at
    - sampled_times: time samples corresponding to the signal
    
    Returns:
    - real_part: real component of the FT
    - imag_part: imaginary component of the FT
    """
    real_part = []
    imag_part = []
    
    for f in frequencies:
        complex_exp = np.exp(-1j * 2 * np.pi * f * sampled_times)
        integrand = signal * complex_exp
        ft_value = np.trapz(integrand, sampled_times)
        real_part.append(np.real(ft_value))
        imag_part.append(np.imag(ft_value))
    
    return np.array(real_part), np.array(imag_part)

# TODO: Implement Inverse Fourier Transform
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    """
    Reconstruct the original signal from its Fourier Transform.
    
    Parameters:
    - ft_signal: tuple of (real_part, imag_part) from Fourier Transform
    - frequencies: array of frequencies
    - sampled_times: time samples
    
    Returns:
    - reconstructed_signal: the reconstructed signal
    """
    ft_complex = ft_signal[0] + 1j * ft_signal[1]
    reconstructed_signal = []
    for t in sampled_times:
        complex_exp = np.exp(1j * 2 * np.pi * frequencies * t)
        integrand = ft_complex * complex_exp
        ift_value = np.trapz(integrand, frequencies)
        reconstructed_signal.append(np.real(ift_value))
    return np.array(reconstructed_signal)

# TODO: Implement Inverse Fourier Transform for computing derivatives
def inverse_fourier_transform_for_derivative(ft_signal, frequencies, sampled_times):
    """
    Compute the derivative of a signal using Fourier Transform.
    The derivative in frequency domain is: 2j*pi*f * F(f)
    
    Parameters:
    - ft_signal: tuple of (real_part, imag_part) from Fourier Transform
    - frequencies: array of frequencies
    - sampled_times: time samples
    
    Returns:
    - derivative_signal: the computed derivative
    """
    ft_complex = ft_signal[0] + 1j * ft_signal[1]
    derivative_in_freq = ft_complex * (1j * 2 * np.pi * frequencies)
    derivative_signal = []
    for t in sampled_times:
        complex_exp = np.exp(1j * 2 * np.pi * frequencies * t)
        integrand = derivative_in_freq * complex_exp
        val = np.trapz(integrand, frequencies)
        derivative_signal.append(np.real(val))
        
    return np.array(derivative_signal)

# Define sampled times and frequency ranges
sampled_times = np.linspace(-5, 5, 1000)
frequencies_list = [np.linspace(-1, 1, 500), np.linspace(-2, 2, 500), np.linspace(-5, 5, 500)]

functions = {
    "Parabolic Function": parabolic_function,
    "Triangular Function": triangular_function,
    "Sawtooth Function": sawtooth_function,
    "Rectangular Function": rectangular_function,
}

# Plotting for each function
for function_name, func in functions.items():
    y_values = func(sampled_times)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sampled_times, y_values, label=f"Original {function_name}")
    plt.title(f"Original {function_name}")
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

    # Fourier Transform and Frequency Spectrum
    for freq_range in frequencies_list:
        ft_signal = fourier_transform(y_values, freq_range, sampled_times)
        reconstructed_signal = inverse_fourier_transform(ft_signal, freq_range, sampled_times)

        plt.figure(figsize=(10, 6))
        plt.plot(freq_range, np.sqrt(ft_signal[0]**2 + ft_signal[1]**2), label="Frequency Spectrum")
        plt.title(f"Frequency Spectrum for {function_name} (Freq Range {freq_range[0]} to {freq_range[-1]})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid()
        plt.show()

        # Reconstructed Signal
        plt.figure(figsize=(10, 6))
        plt.plot(sampled_times, y_values, label=f"Original {function_name}", color='blue')
        plt.plot(sampled_times, reconstructed_signal, label=f"Reconstructed {function_name}", color='red', linestyle='--')
        plt.title(f"Reconstructed {function_name} (Freq Range {freq_range[0]} to {freq_range[-1]})")
        plt.xlabel("Time (t)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()

        # Derivative using Fourier Transform
        derivative_signal = inverse_fourier_transform_for_derivative(ft_signal, freq_range, sampled_times)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sampled_times, derivative_signal, label=f"Derivative of {function_name}", color='green')
        plt.title(f"Derivative of {function_name} using Fourier Transform (Freq Range {freq_range[0]} to {freq_range[-1]})")
        plt.xlabel("Time (t)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()
