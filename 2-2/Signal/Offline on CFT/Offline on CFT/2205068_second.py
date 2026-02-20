import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Using v2 to avoid deprecation warnings


class ContinuousImage:
    """
    Represents an image as a continuous 2D signal.
    """
    def __init__(self, image_path):
        self.image = imageio.imread(image_path, mode='F') # mode='F' for floating point/grayscale
        
        # Normalize pixel values to 0-1
        self.image = self.image / np.max(self.image)
        
        # Resize for performance if needed (Assignment implies small images or patience)
        # Using original shape.
        
        # Define continuous spatial axes [-1, 1]
        self.x = np.linspace(-1, 1, self.image.shape[1]) # Width (Cols)
        self.y = np.linspace(-1, 1, self.image.shape[0]) # Height (Rows)

    def show(self, title="Image"):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.image, cmap='gray', extent=[-1, 1, -1, 1])
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.show()


class CFT2D:
    """
    Computes 2D Continuous Fourier Transform
    using separability and numerical integration.
    """
    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x # Spatial x
        self.y = image_obj.y # Spatial y
        
        # Define Frequency Axes (u, v)
        self.u = np.linspace(-1, 1, 225)
        self.v = np.linspace(-1, 1, 225)

    def compute_cft(self):
        """
        Compute real and imaginary parts of 2D CFT using separable integration.
        """
        rows, cols = self.I.shape
        
        # --- Step 1: Horizontal Integration (over x) ---
        # G(y, u) = Integral( I(x,y) * e^(-j2pi*u*x) dx )
        # We need to store complex results. 
        # Since we can't use complex types easily, we store Real/Imag separately.
        
        G_real = np.zeros((rows, len(self.u)))
        G_imag = np.zeros((rows, len(self.u)))

        for r in range(rows):
            signal_row = self.I[r, :]
            for i, u_val in enumerate(self.u):
                angle = 2 * np.pi * u_val * self.x
                re_part = signal_row * np.cos(angle)
                im_part = signal_row * -np.sin(angle)
                
                G_real[r, i] = np.trapezoid(re_part, self.x)
                G_imag[r, i] = np.trapezoid(im_part, self.x)


        
        F_real = np.zeros((len(self.v), len(self.u)))
        F_imag = np.zeros((len(self.v), len(self.u)))

        for c in range(len(self.u)): # Iterate over columns of G (u-axis)
            col_real = G_real[:, c]
            col_imag = G_imag[:, c]
            
            for j, v_val in enumerate(self.v):
                # Kernel: e^(-j2pi * v * y)
                angle = 2 * np.pi * v_val * self.y
                cos_v = np.cos(angle)
                sin_v = np.sin(angle)
                
                re_integrand = col_real * cos_v + col_imag * sin_v
                im_integrand = col_imag * cos_v - col_real * sin_v
                
                F_real[j, c] = np.trapezoid(re_integrand, self.y)
                F_imag[j, c] = np.trapezoid(im_integrand, self.y)

        return F_real, F_imag

    def plot_magnitude(self):
        """
        Plot log-scaled magnitude spectrum.
        """
        """
        Plot log-scaled magnitude spectrum.
        """
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        
        # Log scale: log(1 + mag) to handle zeros
        log_mag = np.log(1 + magnitude)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(log_mag, cmap='viridis', extent=[self.u[0], self.u[-1], self.v[0], self.v[-1]])
        plt.title("CFT Magnitude Spectrum (Log Scale)")
        plt.xlabel("Frequency u")
        plt.ylabel("Frequency v")
        plt.colorbar()
        plt.show()


class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows//2, cols//2

        # Create a copy to avoid modifying original
        real_f = real.copy()
        imag_f = imag.copy()

        for i in range(rows):
            for j in range(cols):
                # Calculate distance from center (DC component)
                if np.sqrt((i-cx)**2 + (j-cy)**2) > cutoff:
                    real_f[i,j] = 0
                    imag_f[i,j] = 0
        return real_f, imag_f


class InverseCFT2D:
    """
    Reconstructs image from 2D frequency spectrum.
    """
    def __init__(self, real, imag, x, y):
        self.real = real 
        self.imag = imag
        self.x = x # Target spatial x
        self.y = y # Target spatial y
        
        self.u = np.linspace(-1, 1, 225)
        self.v = np.linspace(-1, 1, 225)

    def reconstruct(self):
        """
        Perform inverse 2D CFT using numerical integration.
        """
        # --- Step 1: Inverse Vertical Integration (over v) ---
        H_real = np.zeros((len(self.u), len(self.y)))
        H_imag = np.zeros((len(self.u), len(self.y)))
        
        # Iterate over u columns (frequency cols)
        for i in range(len(self.u)):
            col_real = self.real[:, i]
            col_imag = self.imag[:, i]
            
            # Integrate over v for each target spatial y
            for j, y_val in enumerate(self.y):
                angle = 2 * np.pi * self.v * y_val
                cos_v = np.cos(angle)
                sin_v = np.sin(angle)
                
                re_integrand = col_real * cos_v - col_imag * sin_v
                im_integrand = col_real * sin_v + col_imag * cos_v
                
                H_real[i, j] = np.trapezoid(re_integrand, self.v)
                H_imag[i, j] = np.trapezoid(im_integrand, self.v)

        I_final = np.zeros((len(self.y), len(self.x)))
        
        # Iterate over spatial rows y
        for j in range(len(self.y)):
            # Row in H corresponds to variation over u for fixed y
            row_real = H_real[:, j]
            row_imag = H_imag[:, j]
            
            for k, x_val in enumerate(self.x):
                angle = 2 * np.pi * self.u * x_val
                cos_u = np.cos(angle)
                sin_u = np.sin(angle)
                
                re_integrand = row_real * cos_u - row_imag * sin_u
                
                I_final[j, k] = np.trapezoid(re_integrand, self.u)

        return I_final


if __name__ == "__main__":
    img = ContinuousImage("noisy_image.png")
    img.show("Original Image")

    cft2d = CFT2D(img)
    
    cft2d = CFT2D(img)
    
    real, imag = cft2d.compute_cft()
    
    # Manual plot to avoid re-calculating inside plot_magnitude()
    magnitude = np.sqrt(real**2 + imag**2)
    plt.figure(figsize=(6,6))
    plt.imshow(np.log1p(magnitude), cmap='inferno')
    plt.title("2D CFT Magnitude Spectrum")
    plt.axis('off')
    plt.show()

    # Continue with filtering...
    filt = FrequencyFilter()
    real_f, imag_f = filt.low_pass(real, imag, cutoff=40)

    icft2d = InverseCFT2D(real_f, imag_f, img.x, img.y)
    denoised = icft2d.reconstruct()

    plt.figure(figsize=(6,6))
    plt.imshow(denoised, cmap='gray', extent=[-1, 1, -1, 1])
    plt.title("Reconstructed (Denoised) Image")
    plt.axis('off')
    plt.show()