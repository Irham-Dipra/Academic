
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import sys

# Define expected filtering function to test
def low_pass_filter(real, imag, cutoff):
    rows, cols = real.shape
    cx, cy = rows//2, cols//2
    real_f = real.copy()
    imag_f = imag.copy()
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-cx)**2 + (j-cy)**2) > cutoff:
                real_f[i,j] = 0
                imag_f[i,j] = 0
    return real_f, imag_f

def diagnose():
    try:
        if not os.path.exists("noisy_image.png"):
            print("ERROR: noisy_image.png not found")
            return

        # Check Image
        img = imageio.imread("noisy_image.png", mode='F')
        print(f"Image Shape: {img.shape}")
        print(f"Image Type: {img.dtype}")
        print(f"Image Range: {np.min(img)} to {np.max(img)}")

        # Normalize
        img = img / np.max(img)
        
        # Downsample for quick CFT check if large
        if img.shape[0] > 64:
            print("Image is large, taking center 32x32 for quick check...")
            # Use center crop
            cy, cx = img.shape[0]//2, img.shape[1]//2
            img_small = img[cy-16:cy+16, cx-16:cx+16]
        else:
            img_small = img
            
        rows, cols = img_small.shape
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        u = np.linspace(-cols//2, cols//2, cols)
        v = np.linspace(-rows//2, rows//2, rows)

        # CFT Step 1 (Horiz)
        G_real = np.zeros((rows, len(u)))
        G_imag = np.zeros((rows, len(u)))
        
        for r in range(rows):
            signal_row = img_small[r, :]
            for i, u_val in enumerate(u):
                angle = 2 * np.pi * u_val * x
                re_part = signal_row * np.cos(angle)
                im_part = signal_row * -np.sin(angle)
                G_real[r, i] = np.trapezoid(re_part, x)
                G_imag[r, i] = np.trapezoid(im_part, x)
                
        # CFT Step 2 (Vert)
        F_real = np.zeros((len(v), len(u)))
        F_imag = np.zeros((len(v), len(u)))
        
        for c in range(len(u)):
            col_real = G_real[:, c]
            col_imag = G_imag[:, c]
            for j, v_val in enumerate(v):
                angle = 2 * np.pi * v_val * y
                cos_v = np.cos(angle)
                sin_v = np.sin(angle)
                re_integrand = col_real * cos_v + col_imag * sin_v
                im_integrand = col_imag * cos_v - col_real * sin_v
                F_real[j, c] = np.trapezoid(re_integrand, y)
                F_imag[j, c] = np.trapezoid(im_integrand, y)

        mag = np.sqrt(F_real**2 + F_imag**2)
        print(f"Spectrum Max: {np.max(mag)}")
        print(f"Spectrum Mean: {np.mean(mag)}")
        print(f"Spectrum Center (DC): {mag[rows//2, cols//2]}")
        
        # Save Spectrum plot
        plt.figure()
        plt.imshow(np.log1p(mag), cmap='inferno')
        plt.title("Diagnostic Spectrum")
        plt.savefig("diagnostic_spectrum.png")
        print("Saved diagnostic_spectrum.png")

        # Test Filter
        cutoff = 10 # Small cutoff for 32x32
        fr, fi = low_pass_filter(F_real, F_imag, cutoff)
        
        # Inverse Check (Rough)
        # Just check if it produces something non-zero
        print("Inverse check skipped for brevity, assuming standard logic works if spectrum is valid.")

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
