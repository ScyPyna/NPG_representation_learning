import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from joblib import Parallel, delayed
import os
from PIL import Image
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for graphene layers")

    # Parametri principali
    parser.add_argument("--output_dir", type=str, default="dataset", help="Directory for output files")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--n_images_per_m", type=int, default=2, help="Number of images per m value")
    parser.add_argument("--m_values", type=int, nargs="+", default=[2, 3, 4], help="List of m values to generate images for")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation (default: None)")

    args = parser.parse_args()

    # Set the random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")


    # Passa i parametri al codice esistente
    output_dir = args.output_dir
    n_jobs = args.n_jobs
    n_images_per_m = args.n_images_per_m
    m_values = args.m_values

#----------------------------------------FUNZIONI------------------------------------------------

    def phi_par(x, y):
        return 1 - (1/3)*np.cos((2*np.pi)/a * (x + (y/(np.sqrt(3))))) - (1/3)*np.cos((2*np.pi)/a * (x - (y/(np.sqrt(3))))) - (1/3)*np.cos((4*np.pi*y)/(np.sqrt(3)*a))

    def generate_valid_twist_angles(m, min_difference=4, angle_range=(4, 30)):
        min_diff_rad = np.radians(min_difference)
        angle_min_rad = np.radians(angle_range[0])
        angle_max_rad = np.radians(angle_range[1])

        angles = []
        while len(angles) < m:
            new_angle = np.random.uniform(angle_min_rad, angle_max_rad)
            if all(abs(new_angle - existing) >= min_diff_rad for existing in angles):
                angles.append(new_angle)

        return angles

    def rotate_coordinates(x, y, theta):
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        return x_rot, y_rot

    def translate_coordinates(x, y, tx, ty):
        return x + tx, y + ty

    def n_density(x, y, z, m, l, n0, translations, theta, z_lay):
        sum_phi = 0
        for i in range(m):
            X_trans, Y_trans = translate_coordinates(x, y, translations[i][0], translations[i][1])
            X_rot, Y_rot = rotate_coordinates(X_trans, Y_trans, theta[i])
            sum_phi += phi_par(X_rot, Y_rot) * np.exp(-np.abs(z - z_lay[i]) / l)
        return (n0 / m) * sum_phi

    def target_function(z, x, y, n0, ratio, m, l, translations, theta, z_lay):
        n_val = n_density(x, y, z, m, l, n0, translations, theta, z_lay)
        return n_val / n0 - ratio

    def calculate_z(i, j, X, Y, n0, ratio, z_min, z_max, m, l, translations, theta, z_lay):
        try:
            return bisect(target_function, z_min, z_max, args=(X[i, j], Y[i, j], n0, ratio, m, l, translations, theta, z_lay), maxiter=1000, xtol=1e-6)
        except ValueError:
            return np.nan

    def refine_nan(i, j, X, Y, Z, refine_step, refine_grid_points, n0, ratio, z_min, z_max, m, l, translations, theta, z_lay):
        if np.isnan(Z[i, j]):
            x_local = np.linspace(X[i, j] - refine_step, X[i, j] + refine_step, refine_grid_points)
            y_local = np.linspace(Y[i, j] - refine_step, Y[i, j] + refine_step, refine_grid_points)
            X_local, Y_local = np.meshgrid(x_local, y_local)
            z_values = []
            for xl, yl in zip(X_local.ravel(), Y_local.ravel()):
                try:
                    z_local = bisect(target_function, z_min, z_max, args=(xl, yl, n0, ratio, m, l, translations, theta, z_lay), maxiter=500, xtol=1e-6)
                    z_values.append(z_local)
                except ValueError:
                    pass
            if z_values:
                return np.mean(z_values)
        return Z[i, j]

    def plot_data(data, extent, title, xlabel, ylabel, cbar_label, cmap='viridis', vmin=None, vmax=None, log_scale=False):
        plt.figure(figsize=(8, 6))
        if log_scale:
            plt.imshow(np.log10(data), extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            plt.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar = plt.colorbar()
        cbar.set_label(cbar_label)
        plt.show()

    def calculate_relative_twist_angles(theta):
        base_angle = theta[0]
        relative_angles = [angle - base_angle for angle in theta]
        return relative_angles
    

    #----------------------------------------SCRIPT START--------------------------------------------------

    # GLOBAL VARIABLES DEFINITION---------------------------------------------------------
    a = 2.46
    n0 = 0.0002
    l = 5
    d = 3.34
    z_min = 0.0
    z_max = 200.0
    ratio = 0.05
    current = 1 / ratio
    z0 = l * np.log(current)
    L_x = 200.0
    L_y = 200.0
    N_x = 512
    N_y = 512
    x = np.linspace(-L_x / 2, L_x / 2, N_x)
    y = np.linspace(-L_y / 2, L_y / 2, N_y)
    epsilon = d/100  #Scelta di epsilon come 1/100 della distanza interlayer
    X, Y = np.meshgrid(x, y)
    refine_step = 0.5
    refine_grid_points = 5
    #--------------------------------------------------------------------------------------
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    for idx, m in enumerate(m_values):
        for image_idx in range(n_images_per_m):
            z_lay = -(np.arange(m) * d)
            z_density = np.max(z_lay) + epsilon
            translations = np.random.uniform(-a, a, (m, 2))
            theta = generate_valid_twist_angles(m)
            relative_theta = calculate_relative_twist_angles(theta)

            Z_flat = Parallel(n_jobs=n_jobs)(
                delayed(calculate_z)(i, j, X, Y, n0, ratio, z_min, z_max, m, l, translations, theta, z_lay)
                for i in range(X.shape[0]) for j in range(X.shape[1])
            )
            Z = np.array(Z_flat).reshape(X.shape)

            Z_refined_flat = Parallel(n_jobs=n_jobs)(
                delayed(refine_nan)(i, j, X, Y, Z, refine_step, refine_grid_points, n0, ratio, z_min, z_max, m, l, translations, theta, z_lay)
                for i in range(X.shape[0]) for j in range(X.shape[1])
            )
            Z = np.array(Z_refined_flat).reshape(X.shape)

            density = n_density(X, Y, z_density, m, l, n0, translations, theta, z_lay)

            n_total_fft = np.fft.fftshift(np.fft.fft2(density))
            n_total_fft_magnitude = np.abs(n_total_fft)
            Z_fft = np.fft.fftshift(np.fft.fft2(Z))
            Z_fft_magnitude = np.abs(Z_fft)

            flattened_n_fft = n_total_fft_magnitude.flatten()
            flattened_z_fft = Z_fft_magnitude.flatten()
            mean_n_flat = np.mean(flattened_n_fft)
            std_n_flat = np.std(flattened_n_fft)
            mean_z_flat = np.mean(flattened_z_fft)
            std_z_flat = np.std(flattened_z_fft)

            max_index_n_flat = np.argmax(flattened_n_fft)
            max_index_z_flat = np.argmax(flattened_z_fft)
            max_index_n_2d = np.unravel_index(max_index_n_flat, n_total_fft_magnitude.shape)
            max_index_z_2d = np.unravel_index(max_index_z_flat, Z_fft_magnitude.shape)

            clean_density_fft = np.copy(n_total_fft_magnitude)
            clean_density_fft[max_index_n_2d] = mean_n_flat
            clean_z_fft = np.copy(Z_fft_magnitude)
            clean_z_fft[max_index_z_2d] = mean_z_flat

            k_x = np.fft.fftshift(np.fft.fftfreq(N_x, d=L_x / N_x)) * 2 * np.pi
            k_y = np.fft.fftshift(np.fft.fftfreq(N_y, d=L_y / N_y)) * 2 * np.pi

            dataset_dir = os.path.join(output_dir, f"m_{m}")
            os.makedirs(dataset_dir, exist_ok=True)

            density_path = os.path.join(dataset_dir, f"density_{image_idx}.npy")
            Z_path = os.path.join(dataset_dir, f"Z_{image_idx}.npy")
            Z_fft_path = os.path.join(dataset_dir, f"Z_fft_{image_idx}.npy")
            image_path = os.path.join(dataset_dir, f"Z_fft_image_{image_idx}.png")

            np.save(density_path, density)
            np.save(Z_path, Z)
            np.save(Z_fft_path, clean_z_fft)

            plt.imsave(
                os.path.join(dataset_dir, f"Z_fft_image_{image_idx}.png"),
                clean_z_fft, 
                cmap='inferno', 
                vmin=0, 
                vmax=mean_z_flat + std_z_flat, 
                origin='lower'
            )

            # Creazione file images.txt
            images_txt_path = os.path.join(dataset_dir, "images.txt")
            with open(images_txt_path, "a") as f:
                f.write(f"density_{image_idx}.npy\n")
                f.write(f"Z_{image_idx}.npy\n")
                f.write(f"Z_fft_{image_idx}.npy\n")
                f.write(f"Z_fft_image_{image_idx}.png\n")

            #Salvataggio metadata
            metadata.append({
                "id": len(metadata),
                "path": dataset_dir,
                "m": m,
                "y1": 1 if m > 1 else 0,
                "y2": 1 if m > 2 else 0,
                "y3": 1 if m > 3 else 0,
                "theta1": relative_theta[1] if m > 1 else 0,
                "theta2": relative_theta[2] if m > 2 else 0,
                "theta3": relative_theta[3] if m > 3 else 0,
            })
    
    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("id,path,m,y1,y2,y3,theta1,theta2,theta3\n")
        for entry in metadata:
            f.write(f"{entry['id']},{entry['path']},{entry['m']},{entry['y1']},{entry['y2']},{entry['y3']},{entry['theta1']},{entry['theta2']},{entry['theta3']}\n")