import numpy as np
import matplotlib.pyplot as plt
from wiener_filtering import my_wiener_filter


def plot_region_optimal(k_opt: float, y: np.ndarray, h: np.ndarray, x: np.ndarray):
    k_range = np.linspace(k_opt / 2, k_opt / 2 + k_opt, 1000)
    J_k_sum_values = []

    for k in k_range:
        x_hat = my_wiener_filter(y, h, k)  # Apply the Wiener filter
        J_k = (x - x_hat) ** 2  # Calculate the squared error
        J_k_sum_values.append(J_k.sum())  # Add the sum of squared errors to the list

    # Convert the list of J_k sums to a numpy array for plotting
    J_k_sum_values = np.array(J_k_sum_values)

    # Find the index of k_opt in k_range
    optimal_index = np.argmin(np.abs(k_range - k_opt))
    optimal_J_k_sum = J_k_sum_values[optimal_index]

    # Plotting
    plt.figure()
    plt.scatter(
        k_opt, optimal_J_k_sum, color="red"
    )  # Mark the optimal k with a red dot
    plt.plot(k_range, J_k_sum_values)
    plt.xlabel("k")
    plt.ylabel("J_k values in range [k_opt/2, k_opt/2 + k_opt]")
    plt.title("J_k values vs k in optimal region")
