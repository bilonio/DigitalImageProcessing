from wiener_filtering import my_wiener_filter

K_MAX = 1024


def find_optimal_k(x, y, h):
    k_opt = 1
    x_hat = my_wiener_filter(y, h, k_opt)
    J = (x - x_hat) ** 2
    J_min = J.min()
    for k in range(1, K_MAX):
        x_hat = my_wiener_filter(y, h, k)
        J_k = (x - x_hat) ** 2
        if J_k.min() < J_min:
            J_min = J_k.min()
            k_opt = k
    return k_opt, J_min
