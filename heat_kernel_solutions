nu = 1.0

alphas = [0.25, 0.75]

times = [0.05, 0.10, 0.20, 0.40] # Choose arbitrary times

xmin, xmax = -10.0, 10.0 # Approximate over large, continuous domain

N = 1000 # Choose arbitrary spatial grid size

x = np.linspace(xmin, xmax, N)
dx = x[1] - x[0]

# Define initial conditions from Task 4
def u0_1(x):
    return np.sin(x)

def u0_2(x):
    return np.exp(-x ** 2)

initial_conditions = {r"$u_0(x)=\sin x$": u0_1, r"$u_0(x)=e^{-x^2}$": u0_2,}

# Rewrite Def. 1 from Mao-Karniadakis using project notation
# I_x^alpha f(x) = 1/Gamma(alpha) int_{-infty}^x f(s)/(x-s)^(1-alpha) ds
def fractional_integral(f, alpha):
    I_alpha_f = np.zeros_like(f)

    for i in range(1, len(f)):
        left_indices = np.arange(i)
        k = i - left_indices
        
        weights = k ** alpha - (k - 1) ** alpha
        I_alpha_f[i] = np.sum(weights * f[left_indices])

    I_alpha_f *= dx ** alpha / gamma(alpha + 1)

    return I_alpha_f

# Rewrite Eq. 2 from project description
# D_x^alpha f = I_x^(1-alpha) f_x
def fractional_derivative(f, alpha):
    f_x = np.gradient(f, dx, edge_order = 2)
    D_alpha_f = fractional_integral(f_x, 1.0 - alpha)

    return D_alpha_f

# Define heat kernel solution from Eq. 5
def heat_kernel_solution(theta0, t):
    X, Y = np.meshgrid(x, x, indexing = "ij") # Use matrix indexing of output

    heat_kernel = (1.0 / np.sqrt(4.0 * np.pi * nu * t)) * np.exp(
        -((X - Y) ** 2) / (4.0 * nu * t))

    theta = np.trapezoid(heat_kernel * theta0[None, :], x, axis=1)

    return theta

# Compute exact solution using the fractional Cole-Hopf transform
def exact_solution(u0_function, alpha):
    u0 = u0_function(x)

    # Compute I_x^alpha u_0(x)
    I_alpha_u0 = fractional_integral(u0, alpha)

    log_theta0 = -(1.0 / (2.0 * nu)) * I_alpha_u0 # Shift by constant for numerical stability

    log_theta0 -= np.max(log_theta0)
    theta0 = np.exp(log_theta0)
    theta0 = np.maximum(theta0, 1e-300) # Prevent exact zeros

    u_values = {}

    for t in times:
        theta = heat_kernel_solution(theta0, t) # Compute using the heat kernel
        
        D_alpha_theta = fractional_derivative(theta, alpha) # Compute D_x^alpha theta(x,t)

        u_values[t] = -2.0 * nu * D_alpha_theta / theta

    return u_values

# Compute solutions for each initial condition, alpha
solutions = {}

for initial_condition_name, u0_function in initial_conditions.items():
    solutions[initial_condition_name] = {}

    for alpha in alphas:
        solutions[initial_condition_name][alpha] = exact_solution(u0_function, alpha)

# Plot u(x,t) for each initial condition, alpha
for initial_condition_name, u0_function in initial_conditions.items():
    fig, axes = plt.subplots(1, 2, figsize = (12, 4), sharey=True)

    for ax, alpha in zip(axes, alphas):
        ax.plot(x, u0_function(x), "--", label = r"$u_0(x)$")

        for t in times:
            ax.plot(x, solutions[initial_condition_name][alpha][t], label = f"t = {t}")

        ax.set_title(f"{initial_condition_name}, " + rf"$\alpha = {alpha}$")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u(x, t)$")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# Plot comparisons across times
for initial_condition_name in initial_conditions.keys():
    fig, axes = plt.subplots(2, 2, figsize = (10, 7), sharex = True, sharey = True)
    axes = axes.flatten()

    for ax, t in zip(axes, times):
        for alpha in alphas:
            ax.plot(x, solutions[initial_condition_name][alpha][t],
                label = rf"$\alpha = {alpha}$")

        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u(x, t)$")
        ax.grid(True)
        ax.legend()

    fig.suptitle(initial_condition_name)
    plt.tight_layout()
    plt.show()
