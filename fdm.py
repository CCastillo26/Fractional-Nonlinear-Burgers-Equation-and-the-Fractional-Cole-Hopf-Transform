# Approximate u_xx using central difference method
def fd_uxx(u):
    uxx = np.zeros_like(u)

    # Compute central difference in the interior
    uxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx ** 2

    # Compute second derivatives at boundaries
    uxx[0] = (u[2] - 2.0 * u[1] + u[0]) / dx ** 2
    uxx[-1] = (u[-1] - 2.0 * u[-2] + u[-3]) / dx ** 2

    return uxx

# Define the right-hand side of the PDE
def pde_rhs(u, alpha):
    flux = 0.5 * u ** 2

    # Reuse D_x^alpha from Task 5
    frac_flux = fractional_derivative(flux, alpha)
    uxx = fd_uxx(u)
    rhs = nu * uxx - frac_flux

    return rhs

# Use RK4 for time stepping
def rk4_approx(u, dt, alpha):
    k1 = pde_rhs(u, alpha)
    k2 = pde_rhs(u + 0.5 * dt * k1, alpha)
    k3 = pde_rhs(u + 0.5 * dt * k2, alpha)
    k4 = pde_rhs(u + dt * k3, alpha)

    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# Solve the fractional Burgers equation
def fd_solve(u0_function, alpha, dt=1e-4):
    u = u0_function(x).copy()

    u_time = {}
    t_now = 0.0

    for t_target in times:
        while t_now < t_target - 1e-12:
            dt_now = min(dt, t_target - t_now)

            u = rk4_approx(u, dt_now, alpha)

            t_now += dt_now

        u_time[t_target] = u.copy()

    return u_time

    # Compute finite-difference solutions
fd_sol = {}

for ic_name, u0_function in initial_conditions.items():
    fd_sol[ic_name] = {}

    for alpha in alphas:
        print(f"Finite difference for {ic_name}, alpha = {alpha}")

        fd_sol[ic_name][alpha] = fd_solve(u0_function, alpha)

# Compute relative L2 error against exact solutions
fd_err = {}

for ic_name in initial_conditions.keys():
    fd_err[ic_name] = {}

    for alpha in alphas:
        fd_err[ic_name][alpha] = {}

        for t in times:
            u_exact = solutions[ic_name][alpha][t]
            u_num = fd_sol[ic_name][alpha][t]

            err = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)

            fd_err[ic_name][alpha][t] = err

            print(
                ic_name,
                "alpha =", alpha,
                "t =", t,
                "relative error =", err)

# Plot finite-difference solution against exact solution
for ic_name in initial_conditions.keys():
    for alpha in alphas:
        fig, axes = plt.subplots(2, 2, figsize = (10, 7), sharex = True, sharey = True)
        axes = axes.flatten()

        for ax, t in zip(axes, times):
            u_exact = solutions[ic_name][alpha][t]
            u_num = fd_sol[ic_name][alpha][t]

            ax.plot(x, u_exact, label = "Exact")
            ax.plot(x, u_num, "--", label = "Finite difference")

            ax.set_title(f"t = {t}")
            ax.set_xlabel("x")
            ax.set_ylabel(r"$u(x,t)$")
            ax.grid(True)
            ax.legend()

        fig.suptitle(f"{ic_name}, " + rf"$\alpha = {alpha}$")
        plt.tight_layout()
        plt.show()

# Plot error versus time
for ic_name in initial_conditions.keys():
    plt.figure(figsize = (7, 4))

    for alpha in alphas:
        err_time = [fd_err[ic_name][alpha][t] for t in times]

        plt.plot(times, err_time, marker = "o", label = rf"$\alpha = {alpha}$")

    plt.xlabel("t")
    plt.ylabel("Relative L2 error")
    plt.title(f"Error versus time: {ic_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot error versus alpha
for ic_name in initial_conditions.keys():
    plt.figure(figsize = (7, 4))

    for t in times:
        err_alpha = [fd_err[ic_name][alpha][t] for alpha in alphas]

        plt.plot(alphas, err_alpha, marker = "o", label = f"t = {t}")

    plt.xlabel(r"$\alpha$")
    plt.ylabel("Relative L2 error")
    plt.title(f"Error versus alpha: {ic_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
