torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Copy layers, as in Lecture 8 notebook
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1] # input = (x,t), output = theta(x,t)

N_u = 100 # Number of known data points
N_f = 10000 # Number of residual, collocation points
Nmax = 20000 # Maximum number of training iterations

lr = 1e-3

T_final = max(times)

# Define neural network for theta(x,t)
class ThetaPINN(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()

        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)

        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])

            # Use Xavier/He-style scaling, as in Lecture 8 notebook
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

            self.layers.append(layer)

    def forward(self, X):
        # Normalize input to [-1,1] to improve training stability
        A = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        for layer in self.layers[:-1]:
            A = torch.tanh(layer(A))

        return self.layers[-1](A)

# Build theta_0(x) from Task 4
def theta0_x(u0_function, alpha):
    u0 = u0_function(x)

    I_alpha_u0 = fractional_integral(u0, alpha)

    log_theta0 = -(1.0 / (2.0 * nu)) * I_alpha_u0
    log_theta0 -= np.max(log_theta0)

    theta0 = np.exp(log_theta0)

    return theta0

# Define residual for the heat equation
def heat_residual(model, X_f):
    X_f = X_f.detach().requires_grad_(True)
    theta = model(X_f)

    theta = model(X_f)

    grad_theta = torch.autograd.grad(
        theta,
        X_f,
        grad_outputs = torch.ones_like(theta),
        create_graph = True)[0]

    theta_x = grad_theta[:, 0:1]
    theta_t = grad_theta[:, 1:2]

    grad_theta_x = torch.autograd.grad(
        theta_x,
        X_f,
        grad_outputs = torch.ones_like(theta_x),
        create_graph = True)[0]

    theta_xx = grad_theta_x[:, 0:1]

    residual = theta_t - nu * theta_xx

    return residual

# Train one PINN for one initial condition, alpha
def train_pinn(u0_function, alpha):
    theta0 = theta0_x(u0_function, alpha)

    # Define domain bounds for input (x,t)
    lb = np.array([xmin, 0.0]) # Lower bound [x_min, t_min]
    ub = np.array([xmax, T_final]) # Upper bound [x_max, t_max]

    model = ThetaPINN(layers, lb, ub).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create initial condition data theta(x,0) = theta_0(x)
    idx_initial = np.random.choice(len(x), N_u, replace=False)

    x_initial = x[idx_initial]
    t_initial = np.zeros_like(x_initial)
    theta_initial = theta0[idx_initial]

    X_initial = np.column_stack([x_initial, t_initial])

    # Create boundary data for theta at x = xmin and x = xmax
    t_boundary = T_final * np.random.rand(N_u)

    X_boundary = []
    theta_boundary = []

    for t_b in t_boundary:
        theta_full = heat_kernel_solution(theta0, t_b)

        X_boundary.append([xmin, t_b])
        theta_boundary.append(theta_full[0])

        X_boundary.append([xmax, t_b])
        theta_boundary.append(theta_full[-1])

    X_boundary = np.array(X_boundary)
    theta_boundary = np.array(theta_boundary)

    # Combine initial, boundary data
    X_u_train = np.vstack([X_initial, X_boundary])
    theta_u_train = np.concatenate([theta_initial, theta_boundary])

    X_u_train = torch.tensor(X_u_train, dtype=torch.float32, device = device)
    theta_u_train = torch.tensor(theta_u_train[:, None], dtype = torch.float32, device = device)

    # Separate x, t for collocation points
    x_f = xmin + (xmax - xmin) * np.random.rand(N_f)
    t_f = T_final * np.random.rand(N_f)

    X_f_train = np.column_stack([x_f, t_f])
    X_f_train = torch.tensor(X_f_train, dtype=torch.float32, device=device)

    loss_history = []

    for n in range(Nmax + 1):
        optimizer.zero_grad()

        # Enforce initial condition, boundary values
        theta_pred = model(X_u_train)
        loss_data = torch.mean((theta_pred - theta_u_train) ** 2)

        # Enforce theta_t - nu theta_xx = 0
        residual = heat_residual(model, X_f_train)
        loss_physics = torch.mean(residual ** 2)

        loss = loss_data + loss_physics
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if n % 1000 == 0:
            print(
                f"alpha={alpha}, iteration={n}, "
                f"loss={loss.item():.4e}, "
                f"data={loss_data.item():.4e}, "
                f"physics={loss_physics.item():.4e}")

    return model, theta0, loss_history

# Recover u(x,t) from PINN
def u_xt(model, alpha):
    u_values = {}

    model.eval()

    for t in times:
        X_eval = np.column_stack([x, t * np.ones_like(x)])
        X_eval_torch = torch.tensor(X_eval, dtype = torch.float32, device = device)

        with torch.no_grad():
            theta = model(X_eval_torch).cpu().numpy().flatten()

        D_alpha_theta = fractional_derivative(theta, alpha)

        u_values[t] = -2.0 * nu * D_alpha_theta / theta # As in Eq. 3

    return u_values

# Train PINNs for each initial condition, alpha
pinn_results = {}

for initial_condition_name, u0_function in initial_conditions.items():
    pinn_results[initial_condition_name] = {}

    for alpha in alphas:
        print(f"\nPINN for {initial_condition_name}, alpha = {alpha}")

        model, theta0, loss_history = train_pinn(u0_function, alpha)

        pinn_results[initial_condition_name][alpha] = {
            "model": model,
            "theta0_x": theta0,
            "loss_history": loss_history,
            "u_xt": u_xt(model, alpha),}

# Compare results with exact solutions
for initial_condition_name in initial_conditions.keys():
    for alpha in alphas:
        fig, axes = plt.subplots(2, 2, figsize = (10, 7), sharex = True, sharey = True)
        axes = axes.flatten()

        for ax, t in zip(axes, times):
            u_exact = solutions[initial_condition_name][alpha][t] # As defined in Task 5
            u_pinn = pinn_results[initial_condition_name][alpha]["u_xt"][t]

            ax.plot(x, u_exact, label = "Exact")
            ax.plot(x, u_pinn, "--", label = "PINN")

            ax.set_title(f"t = {t}")
            ax.set_xlabel("x")
            ax.set_ylabel(r"$u(x,t)$")
            ax.grid(True)
            ax.legend()

        fig.suptitle(f"{initial_condition_name}, " + rf"$\alpha = {alpha}$")
        plt.tight_layout()
        plt.show()

# Plot training loss
for initial_condition_name in initial_conditions.keys():
    plt.figure(figsize = (7, 4))

    for alpha in alphas:
        loss_history = pinn_results[initial_condition_name][alpha]["loss_history"]

        plt.semilogy(loss_history, label = rf"$\alpha={alpha}$")

    plt.title(f"Training loss: {initial_condition_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
