class KPZSolver:
    """
    Generic solver for the d-dimensional KPZ equation (For Computational Physics project):

        ∂h/∂t = ν ∇²h + (λ/2) |∇h|² + η(x, t)

    where x ∈ ℝᵈ and η is space–time white noise.
    """

    def __init__(
        self,
        L,
        N,
        dt,
        d,
        nu,
        lam,
        noise_strength,
        initial_condition=None,
        boundary="periodic",
        rng=None
    ):
        """
        Parameters
        ----------
        L : float or sequence
            Domain length in each dimension; if float, replicated to all dims.
        N : int or sequence
            Number of grid points in each dimension; if int, replicated to all dims.
        dt : float
            Time step.
        d : int
            Spatial dimension.
        nu : float
            Diffusion coefficient.
        lam : float
            Nonlinear coupling λ.
        noise_strength : float
            Noise amplitude (see _noise_term for scaling).
        initial_condition : callable or ndarray
            Function h(x_vec) or array of size (N₁, ..., N_d).
        boundary : str
            Boundary type ('periodic' supported; others raise NotImplementedError).
        rng : numpy.random.Generator
            Optional random number generator.
        """
        import numpy as np

        self.d = d
        self.dt = dt
        self.nu = nu
        self.lam = lam
        self.noise_strength = noise_strength
        self.boundary = boundary

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        # Normalize L and N to dimension-d tuples, makes life easier.
        if isinstance(L, (int, float)):
            self.L = (L,) * d
        else:
            self.L = tuple(L)

        if isinstance(N, int):
            self.N = (N,) * d
        else:
            self.N = tuple(N)

        # Grid spacing per dimension
        self.dx = tuple(L_i / N_i for L_i, N_i in zip(self.L, self.N))

        # Build grids (x1, x2, ..., xd), each an ndarray with shape (N1,...,Nd)
        self.coords = self._setup_grid()

        # Height field
        self.h = self._initialize_h(initial_condition)

# Initialization

    def _setup_grid(self):
        """Create d-dimensional meshgrid."""
        import numpy as np
        axes = [np.linspace(0, L_i, N_i, endpoint=False)
                for L_i, N_i in zip(self.L, self.N)]
        return np.meshgrid(*axes, indexing="ij")

    def _initialize_h(self, init):
        """Initialize the height field in d dimensions."""
        import numpy as np

        shape = self.N
        if init is None:
            return np.zeros(shape)
        elif callable(init):
            # Evaluate h(x) = init(x1, ..., xd)
            return init(*self.coords)
        else:
            return np.array(init).reshape(shape)


# Spatial operators

    def _laplacian(self, f):
        """
        Compute the d-dimensional Laplacian of f using central differences
        with periodic boundary conditions.
        """
        import numpy as np

        if self.boundary != "periodic":
            raise NotImplementedError("Only periodic boundary conditions are implemented.")

        lap = np.zeros_like(f)
        # Sum second derivatives along each axis
        for axis, dx in enumerate(self.dx):
            f_plus = np.roll(f, -1, axis=axis)
            f_minus = np.roll(f, 1, axis=axis)
            lap += (f_plus - 2.0 * f + f_minus) / (dx * dx)
        return lap

    def _gradient(self, f):
        """
        Compute the d-dimensional gradient of f using central differences
        with periodic boundary conditions.

        Returns
        -------
        list of arrays
            [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂x_d]
        """
        import numpy as np

        if self.boundary != "periodic":
            raise NotImplementedError("Only periodic boundary conditions are implemented.")

        grads = []
        for axis, dx in enumerate(self.dx):
            f_plus = np.roll(f, -1, axis=axis)
            f_minus = np.roll(f, 1, axis=axis)
            grads.append((f_plus - f_minus) / (2.0 * dx))
        return grads

    def _noise_term(self):
        """
        Generate η(x,t) for d-dimensional KPZ with spatially and temporally
        white noise discretization.

        For white noise with covariance ~ δ(x-x') δ(t-t'), a natural
        discretization scales like:

            η_discrete ~ N(0, noise_strength^2 * dt / V_cell)

        where V_cell = ∏_i dx_i is the volume of a grid cell.
        """
        import numpy as np

        cell_volume = 1.0
        for dx in self.dx:
            cell_volume *= dx

        # Standard normal field
        xi = self.rng.standard_normal(self.N)

        # Scale to get correct variance
        prefactor = self.noise_strength * np.sqrt(self.dt / cell_volume)
        return prefactor * xi


# Time stepping

    def step(self):
        """
        Advance by one time step using explicit Euler–Maruyama:

            h_{n+1} = h_n + dt [ ν ∇²h + (λ/2) |∇h|² ] + η(x,t)

        where η(x,t) is drawn independently at each step.
        """
        import numpy as np

        lap_h = self._laplacian(self.h)
        grads = self._gradient(self.h)

        # |∇h|² = sum_i (∂h/∂x_i)^2
        grad_sq = np.zeros_like(self.h)
        for g in grads:
            grad_sq += g * g

        drift = self.nu * lap_h + 0.5 * self.lam * grad_sq
        noise = self._noise_term()

        self.h = self.h + self.dt * drift + noise

# Run

    def run(self, T, store_interval=None):
        """
        Run simulation until time T.
        
        Parameters
        ----------
        T : float
            Final time.
        store_interval : int or None
            If set, store snapshots every store_interval steps.
        """
        snapshots = []
        num_steps = int(T / self.dt)

        for n in range(num_steps):
            self.step()
            if store_interval and (n % store_interval == 0):
                snapshots.append(self.h.copy())

        return snapshots

# Utility

    def set_height(self, h_new):
        self.h = h_new.copy()

    def get_height(self):
        return self.h.copy()


