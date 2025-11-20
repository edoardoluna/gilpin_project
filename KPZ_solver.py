class KPZSolver:
    """
    Generic solver for the d-dimensional KPZ equation:

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
            Noise amplitude.
        initial_condition : callable or ndarray
            Function h(x_vec) or array of size (N₁, ..., N_d).
        boundary : str
            Boundary type ('periodic', 'dirichlet', etc.).
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
        self.rng = rng

        # Normalize L and N to dimension-d tuples
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

    # ----------------------------
    # Initialization
    # ----------------------------

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

    # ----------------------------
    # Spatial operators
    # ----------------------------

    def _laplacian(self, f):
        """
        Compute the d-dimensional Laplacian of f.
        Must be implemented by subclass or user.
        """
        raise NotImplementedError("Implement a discretization for ∇²h.")

    def _gradient(self, f):
        """
        Compute the d-dimensional gradient of f.
        
        Returns
        -------
        list of arrays
            [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂x_d]
        """
        raise NotImplementedError("Implement a discretization for ∇h.")

    def _noise_term(self):
        """
        Generate η(x,t) for d-dimensional KPZ.
        Should scale like sqrt(dt * Π_i dx_i^(-1)) for white noise.
        """
        raise NotImplementedError("Implement noise term η(x,t).")

    # ----------------------------
    # Time stepping
    # ----------------------------

    def step(self):
        """
        Advance by one time step using a choice of numerical scheme
        (Euler–Maruyama, semi-implicit, spectral, etc.).
        """
        raise NotImplementedError("Implement update rule for KPZ step.")

    # ----------------------------
    # Simulation control
    # ----------------------------

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

    # ----------------------------
    # Utility
    # ----------------------------

    def set_height(self, h_new):
        self.h = h_new.copy()

    def get_height(self):
        return self.h.copy()