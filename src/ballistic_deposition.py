# ballistic deposition KPZ growth model with periodic boundary conditions
# in 1D and 2D
import numpy as np

class BallisticDeposition:
    def __init__(self, N, nt=100, dim=2, events_per_site=0.01, seed=None):
        """
        N:      number of lattice sites per dimension
        nt:     number of time steps (snapshots)
        dim:    spatial dimension (1 or 2)
        events_per_site: expected deposition events per site per time step
        """
        self.N = N
        self.dim = dim
        self.nt = nt
        self.events_per_site = events_per_site

        self.rng = np.random.default_rng(seed)

        if dim == 1:
            self.h = np.zeros(N, dtype=np.int64)
            shape_t = (nt, N)
        elif dim == 2:
            self.h = np.zeros((N, N), dtype=np.int64)
            shape_t = (nt, N, N)
        else:
            raise ValueError("Dimension must be 1 or 2.")

        # time snapshots h(x,t)
        self.h_time = np.zeros(shape_t, dtype=np.int64)

    def step(self):
        """Perform one Monte Carlo time step of ballistic deposition."""
        if self.dim == 1:
            n_events = max(1, int(self.N * self.events_per_site))
            for _ in range(n_events):
                i = self.rng.integers(0, self.N)
                left  = self.h[(i - 1) % self.N]
                right = self.h[(i + 1) % self.N]
                self.h[i] = 1 + max(self.h[i], left, right)

        elif self.dim == 2:
            n_events = max(1, int(self.N * self.N * self.events_per_site))
            for _ in range(n_events):
                i = self.rng.integers(0, self.N)
                j = self.rng.integers(0, self.N)

                center = self.h[i, j]
                up     = self.h[(i - 1) % self.N, j]
                down   = self.h[(i + 1) % self.N, j]
                left   = self.h[i, (j - 1) % self.N]
                right  = self.h[i, (j + 1) % self.N]

                self.h[i, j] = 1 + max(center, up, down, left, right)

    def run(self):
        """Run nt steps and store the height field at each step."""
        for t in range(self.nt):
            self.step()
            self.h_time[t] = self.h
