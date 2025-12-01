# spectral solver for kpz equation in 1 or 2 dimensions
# Euler-Maruyama time stepping
import numpy as np

class KPZ_Spec:
    def __init__(self, L, N, nu, lam, D, dt=0.001, nt =100,dim=2):
        self.L = L      # domain size
        self.N = N      # number of grid points
        self.nu = nu    # diffusion coefficient
        self.lam = lam  # nonlinear coupling
        self.D = D      # noise strength
        self.dt = dt    # time step
        self.dim = dim  # spatial dimension (1 or 2)
        self.nt = nt    # number of time steps

        self.dx = L / N # spatial discretization
        self.x = np.linspace(0, L, N, endpoint=False)   # spatial grid

        if dim == 1:
            self.k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
            self.k_squared = self.k**2
        elif dim == 2:
            kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
            ky = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
            self.kx, self.ky = np.meshgrid(kx, ky, indexing='ij')
            self.k_squared = self.kx**2 + self.ky**2
        else:
            raise ValueError("Dimension must be 1 or 2.")
        
        self.h_hat = np.zeros((N,) * dim, dtype=complex)  # Fourier coefficients of height field
        # heigh field h(x,t) time snapshots
        self.h_time = np.zeros((nt,) + (N,) * dim)

    def _laplacian(self, h_hat):
        """Compute the Laplacian in Fourier space."""
        return -self.nu*self.k_squared * h_hat
    
    def _nonlinear_term(self, h_hat):
        """Compute the nonlinear term in Fourier space."""
        if self.dim == 1:
            h_x = np.fft.ifft(1j * self.k * h_hat).real
            nonlinear_real =  h_x**2
            return 0.5 * self.lam * np.fft.fft(nonlinear_real)
        elif self.dim == 2:
            h_x = np.fft.ifft(1j * self.kx * h_hat).real
            h_y = np.fft.ifft(1j * self.ky * h_hat).real
            nonlinear_real = (h_x**2 + h_y**2)
            return 0.5 * self.lam * np.fft.fft2(nonlinear_real)
        
    def step(self):
        """Perform a single time step using Euler-Maruyama method."""
        noise_amplitude = np.sqrt(2 * self.D / self.dt)
        if self.dim == 1:
            eta = np.random.normal(size=self.N)
            eta_hat = noise_amplitude * (np.fft.fft(eta))
        elif self.dim == 2:
            eta = np.random.normal(size=(self.N, self.N))
            eta_hat = noise_amplitude * (np.fft.fft2(eta))

        laplacian_term = self._laplacian(self.h_hat)
        nonlinear_term = self._nonlinear_term(self.h_hat)
        self.h_hat += self.dt * (laplacian_term + nonlinear_term) + np.sqrt(self.dt) * eta_hat

    def run(self, h0_k):
        """Run the simulation for nt time steps."""
        self.h_hat = h0_k
        for t in range(self.nt):
            self.step()
            if self.dim == 1:
                self.h_time[t, :] = np.fft.ifft(self.h_hat).real
            elif self.dim == 2:
                self.h_time[t, :, :] = np.fft.ifft2(self.h_hat).real
        return self.h_time
        