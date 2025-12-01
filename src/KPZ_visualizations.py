import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from KPZ_solver import KPZSolver

### Arbitrary projection functions
def sample_line(h, coords, x0, v, num_points=200, t_range=0.5):
    """
    Sample a d-dimensional field h along an arbitrary line,
    keeping only points within the domain.
    """

    d = h.ndim
    x0 = np.array(x0)
    v = np.array(v)

    # Extract 1D axes safely
    grid_axes = [np.unique(coords[i].flatten()) for i in range(d)]
    x_min = np.array([ax.min() for ax in grid_axes])
    x_max = np.array([ax.max() for ax in grid_axes])

    # Interpolator
    interpolator = RegularGridInterpolator(grid_axes, h, bounds_error=False, fill_value=np.nan)

    # Full t_vals
    t_vals = np.linspace(-t_range, t_range, num_points)
    X = np.array([x0 + t*v for t in t_vals])

    # Clip points outside the domain
    mask = np.all((X >= x_min) & (X <= x_max), axis=1)
    t_vals = t_vals[mask]
    X = X[mask]
    h_vals = interpolator(X)

    return t_vals, h_vals, X


def sample_plane(h, coords, x0, v1, v2, n1=100, n2=100, t1_range=0.5, t2_range=0.5):
    """
    Sample a d-dimensional field h on an arbitrary 2D plane,
    keeping a full rectangular grid and clipping t1/t2 to domain bounds.
    """

    d = h.ndim
    x0 = np.array(x0)
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Extract 1D axes
    grid_axes = [np.unique(coords[i].flatten()) for i in range(d)]
    x_min = np.array([ax.min() for ax in grid_axes])
    x_max = np.array([ax.max() for ax in grid_axes])

    def clip_t_range(v, t_range):
        # Compute t_min, t_max such that x0 + t*v stays inside domain
        with np.errstate(divide='ignore', invalid='ignore'):
            t_min_axes = (x_min - x0) / v
            t_max_axes = (x_max - x0) / v
        t_min_axes[v==0] = -np.inf
        t_max_axes[v==0] = np.inf
        t_min = max(-t_range, np.max(np.minimum(t_min_axes, t_max_axes)))
        t_max = min(t_range, np.min(np.maximum(t_min_axes, t_max_axes)))
        return t_min, t_max

    t1_min, t1_max = clip_t_range(v1, t1_range)
    t2_min, t2_max = clip_t_range(v2, t2_range)

    # Create clipped regular grid
    t1_vals = np.linspace(t1_min, t1_max, n1)
    t2_vals = np.linspace(t2_min, t2_max, n2)
    T1, T2 = np.meshgrid(t1_vals, t2_vals, indexing='ij')

    # Build interpolator
    interpolator = RegularGridInterpolator(grid_axes, h, bounds_error=False, fill_value=np.nan)

    # Sample field
    X_plane = x0 + T1[..., None]*v1 + T2[..., None]*v2  # shape (n1, n2, d)
    h_plane = interpolator(X_plane)

    return t1_vals, t2_vals, h_plane, X_plane


###PLOT OPTION 1. 1D Slice
def plot_1d_line(t_vals, h_vals, title="1D Slice"):
    plt.figure(figsize=(8,4))
    plt.plot(t_vals, h_vals)
    plt.xlabel("t (parametric)")
    plt.ylabel("h")
    plt.title(title)
    plt.grid(True)
    plt.show()

#PLOT OPTION 2. 3d surface

def plot_3d_surface(h2d, t1_vals, t2_vals, title="3D Surface"):
    """
    Plot a 3D surface of h2d using t1 and t2 as axes.
    """
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    T1, T2 = np.meshgrid(t1_vals, t2_vals, indexing='ij')
    ax.plot_surface(T1, T2, h2d, cmap='viridis')
    ax.set_xlabel("t1 along v1")
    ax.set_ylabel("t2 along v2")
    ax.set_zlabel("h")
    ax.set_title(title)
    plt.show()

#PLOT OPTION 3. 2d heatmap
def plot_2d_heatmap(h2d, t1_vals, t2_vals, title="2D Plane Slice"):
    """
    Plot a 2D heatmap of h2d using t1 and t2 as axes.
    """
    plt.figure(figsize=(6,5))
    extent = (t1_vals.min(), t1_vals.max(), t2_vals.min(), t2_vals.max())
    plt.imshow(h2d.T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
    plt.colorbar(label="h")
    plt.xlabel("t1 along v1")
    plt.ylabel("t2 along v2")
    plt.title(title)
    plt.show()

#PLOT OPTION 3. 2d heatmap (gradient)
def plot_gradient_magnitude(h2d, t1_vals, t2_vals, title="|∇h| Heatmap"):
    """
    Plot the magnitude of the gradient of h2d on a 2D heatmap with t1 and t2 axes.
    """
    gx, gy = np.gradient(h2d)
    mag = np.sqrt(gx**2 + gy**2)

    plt.figure(figsize=(6,5))
    extent = (t1_vals.min(), t1_vals.max(), t2_vals.min(), t2_vals.max())
    plt.imshow(mag.T, origin='lower', aspect='auto', cmap='magma', extent=extent)
    plt.colorbar(label="|∇h|")
    plt.xlabel("t1 along v1")
    plt.ylabel("t2 along v2")
    plt.title(title)
    plt.show()



def animate_1d(t_vals_list, h_vals_list, title="1D Line Animation", interval=200):
    fig, ax = plt.subplots(figsize=(8,4))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(t_vals_list[0].min(), t_vals_list[0].max())
    y_min = min([h.min() for h in h_vals_list])
    y_max = max([h.max() for h in h_vals_list])
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("t (parametric)")
    ax.set_ylabel("h")
    ax.set_title(title)
    ax.grid(True)

    def update(frame):
        line.set_data(t_vals_list[frame], h_vals_list[frame])
        return line,

    anim = FuncAnimation(fig, update, frames=len(t_vals_list), interval=interval, blit=True)
    plt.show()
    return anim


def animate_2d_heatmap(h_planes_list, t1_vals, t2_vals, title="2D Heatmap Animation", interval=200):
    fig, ax = plt.subplots(figsize=(6,5))
    extent = (t1_vals.min(), t1_vals.max(), t2_vals.min(), t2_vals.max())
    im = ax.imshow(h_planes_list[0].T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("h")
    ax.set_xlabel("t1 along v1")
    ax.set_ylabel("t2 along v2")
    ax.set_title(title)

    # FORCE full axis limits
    ax.set_xlim(t1_vals.min(), t1_vals.max())
    ax.set_ylim(t2_vals.min(), t2_vals.max())

    def update(frame):
        im.set_data(h_planes_list[frame].T)
        # ensure extent is preserved each frame
        im.set_extent(extent)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(h_planes_list), interval=interval, blit=True)
    plt.show()
    return anim

def animate_3d_surface(h_planes_list, t1_vals, t2_vals, title="3D Surface Animation", interval=200):
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    T1, T2 = np.meshgrid(t1_vals, t2_vals, indexing='ij')

    z_min = min([h.min() for h in h_planes_list])
    z_max = max([h.max() for h in h_planes_list])
    ax.set_zlim(z_min, z_max)
    ax.set_xlim(t1_vals.min(), t1_vals.max())
    ax.set_ylim(t2_vals.min(), t2_vals.max())
    ax.set_xlabel("t1 along v1")
    ax.set_ylabel("t2 along v2")
    ax.set_zlabel("h")
    ax.set_title(title)

    surf = [ax.plot_surface(T1, T2, h_planes_list[0], cmap='viridis')]

    def update(frame):
        surf[0].remove()  # remove previous surface
        surf[0] = ax.plot_surface(T1, T2, h_planes_list[frame], cmap='viridis')
        return surf

    anim = FuncAnimation(fig, update, frames=len(h_planes_list), interval=interval, blit=False)
    return anim



