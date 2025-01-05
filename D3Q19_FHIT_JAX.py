import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# Domain size and DHIT parameters
nx, ny, nz = 64, 64, 64
u0 = 0.1  # Initial velocity amplitude
nu = 1.0000070E-03  # Kinematic viscosity tau = 0.503
OMEGA = 1.0 / (3.0 * nu + 0.5)

PLOT_EVERY_N_STEPS = 100000
SAVE_EVERY_N_STEPS = 200
N_ITERATIONS = 6000

x = jnp.arange(nx)
y = jnp.arange(ny)   # => index based such as 0, 1, ..., nx-1
z = jnp.arange(nz)
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

# x = jnp.linspace(0, 1, nx)  # evenly spaced points between 0 and 1
# y = jnp.linspace(0, 1, ny)  # evenly spaced points between 0 and 1
# z = jnp.linspace(0, 1, nz)  # evenly spaced points between 0 and 1
# X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

# Define the number of discrete velocities for D3Q19
N_DISCRETE_VELOCITIES = 19

# Lattice velocities for D3Q19 model
LATTICE_VELOCITIES_X = jnp.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1])
LATTICE_VELOCITIES_Y = jnp.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0])
LATTICE_VELOCITIES_Z = jnp.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1])
LATTICE_WEIGHTS = jnp.array([
    1/3,     # Rest particle
    
    # Face-connected neighbors
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    
    # Edge-connected neighbors
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36
])

LATTICE_VELOCITIES = jnp.array([LATTICE_VELOCITIES_X, LATTICE_VELOCITIES_Y, LATTICE_VELOCITIES_Z])

# Forcing function
def apply_forcing(t):
    """
    Generate periodic forcing for homogeneous isotropic turbulence.

    Parameters:
    - t: int, current timestep (used for seeding randomness).

    Returns:
    - local_Fx, local_Fy, local_Fz: Force components applied to the density field.
    """
    kmax = 3
    pp = 2.0 * jnp.pi
    amp = 0.0001

    # Random phases using timestep as the seed
    key = jax.random.PRNGKey(t)
    vrandom = pp * jax.random.uniform(key, (2 * kmax + 1, 2 * kmax + 1, 2 * kmax + 1))

    # Prepare spatial grid
    i, j, k = jnp.arange(1, nx + 1), jnp.arange(1, ny + 1), jnp.arange(1, nz + 1)
    ddx, ddy, ddz = jnp.meshgrid(i, j, k, indexing="ij")
    ddx, ddy, ddz = pp * ddx / nx, pp * ddy / ny, pp * ddz / nz

    # Force arrays
    local_Fx, local_Fy, local_Fz = jnp.zeros((nx, ny, nz)), jnp.zeros((nx, ny, nz)), jnp.zeros((nx, ny, nz))

    # Wave numbers for forcing calculation
    k1, k2, k3 = jnp.arange(-kmax, kmax + 1), jnp.arange(-kmax, kmax + 1), jnp.arange(-kmax, kmax + 1)
    k1_grid, k2_grid, k3_grid = jnp.meshgrid(k1, k2, k3, indexing="ij")
    index1 = k1_grid**2 + k2_grid**2 + k3_grid**2

    # Avoid division by zero
    index1_safe = jnp.where(index1 > 0, index1, 1e-6)

    # Mask to apply cut-off for forcing
    mask = (index1 > 0) & (jnp.sqrt(index1) <= jnp.sqrt(kmax**2))

    # Calculate ab with sinusoidal forcing
    ab = jnp.sin(k1_grid * ddx[..., None, None, None] + 
                 k2_grid * ddy[..., None, None, None] + 
                 k3_grid * ddz[..., None, None, None] + vrandom) * mask

    # Calculate force contributions, applying cut-off and scaling
    force_contribution = ab / index1_safe
    local_Fx += amp * jnp.sum((k2_grid * k3_grid * force_contribution), axis=(3, 4, 5))
    local_Fy += amp * jnp.sum((-2.0 * k1_grid * k3_grid * force_contribution), axis=(3, 4, 5))
    local_Fz += amp * jnp.sum((k1_grid * k2_grid * force_contribution), axis=(3, 4, 5))

    return local_Fx, local_Fy, local_Fz

# JIT-compiled version
apply_forcing_jit = jax.jit(apply_forcing)



# Define utility functions
def get_density(discrete_velocities):
    return jnp.sum(discrete_velocities, axis=-1)

def get_macroscopic_velocities(discrete_velocities, density):
    return jnp.einsum("dQ,ijkQ->ijkd", LATTICE_VELOCITIES, discrete_velocities) / density[..., jnp.newaxis]

def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum("dQ,ijkd->ijkQ", LATTICE_VELOCITIES, macroscopic_velocities)
    macroscopic_velocity_magnitude = jnp.linalg.norm(macroscopic_velocities, axis=-1)
    equilibrium_discrete_velocities = (
        density[..., jnp.newaxis] * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] *
        (1 + 3 * projected_discrete_velocities +
         9/2 * projected_discrete_velocities**2 -
         3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2)
    )
    return equilibrium_discrete_velocities

# Collision and streaming update
@jax.jit
def update(discrete_velocities_prev, Fx, Fy, Fz):
    density_prev = get_density(discrete_velocities_prev)
    macroscopic_velocities_prev = get_macroscopic_velocities(discrete_velocities_prev, density_prev)
    equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(macroscopic_velocities_prev, density_prev)
    
    tau1 = 1.0 / OMEGA
    tau2 = 1.0 - tau1

    discrete_velocities_post_collision = discrete_velocities_prev.copy()
    
    for alpha in range(N_DISCRETE_VELOCITIES):
        force_term = 3.0 * LATTICE_WEIGHTS[alpha] * density_prev * (
            LATTICE_VELOCITIES_X[alpha] * Fx +
            LATTICE_VELOCITIES_Y[alpha] * Fy +
            LATTICE_VELOCITIES_Z[alpha] * Fz
        )
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[..., alpha].set(
            tau2 * discrete_velocities_prev[..., alpha] +
            tau1 * equilibrium_discrete_velocities[..., alpha] +
            force_term
        )

    # Streaming step
    discrete_velocities_streamed = jnp.zeros_like(discrete_velocities_post_collision)
    for alpha in range(N_DISCRETE_VELOCITIES):
        discrete_velocities_streamed = discrete_velocities_streamed.at[..., alpha].set(
            jnp.roll(
                jnp.roll(
                    jnp.roll(discrete_velocities_post_collision[..., alpha],
                             shift=LATTICE_VELOCITIES_X[alpha], axis=0),
                    shift=LATTICE_VELOCITIES_Y[alpha], axis=1),
                shift=LATTICE_VELOCITIES_Z[alpha], axis=2)
        )
    
    return discrete_velocities_streamed

# Run simulation
def run(discrete_velocities_prev):
    kinetic_energy_over_time = []

    for i in tqdm(range(N_ITERATIONS)):
        Fx, Fy, Fz = apply_forcing_jit(i)
        discrete_velocities_next = update(discrete_velocities_prev, Fx, Fy, Fz)
        discrete_velocities_prev = discrete_velocities_next
        
        density = get_density(discrete_velocities_next)
        macroscopic_velocities = get_macroscopic_velocities(discrete_velocities_next, density)

        # Save data every N steps as separate files without leading zeros in filenames
        if i % SAVE_EVERY_N_STEPS == 0:
            filename = f"velocity_data_{i}.dat"  # No leading zeros
            print(f"Saving data to {filename} at iteration {i}...")
            
            with open(filename, "w") as f:
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            f.write(f"{macroscopic_velocities[ix, iy, iz, 0]:21.10e}  "
                                    f"{macroscopic_velocities[ix, iy, iz, 1]:21.10e}  "
                                    f"{macroscopic_velocities[ix, iy, iz, 2]:21.10e}\n")
            print(f"Data saved to {filename}.")

        kinetic_energy = 0.5 * jnp.sum(jnp.square(macroscopic_velocities))
        kinetic_energy_over_time.append(kinetic_energy)

        if i % PLOT_EVERY_N_STEPS == 0:
            plt.figure(figsize=(4, 4))
            plt.contourf(X[:, :, nx//2], Y[:, :, nx//2],
                         jnp.linalg.norm(macroscopic_velocities[:, :, nx//2], axis=-1), cmap='inferno')
            plt.colorbar(label="Velocity Magnitude")
            plt.title(f"Iteration {i}")
            plt.axis("scaled")
            plt.show()

    plt.plot(kinetic_energy_over_time)
    plt.xlabel("Iterations")
    plt.ylabel("Kinetic Energy")
    plt.show()

# Initialization
ux = jnp.zeros((nx, ny, nz))
uy = jnp.zeros((nx, ny, nz))
uz = jnp.zeros((nx, ny, nz))
rho = jnp.ones((nx, ny, nz))

VELOCITY_PROFILE = jnp.stack((ux, uy, uz), axis=-1)
discrete_velocities_prev = get_equilibrium_discrete_velocities(VELOCITY_PROFILE, rho)

run(discrete_velocities_prev)
