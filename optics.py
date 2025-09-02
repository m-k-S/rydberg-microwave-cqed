import numpy as np
from tqdm import tqdm
from pint import get_application_registry
from scipy.integrate import solve_ivp
from scipy.constants import hbar, c, epsilon_0

si = get_application_registry()
hbar = hbar * si.joule * si.second
c = c * si.meter / si.second
epsilon_0 = epsilon_0 * si.farad / si.meter

# Rabi frequency, dependendent on intensity and the saturation intensity of the transition
# this is for the 5s1/2 to 5p3/2, closed cycle transition
# gamma may or may not include experimental dephasing
def G_to_5P_Rabi_frequency(I, gamma, Isat):
    """
    Inputs:
      I     : intensity [W/m^2]
      gamma : natural linewidth (angular) [1/s]
      Isat  : saturation intensity [W/m^2]
    Returns:
      Ω_ge  : Rabi frequency [1/s]
    """
    return gamma * np.sqrt(I/(2*Isat))

# this is for the transition from the 5p3/2 state to the Rydberg state
# we take the dipole moment from ARC calculations
def Rydberg_Rabi_frequency(I, dipole_matrix_element):
    """
    Inputs:
      I        : intensity [W/m^2]
      mu       : dipole moment [C·m] 
    Returns:
      Ω_er     : Rabi frequency [1/s]
    """
    E = np.sqrt((2*I/(c*epsilon_0)))
    return dipole_matrix_element*E/hbar

# ---------- 3-level ladder Lindblad RHS ----------
# define the optical Bloch equations for a 3-level system (g, e, r)
# this is used for numerical integration
# gamma_phi are dephasing rates, estimated from e.g.:
#  - laser intrinsic linewidth
#  - Doppler broadening
#  - collisional broadening
#  these are usually dominated by laser linewidth (10-100 kHz)
def ladder_rhs(t, rho_vec,
               Omega_ge, Omega_er,
               Delta_ge, Delta_er,
               Gamma_e, Gamma_r,
               gamma_ge_phi, gamma_er_phi):
    """
    t : time [s]
    rho_vec : state vector, flattened complex (9,) 
    
    Omega_ge : Rabi frequency, 5s1/2 to 5p/3 [1/s]
    Omega_er : Rabi frequency, 5p/3 to Rydberg [1/s]
    
    Delta_ge : detuning, 5s1/2 to 5p/3 [1/s]
    Delta_er : detuning, 5p/3 to Rydberg [1/s]
    
    Gamma_e : decay rate, 5p/3 to 5s1/2 [1/s]
    Gamma_r : decay rate, Rydberg to 5p/3 [1/s]
    
    gamma_ge_phi : dephasing rate, 5s1/2 to 5p/3 [1/s]
    gamma_er_phi : dephasing rate, 5p/3 to Rydberg [1/s]
    """
    rho = rho_vec.reshape((3, 3))

    # Hamiltonian
    H = np.array([
        [0.0,                                (0.5*hbar*Omega_ge).magnitude,             0.0],
        [(0.5*hbar*Omega_ge).magnitude,      (-hbar*Delta_ge).magnitude,                (0.5*hbar*Omega_er).magnitude],
        [0.0,                                (0.5*hbar*Omega_er).magnitude,             (-hbar*(Delta_ge+Delta_er)).magnitude]
    ])
    H = H.astype(np.complex128)

    comm = (-1j/hbar.magnitude) * (H @ rho - rho @ H) # commutator

    L = np.zeros((3,3), dtype=np.complex128)
    C_eg = np.zeros((3,3)); C_eg[0,1] = 1.0   # |g><e|
    C_re = np.zeros((3,3)); C_re[1,2] = 1.0   # |e><r|

    def lindblad(C, gamma):
        return gamma.magnitude*(C @ rho @ C.conj().T - 0.5*(C.conj().T @ C @ rho + rho @ C.conj().T @ C))

    L += lindblad(C_eg, Gamma_e)  # e->g
    L += lindblad(C_re, Gamma_r)  # r->e

    D = np.zeros((3,3), dtype=np.complex128)
    D[0,1] += -gamma_ge_phi.magnitude * rho[0,1]; D[1,0] += -gamma_ge_phi.magnitude * rho[1,0]
    D[1,2] += -gamma_er_phi.magnitude * rho[1,2]; D[2,1] += -gamma_er_phi.magnitude * rho[2,1]
    D[0,2] += -(gamma_ge_phi.magnitude+gamma_er_phi.magnitude) * rho[0,2]; D[2,0] += -(gamma_ge_phi.magnitude+gamma_er_phi.magnitude) * rho[2,0]

    return (comm + L + D).reshape(-1).view(np.float64)

# steady state Rydberg population
# solve the Lindblad equation at equilibrium
def steady_state_Rydberg_pop(
    Omega_ge, 
    Omega_er, 
    Delta_ge, 
    Delta_er, 
    Gamma_e, 
    Gamma_r, 
    gamma_ge_phi, 
    gamma_er_phi,
    t_final=30.0 * si.microsecond):
    """
    Inputs (Quantities):
      Omega_ge, Omega_er, Delta_ge, Delta_er, Gamma_e, Gamma_r, gamma_ge_phi, gamma_er_phi : [1/s] (angular)
      t_final : integration time [s]
    Returns:
      P_r (float): steady-state population in |r>, as a fraction
    """
    rho0 = np.zeros((3,3), dtype=np.complex128); rho0[0,0] = 1.0
    y0 = rho0.reshape(-1).view(np.float64)

    rhs = lambda t,y: ladder_rhs(t, y.view(np.complex128),
                                          Omega_ge, Omega_er, Delta_ge, Delta_er, Gamma_e, Gamma_r, gamma_ge_phi, gamma_er_phi)
    sol = solve_ivp(rhs, [0.0, t_final.magnitude], y0, method="RK45", rtol=2e-6, atol=1e-9)
    rho_ss = sol.y[:, -1].view(np.complex128).reshape((3,3))
    return float(np.real(rho_ss[2,2])) # fractional population

# sampling positions of atoms within a rectangular cavity
# we take the origin at the center of the cavity
def sample_positions_uniform(x,y,z, N):
    """
    Uniformly sample N positions inside a rectangular box.
    x,y,z are cavity dimensions
    Returns (N,3) numpy array of floats in meters.
    """
    xs = np.random.uniform(-x.to(si.meter).magnitude/2, x.to(si.meter).magnitude/2, size=N)
    ys = np.random.uniform(-y.to(si.meter).magnitude/2, y.to(si.meter).magnitude/2, size=N)
    zs = np.random.uniform(-z.to(si.meter).magnitude/2, z.to(si.meter).magnitude/2, size=N)
    return np.column_stack([xs,ys,zs]) * si.meter

# define Gaussian intensity profile, at a fixed power and beam waist
def gaussian_intensity_profile(P, w):
    """Peak intensity on axis for TEM00: I0 = 2P/(pi w^2). P [W], w [m]."""
    I0 = 2*P/(np.pi*w**2)   
    return lambda x,y,z: I0 * np.exp(-2*((x * si.meter)**2 + (y * si.meter)**2)/w**2)

def estimate_pumped_atoms_general(positions_m,  # (N,3) floats in meters
                                  laser_power_780,
                                  beam_waist_780,
                                  laser_power_481,
                                  beam_waist_481,
                                  Isat_780, 
                                  dipole_matrix_element_er, # from ARC
                                  Delta_ge, Delta_er,     # Quantities [1/s] detunings; experimental
                                  Gamma_e, Gamma_r,       # Quantities [1/s] decays
                                  gamma_ge_phi, gamma_er_phi,   # Quantities [1/s] pure dephasings, experimental
                                  grid_points=9,
                                  t_final=30.0 * si.microsecond):
    """
    Returns:
      pump_frac  : (N,) steady-state P_r per atom <--- per atom!!
      N_pumped   : scalar = mean(P_r) * N_total (i.e. N_eff)
    """
    
    pos = np.asarray(positions_m.magnitude, float)
    N_total = len(pos)
    x,y,z = pos[:,0], pos[:,1], pos[:,2]

    # 1) local intensities (Quantities) at each position
    Ige = gaussian_intensity_profile(laser_power_780, beam_waist_780)(x,y,z)   # [W/m^2]
    Ier = gaussian_intensity_profile(laser_power_481, beam_waist_481)(x,y,z)   # [W/m^2]

    # 2) Rabi frequencies from dipoles (Quantities [1/s])
    Omega_ge = G_to_5P_Rabi_frequency(Ige, gamma_ge_phi, Isat_780)
    Omega_er = Rydberg_Rabi_frequency(Ier, dipole_matrix_element_er)

    # 3) build coarse grids in observed Ω ranges
    ge_min, ge_max = Omega_ge.min(), Omega_ge.max()
    er_min, er_max = Omega_er.min(), Omega_er.max()
    ge_grid = np.linspace(ge_min.to(1/si.second).magnitude,
                          ge_max.to(1/si.second).magnitude,
                          max(2, grid_points)) * (1/si.second)
    er_grid = np.linspace(er_min.to(1/si.second).magnitude,
                          er_max.to(1/si.second).magnitude,
                          max(2, grid_points)) * (1/si.second)

    # 4) solve steady state on grid
    print("Computing steady state Rydberg populations...")
    Pr_grid = np.zeros((len(ge_grid), len(er_grid)))
    for i, oge in tqdm(enumerate(ge_grid), total=len(ge_grid), desc="Ω_ge"):
        for j, oer in tqdm(enumerate(er_grid), total=len(er_grid), desc="Ω_er", leave=False):
            Pr_grid[i,j] = steady_state_Rydberg_pop(oge, oer, Delta_ge, Delta_er, Gamma_e, Gamma_r, gamma_ge_phi, gamma_er_phi, t_final=t_final)

    # 5) nearest-neighbor assign per atom
    Omega_ge_f = Omega_ge.to(1/si.second).magnitude
    Omega_er_f = Omega_er.to(1/si.second).magnitude
    ge_g  = ge_grid.to(1/si.second).magnitude
    er_g  = er_grid.to(1/si.second).magnitude

    ge_idx = np.abs(Omega_ge_f[:,None] - ge_g[None,:]).argmin(axis=1)
    er_idx = np.abs(Omega_er_f[:,None] - er_g[None,:]).argmin(axis=1)
    pump_frac = Pr_grid[ge_idx, er_idx]

    N_pumped = float(pump_frac.mean() * N_total)
    return pump_frac, N_pumped