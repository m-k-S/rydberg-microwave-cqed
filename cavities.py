import numpy as np
from scipy.integrate import nquad
from scipy.constants import c, epsilon_0, mu_0, hbar, Boltzmann, e, physical_constants

from arc import Rubidium85
from pint import get_application_registry

si = get_application_registry()

c = c * si.meter / si.second
mu_0 = mu_0 * si.newton / si.ampere**2
epsilon_0 = epsilon_0 * si.farad / si.meter
hbar = hbar * si.joule * si.second
bohr_radius = physical_constants['Bohr radius'][0] * si.meter 
e = e * si.coulomb
atom = Rubidium85()

def find_suitable_transitions(frequency_bounds, max_n=100):
    suitable_transitions = []
    # nS -> nP transitions
    for i in range(6, max_n):
        for j in range(i+1, max_n):
            l1 = 2
            l2 = 3
            j1 = 2.5
            j2 = 3.5

            frequency = np.abs(atom.getTransitionFrequency(j, l1, j1, i, l2, j2)) * si.hertz
            if frequency > frequency_bounds[0] and frequency < frequency_bounds[1]:
                suitable_transitions.append([j, i, frequency.to(si.gigahertz)])

    return suitable_transitions

def get_transition_parameters(n1, n2, temperature_K=300):
    l1 = 2
    l2 = 3
    j1 = 2.5
    j2 = 3.5
    mj1 = j1
    mj2 = j2
    
    dipole_matrix_element = atom.getDipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, s=0.5, q=1)
    dipole_matrix_element_SI = dipole_matrix_element * bohr_radius * e

    tau1 = atom.getStateLifetime(n1, l1, j1, temperature_K, includeLevelsUpTo=150) * si.second
    tau2 = atom.getStateLifetime(n2, l2, j2, temperature_K, includeLevelsUpTo=150) * si.second

    Gamma_p = 0.5 * (1.0/tau1  + 1.0/tau2) * 2 * np.pi
    return dipole_matrix_element_SI, Gamma_p

def cavity_dims_from_aspect_ratios(f_c, m, n, p, r_ab=1.0, r_db=1.3):
    K = 2.0*f_c/c
    num = (m/r_ab)**2 + n**2 + (p/r_db)**2
    b = np.sqrt(num)/K
    a = r_ab*b
    d = r_db*b
    return a.to(si.meter), b.to(si.meter), d.to(si.meter)

class Cavity:
    def __init__(self, a, b, d, frequency, mode, Q, grid=(100,100,100), mu=None):
        """
        Rectangular cavity with TE_mnp mode.
        mode: tuple (m,n,p)
        dims: x=a, y=b, z=d in meters
        """
        self.a, self.b, self.d = a,b,d
        self.dims = (a, b, d)
        self.mode = mode
        self.mu = mu
        m,n,p = self.mode

        self.kx, self.ky, self.kz = m*np.pi/a, n*np.pi/b, p*np.pi/d # wavevector components
        self.kc2 = self.kx**2 + self.ky**2 + self.kz**2 # wavevector squared
        self.omega_geometry = (c*np.sqrt(self.kc2)).to(si.hertz) # angular frequency according to geometry

        self.frequency = frequency
        self.wavelength = c / frequency
        self.omega_design = 2 * np.pi * frequency

        self.Q = Q
        self.kappa_design = self.omega_design / Q
        self.kappa_geometry = self.omega_geometry / Q

        # we will use the geometric frequency for everything
        self.omega = self.omega_geometry
        self.kappa = self.kappa_geometry

        # self.cavity_grid = np.meshgrid(np.linspace(0, a.to(si.meter).magnitude, grid[0]), np.linspace(0, b.to(si.meter).magnitude, grid[1]), np.linspace(0, d.to(si.meter).magnitude, grid[2]))
        self.cavity_grid = np.meshgrid(np.linspace(0, a, grid[0]), np.linspace(0, b, grid[1]), np.linspace(0, d, grid[2]))

    def TE_cavity_mode_normalized(self):
        Ex = lambda x, y, z: (np.cos(self.kx.to(1 / si.meter).magnitude * x) * np.sin(self.ky.to(1 / si.meter).magnitude * y) * np.cos(self.kz.to(1 / si.meter).magnitude * z))
        Ey = lambda x, y, z: (np.sin(self.kx.to(1 / si.meter).magnitude * x) * np.cos(self.ky.to(1 / si.meter).magnitude * y) * np.sin(self.kz.to(1 / si.meter).magnitude * z))
        Ez = lambda x, y, z: 0
        return [Ex, Ey, Ez]

    def TE_cavity_mode_intensity(self):
        Ex, Ey, Ez = self.TE_cavity_mode_normalized()
        I = lambda x, y, z: Ex(x, y, z)**2 + Ey(x, y, z)**2 + Ez(x, y, z)**2
        return I

    def box_integral_nquad(self, epsabs=1e-10, epsrel=1e-10):
        f = self.TE_cavity_mode_intensity()
        res, err = nquad(f, [[0, self.a.to(si.meter).magnitude], [0, self.b.to(si.meter).magnitude], [0, self.d.to(si.meter).magnitude]], opts={'epsabs': epsabs, 'epsrel': epsrel})
        return res, err

    def V_eff(self):
        IE, IE_err = self.box_integral_nquad()
        I_c = self.TE_cavity_mode_intensity()
        return lambda x, y, z: (IE * si.meter**3) / I_c(x.to(si.meter).magnitude, y.to(si.meter).magnitude, z.to(si.meter).magnitude)

    def E_zpf(self):
        IE, IE_err = self.box_integral_nquad()
        I_c = self.TE_cavity_mode_intensity()
        return lambda x,y,z: np.sqrt(hbar * self.omega * I_c(x.to(si.meter).magnitude, y.to(si.meter).magnitude, z.to(si.meter).magnitude) / (2 * epsilon_0 * IE * si.meter**3))

    def g_0(self, mu, eta_pol=1):
        return lambda x,y,z: mu * self.E_zpf()(x,y,z) / (hbar * eta_pol)

    def cavity_params_field(self):
        V_eff_field = self.V_eff()(*self.cavity_grid)
        E_zpf_field = self.E_zpf()(*self.cavity_grid)
        g_0_field = self.g_0(self.mu)(*self.cavity_grid)
        self.V_eff_field = V_eff_field
        self.E_zpf_field = E_zpf_field
        self.g_0_field = g_0_field
        return V_eff_field, E_zpf_field, g_0_field

    def cavity_statistics(self):
        if not hasattr(self, 'g_0_field') or not hasattr(self, 'V_eff_field') or not hasattr(self, 'E_zpf_field'):
            self.cavity_params_field()
        
        g_0_max = np.max(np.abs(self.g_0_field)).to(si.hertz)
        g_0_avg = np.mean(self.g_0_field).to(si.hertz)
        V_eff_max = np.max(np.abs(self.V_eff_field)).to(si.meter**3)
        V_eff_avg = np.mean(self.V_eff_field).to(si.meter**3)
        E_zpf_max = np.max(np.abs(self.E_zpf_field)).to(si.volt / si.meter)
        E_zpf_avg = np.mean(self.E_zpf_field).to(si.volt / si.meter)
        
        return g_0_max, g_0_avg, V_eff_max, V_eff_avg, E_zpf_max, E_zpf_avg
