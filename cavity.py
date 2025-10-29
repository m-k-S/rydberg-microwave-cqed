from scipy.constants import c, epsilon_0, mu_0, hbar
from pint import get_application_registry
import numpy as np

si = get_application_registry()

c = c * si.meter / si.second
mu_0 = mu_0 * si.newton / si.ampere**2
epsilon_0 = epsilon_0 * si.farad / si.meter
hbar = hbar * si.joule * si.second

class Cavity:
    def __init__(self, a, b, d, frequency, mode, Q):
        """
        Rectangular cavity with TE_mnp mode.
        mode: tuple (m,n,p)
        dims: x=a, y=b, z=d in meters
        """
        self.a, self.b, self.d = a,b,d
        self.dims = (a, b, d)
        self.mode = mode
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

        print(f"Cavity dimensions:\n a={self.a:#~P}\n b={self.b:#~P}\n d={self.d:#~P}")
        print(f"Design angular frequency vs. angular frequency according to geometry:\n design: {self.omega_design:#~P} \n geometry: {self.omega_geometry:#~P}")

    def spatial_volume(self):
        return self.a * self.b * self.d

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

    # this is a function of (x,y,z)
    def V_eff(self):
        IE, IE_err = self.box_integral_nquad()
        I_c = self.TE_cavity_mode_intensity()
        return lambda x, y, z: (IE * si.meter**3) * I_c(x.to(si.meter).magnitude, y.to(si.meter).magnitude, z.to(si.meter).magnitude)

    def E_zpf(self):
        return lambda x,y,z: np.sqrt(hbar * self.omega / (2 * epsilon_0 * self.V_eff()(x,y,z)))

    def Purcell(self):
        return lambda x,y,z: 3 * self.wavelength**3 * self.Q / (4 * np.pi**2 * self.V_eff()(x,y,z))

def cavity_dims_from_aspect_ratios(f_c, m, n, p, r_ab=1.0, r_db=1.3):
    K = 2.0*f_c/c
    num = (m/r_ab)**2 + n**2 + (p/r_db)**2
    b = np.sqrt(num)/K
    a = r_ab*b
    d = r_db*b
    return a.to(si.meter), b.to(si.meter), d.to(si.meter)

