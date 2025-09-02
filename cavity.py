from scipy.constants import c, epsilon_0, mu_0, hbar
from pint import get_application_registry
import numpy as np

si = get_application_registry()
c = c * si.meter / si.second
mu_0 = mu_0 * si.newton / si.ampere**2
epsilon_0 = epsilon_0 * si.farad / si.meter
hbar = hbar * si.joule * si.second

class Cavity:
    def __init__(self, x, y, z, frequency, mode, Q):
        """
        Rectangular cavity with TE_mnp mode.
        mode: tuple (m,n,p)
        dims: x=a, y=b, z=d in meters
        """
        self.x, self.y, self.z = x,y,z
        self.dims = (x, y, z)
        self.mode = mode
        m,n,p = self.mode

        self.kx, self.ky, self.kz = m*np.pi/x, n*np.pi/y, p*np.pi/z # wavevector components
        self.kc2 = self.kx**2 + self.ky**2 + self.kz**2 # wavevector squared
        self.omega_geometry = (c*np.sqrt(self.kc2)).to(si.hertz) # angular frequency according to geometry

        self.frequency = frequency
        self.omega_design = 2 * np.pi * frequency

        self.Q = Q
        self.H0 = (1.0 * si.ampere / si.meter)

        self.kappa_design = self.omega_design / Q
        self.kappa_geometry = self.omega_geometry / Q

        # we will use the geometric frequency for everything
        self.omega = self.omega_geometry
        self.kappa = self.kappa_geometry

        print(f"Cavity dimensions:\n x={self.x:#~P}\n y={self.y:#~P}\n z={self.z:#~P}")
        print(f"Design angular frequency vs. angular frequency according to geometry:\n design: {self.omega_design:#~P} \n geometry: {self.omega_geometry:#~P}")

    def _int_E2(self):
        """compute the integral of the squared electric field over the cavity volume ∫|E|^2 dV"""

        # if p=0, cos(k_z * z) = 1, and therefore the integral over z is just the z dimension
        # else it's z/2
        z_factor = 2.0 if self.mode[2]==0 else 1.0
        
        IE = (mu_0**2*c**2/self.kc2) * (self.x*self.y*self.z/8.0) * (self.kx**2+self.ky**2) * z_factor * self.H0**2
        return IE

    def volume(self):
        return self.x * self.y * self.z

    def U(self, pts):
        """
        Time-averaged energy density U(r) [J/m^3] at pts[:,3].
        Normalized so ∫U dV = ½ ħω (vacuum energy of the mode).
        """
        IE = self._int_E2()
        s = np.sqrt((hbar * self.omega) / (epsilon_0 * IE))   # dimensionless scale

        # shift coords into [0,L] for the sin/cos forms
        X = pts[:, 0] + self.x/2
        Y = pts[:, 1] + self.y/2
        Z = pts[:, 2] + self.z/2

        # --- phasor fields for TE_mnp with base amplitude H0 ---
        # E_z = 0 for TE; E ∝ (ω μ0 / k_c^2) * (transverse derivatives of H_z)
        Ex =  1j * self.omega * mu_0 / self.kc2 * ( self.ky * np.cos(self.kx*X) * np.sin(self.ky*Y) * np.cos(self.kz*Z) ) * self.H0
        Ey = -1j * self.omega * mu_0 / self.kc2 * ( self.kx * np.sin(self.kx*X) * np.cos(self.ky*Y) * np.cos(self.kz*Z) ) * self.H0

        # H fields (from H_z = cos(kxX) cos(kyY) cos(kzZ) with amplitude H0)
        Hx = -(self.kx*self.kz/self.kc2) * np.sin(self.kx*X) * np.cos(self.ky*Y) * np.sin(self.kz*Z) * self.H0
        Hy = -(self.ky*self.kz/self.kc2) * np.cos(self.kx*X) * np.sin(self.ky*Y) * np.sin(self.kz*Z) * self.H0
        Hz =  np.cos(self.kx*X) * np.cos(self.ky*Y) * np.cos(self.kz*Z) * self.H0

        # apply vacuum normalization
        Ex, Ey, Hx, Hy, Hz = s*Ex, s*Ey, s*Hx, s*Hy, s*Hz

        E2 = np.abs(Ex)**2 + np.abs(Ey)**2                  # (V/m)^2
        H2 = np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2  # (A/m)^2

        return 0.25 * (epsilon_0 * E2 + mu_0 * H2).real     # J/m^3

    def V_eff(self, r0, e_d):
        """Effective mode volume at point r0 (m) with dipole direction e_d."""
        IE = self._int_E2()
        X,Y,Z = r0[0]+self.x/2, r0[1]+self.y/2, r0[2]+self.z/2

        Ex = 1j*self.omega*mu_0/self.kc2*(self.ky*np.cos(self.kx*X)*np.sin(self.ky*Y)*np.cos(self.kz*Z)*self.H0)
        Ey =-1j*self.omega*mu_0/self.kc2*(self.kx*np.sin(self.kx*X)*np.cos(self.ky*Y)*np.cos(self.kz*Z)*self.H0)
        E = np.array([Ex,Ey,0.0],dtype=complex)

        e = np.array(e_d,dtype=float); e/=np.linalg.norm(e)
        denom = epsilon_0*np.abs(np.dot(e,E))**2

        return np.inf if denom==0 else (epsilon_0*IE)/denom # m^3

    def E_zpf(self, r0, e_d):
        """Zero-point RMS field at r0 along e_d, in V/m"""
        Veff = self.V_eff(r0,e_d)
        if not np.isfinite(Veff): 
            return 0.0
        return np.sqrt(hbar*self.omega/(2*epsilon_0*Veff))

    def V_eff_best(self, pol="auto"):
        """Best-case constant V_eff (atom at antinode, optimal pol)."""
        m,n,p = self.mode
        mx2, ny2 = (m/self.x)**2, (n/self.y)**2
        denom = max(mx2,ny2) if pol=="auto" else (mx2 if pol=="x" else ny2)
        factor_p = 2.0 if p==0 else 1.0
        return factor_p*(self.x*self.y*self.z/16.0)*((mx2+ny2)/denom)

    def E_zpf_best(self, pol="auto"):
        """Best-case constant E_zpf."""
        Veff = self.V_eff_best(pol)
        return np.sqrt(hbar*self.omega/(2*epsilon_0*Veff))

def generate_cavity(frequency, mode, Q):
    m, n, p = mode
    wavelength = (c / frequency).to(si.meter)

    if m == 0 and n == 0:
        z = p * wavelength / 2
        x = 0.75 * wavelength
        y = 1.0 * wavelength
    elif n == 0 and p == 0:
        x = m * wavelength / 2
        z = 0.75 * wavelength
        y = 1.0 * wavelength
    elif m == 0 and p == 0:
        y = n * wavelength / 2
        x = 0.75 * wavelength
        z = 1.0 * wavelength
    else:
        x = 0.75 * wavelength
        y = 1.0 * wavelength
        z = p / np.sqrt((2 / wavelength)**2 - (m / x)**2 - (n / y)**2)

    return Cavity(x, y, z, frequency, mode, Q)

