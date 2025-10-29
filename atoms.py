from arc import Rubidium85
import numpy as np
from scipy.constants import hbar, e, Boltzmann, physical_constants
from pint import get_application_registry

si = get_application_registry()
bohr_radius = physical_constants['Bohr radius'][0] * si.meter 
hbar = hbar * si.joule * si.second
kB = Boltzmann * si.joule / si.kelvin
e = e * si.coulomb

room_temp_vapor_pressure_Rb = 4.9 * 10**-5 * si.pascal
saturation_intensity_Rb_780 = 3.129 * si.milliwatt / si.centimeter**2
gamma_780 = 6 * 1e6 * si.hertz

def find_suitable_transitions(frequency_bounds, max_n=100):
    atom = Rubidium85()
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
                suitable_transitions.append([j, i, frequency])

    return suitable_transitions

def get_transition_parameters(n1, n2, temperature_K=300):
    atom = Rubidium85()
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

def atom_number(cavity_volume, temperature = 300 * si.kelvin):
    density = room_temp_vapor_pressure_Rb / (kB * temperature)
    return density * cavity_volume

def single_atom_g(Ezpf, dipole_matrix_element):
    # single atom coupling rate
    return ((dipole_matrix_element * Ezpf) / hbar).to(si.hertz)

def estimate_collective_G_from_E(positions, Ezpf, dipole_matrix_element, pump_fraction):
    """
    Ezpf(positions) -> either:
      - (N,) array of V/m along dipole direction, or
      - (N,3) array (V/m) and we project with e_d.
    mu: dipole [C·m]
    """
    E = Ezpf(positions)  # V/m
    if E.ndim == 2:
        e = np.asarray(e_d if e_d is not None else [1,0,0], float)
        e /= np.linalg.norm(e)
        Edot = np.abs(E @ e)
    else:
        Edot = np.abs(E)

    gi = (dipole_matrix_element * Edot) / hbar        # 1/s
    G2 = np.sum((gi**2) * pump_fraction)  # 1/s^2
    return np.sqrt(G2), gi, Edot
    