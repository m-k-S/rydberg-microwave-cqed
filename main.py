# cavity calcs - Q, omega, kappa, V_eff, E_zpf, eta_pol
# atoms calcs - mu (dipole), gamma (decoherence)
# optics calcs - pump beam intensity, pump beam waist, pump beam divergence
# coupling calcs - g_0, G, C

# define cavity
# estimate Q, or take from COMSOL
# calculate U(r)
# calculate Veff
# calculate Ezpf

# define atoms
# get frequency, lifetime, dipole matrix element
# estimate gamma_phi_Hz
# calculate gamma
# calculate single_atom_g

# define optics
# simulate atom cloud 
# simulate beams (780 and 481)
# estimate pumped atom number

# coupling calculations
# estimate collective G, 

import numpy as np
from pint import UnitRegistry, set_application_registry
si = UnitRegistry()
set_application_registry(si)

from cavity import generate_cavity
from atoms import (
    find_suitable_transitions,
    get_transition_parameters,
    single_atom_g,
    atom_number,
    saturation_intensity_Rb_780,
    gamma_780,
)
from optics import estimate_pumped_atoms_general, sample_positions_uniform

### PARAMETERS ###

# for atoms:
# - transition bounds / transition frequency
transition_bounds = [10 * 1e9 * si.hertz, 20 * 1e9 * si.hertz]

# for cavity:
# - estimated Q
Q = 1e5
# - mode (m,n,p)
cavity_mode = (1, 0, 1)

# for optics:
# - number of atoms; can estimate from vapor pressure * cavity volume
# - laser intensities (780, 481); power and beam waist
laser_power_780 = 100 * si.milliwatt
beam_waist_780 = 2 * si.millimeter
laser_power_481 = 500 * si.milliwatt
beam_waist_481 = 2 * si.millimeter
# - laser detunings 
Delta_ge = 0 * si.hertz
Delta_er = 0 * si.hertz
# - estimated gamma dephasing (from experimental conditions; usually dominated by Doppler at room temp)
gamma_ge_phi = 100 * si.kilohertz
gamma_er_phi = 100 * si.kilohertz
# simulation params:
#   - grid points
#   - t_final
grid_points = 5
t_final = 30 * si.microsecond

### CALCULATIONS ###

# Define the cavity based on some Rydberg atom calculations
transitions = find_suitable_transitions(transition_bounds)
slowest_transition = min(transitions, key=lambda x: x[2])

dipole_matrix_element, gamma_Ryd_population = get_transition_parameters(slowest_transition[0], slowest_transition[1])

# Generate cavity
Cavity = generate_cavity(
    frequency=slowest_transition[2],
    mode=cavity_mode,
    Q=Q
)

E_zpf = Cavity.E_zpf_best()

g0 = single_atom_g(
    E_zpf,
    dipole_matrix_element
)

# # Optical / laser pumping simulations
# N_atom = int(atom_number(Cavity.volume()).magnitude)
N_atom = 100
positions = sample_positions_uniform(Cavity.x, Cavity.y, Cavity.z, N_atom)

pump_frac, N_pumped = estimate_pumped_atoms_general(
    positions,
    laser_power_780,
    beam_waist_780,
    laser_power_481,
    beam_waist_481,
    saturation_intensity_Rb_780,
    dipole_matrix_element,
    Delta_ge,
    Delta_er,
    gamma_780,
    gamma_Ryd_population,
    gamma_ge_phi,
    gamma_er_phi,
    grid_points,
    t_final
)

print (pump_frac)