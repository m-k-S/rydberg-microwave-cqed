- Geometry, frequency -> V_eff, omega_c, Q, kappa - > E_zpf
- Rydberg optical sims -> mu, C_CG, eta_pol, gamma
- Atom cloud pumping sims -> N_eff

- Zero-point field -> g_0 
- Scaled by N_eff -> G
- From G, kappa, gamma -> C and then observable splitting 2G (compare to simulated)

**Cavity**
- Q = cavity quality factor, dimensionless
- omega_c = cavity resonance frequency (set by geometry)
- kappa = cavity decay rate = omega_c / 2piQ, the linewidth of the empty cavity resonance
- V_eff = effective mode volume, is equal to the integral of |E|^2 over dV normalized by the maximum |E|^2. A smaller V_eff means stronger fields per single photon
- E_zpf = zero-point field amplitude = sqrt(hbar omega_c / 2 eps_0 V_eff), which means the RMS electric field per vacuum photon

V_eff is a function primarily of the geometry, but also depends on the dipole orientation of the atoms / the Rydberg transition. In the absence of a bias field, there is a degeneracy in the m_j sublevels, so there is no quantized axis. Thus we can assume that the dipole orientation (e_d) is aligned with the cavity mode polarization, and just take the denominator of V_eff as e_0 |E(r_0)|^2.

**Atoms**
- mu = electric dipole matrix element for the given Rydberg transition (nS -> mP etc), from ARC
- gamma = atomic decoherence rate, set by the Rydberg state, dephasing due to collisions, etc
- eta_pol (aka p_0) = polarization overlap, cavity mode polarization has to match atomic dipole orientation (see above)

**Optics**
- N_eff = the number of atoms that couple to the mode, N_total <|U(r)|^2> where u(r) is the cavity mode function, averaged over atom distribution

**Coupling**
- g_0 = single-atom coupling rate, g_0 = mu E_zpf / hbar * eta_pol 
- G = collective coupling rate, g_0 sqrt(N_eff)
- 2G = vacuum Rabi splitting
- C = cooperativity, 4G^2 / kappa * gamma (should be >1 for strong coupling regime), i.e. the Rabi splitting is observable


**TODO**
- use E_zpf to get g_0

- calculate optical pumping
- then calculate N_eff
- then calculate G


------

## notes 9/16

- want: small effective mode volume; Q ~ 1/V_eff
- V_eff, E_zpf, and thus g_0 are functions of space (r_0); or in other words the location of an atom
- for computing a single value for g_0, just take the one that maximizes V_eff (this is essentially the 'worst case' single atom coupling rate)
- the argument of V_eff is the cavity mode (the electric field)

**getDrivingPower notes**
- n1 = 5 (we are calculating the driving power for the 5P -> Rydberg transition)
- what is the rabiFrequency? 
    - rabiFrequency = 5MHz (what is this?)


- 480 locked to EIT (on resonance) through the vacuum chamber itself
- assume a detuning of 0?

- Gamma_k; dependent on dephasing
    - Gamma_transit is atoms just leave the interaction region (v_rms of atoms / optical waist or cavity mode waist)

- experimental question: VNA?
- just need homodyne detection; Bejoy and Hanfeng have built this circuit
- phase noise on VNA should be: k_B * T (W / Hz, power spectral density), ~174dBm/Hz