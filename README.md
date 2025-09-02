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

**Atoms**
- mu = electric dipole matrix element for the given Rydberg transition (nS -> mP etc), from ARC
- C_CG = Clebsch-Gordan coefficient for the mF states involved, dimensionless
- gamma = atomic decoherence rate, set by the Rydberg state, dephasing due to collisions, etc
- eta_pol (or p_0) = polarization overlap, cavity mode polarization has to match atomic dipole orientation

**Optics**
- N_eff = the number of atoms that couple to the mode, N_total <|U(r)|^2> where u(r) is the cavity mode function, averaged over atom distribution

**Coupling**
- g_0 = single-atom coupling rate, g_0 = mu E_zpf / hbar * eta_pol * C_CG
- G = collective coupling rate, g_0 sqrt(N_eff)
- 2G = vacuum Rabi splitting
- C = cooperativity, 4G^2 / kappa * gamma (should be >1 for strong coupling regime), i.e. the Rabi splitting is observable


**TODO**
- use E_zpf to get g_0

- calculate optical pumping
- then calculate N_eff
- then calculate G
