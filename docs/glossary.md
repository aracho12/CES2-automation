# Glossary

<!-- Seeded with the terms first-timers ask about; TODO: extend. -->

**CES / DFT-CES2** — *DFT in Classical Explicit Solvent*: mean-field QM/MM
coupling where QM and MM exchange time-averaged densities/potentials on grids
instead of instantaneous forces.

**QM/MM step (outer loop)** — one iteration of (MD equilibration → MD averaging
→ QM SCF → cube post-processing), producing one `qm_N/` + `mm_N/` pair.

**ortho / non-ortho** — the two `pw.x` runs per step. *Ortho*: MM Pauli
repulsion (`repA.cube`) included, QM density pushed out of the solvent region —
use for observables. *Non-ortho*: repulsion off — used to build the coupling
potentials for the next MM run.

**SAW potential (`v_saw.cube`)** — the sawtooth potential QE adds for the
dipole correction of a periodic slab (`edir`, `emaxpos`). Added into the solute
potentials so the MM side feels the corrected field.

**blur** — Gaussian smearing of a density cube. MM densities are blurred (with
element-specific radius from the isotropic polarizability α) into effective
Pauli-repulsion potentials; the QM density is blurred per MM species into
ρ̂^A (`c_rho_*.cube`).

**repA** — element-specific Pauli-repulsion strength (a.u.); together with the
blur it models exchange repulsion between QM electrons and MM atoms.

**`MOBILE_final.cube`** — the total time-averaged MM charge density (mobile
ions/water charges + polarization bound charge, dipole-corrected) handed to
`pw.x` as `rho_ces`.

**PZC** — potential of zero charge; the `TOTCHG=0` run.

**TOTCHG** — QE `tot_charge` *per unit cell* = total electrode charge ÷
supercell factor. Negative = extra electrons on the electrode.

**point-charge scheme / plate scheme** — two ways the compensating counter
charge of a charged electrode is represented when post-processing the QM
potentials: point charges on the top slab layer (`mdipc`, for electrostatics)
vs. a uniform plate inside the slab (`make_rho_mino`, for induction).

**equilibration vs. averaging run** — each MD phase runs twice: first to relax
the solvent in the new QM potential (nothing recorded), then to accumulate the
time-averaged density cubes and dispersion forces.

**dispersion forces (`dispf.ave`)** — time-averaged MM→QM van der Waals forces
on the slab atoms, fed back into `pw.x` via the `###dispf` marker so QM forces
(for `opt`) include the solvent's pull.

**`solute`** — QE's `prefix` for the QM system; hence `solute.save/`,
`solute.rho.cube`, `solute.pot.cube`.
