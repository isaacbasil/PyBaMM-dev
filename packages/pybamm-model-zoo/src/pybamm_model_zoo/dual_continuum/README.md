# DualContinuum

![status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm-model-zoo/badges/dual_continuum.json)

## Summary

Dual-continuum (DC) model of a lithium-ion cell (Paten et al., 2026), built as a
subclass of `pybamm.lithium_ion.DFN`. The electrolyte, solid-phase conduction,
kinetics, thermal and lithium-metal counter-electrode physics are PyBaMM's.
The difference is the active material (AM): instead of solving radial
diffusion in idealised spherical particles at every point of the electrode,
the AM mass balance is homogenised with the volume-averaging technique,

    d c_vol / dt = - a j / (eps_s F),

where `c_vol` is the volume-averaged AM concentration, `a` the specific surface
area, `eps_s` the AM volume fraction and `j` the reaction rate. The
surface-averaged concentration `c_surf` used in the kinetics is given by one of
three definitions (Table I of the paper):

| `"model type"` | surface concentration |
| --- | --- |
| `"DC1"` (default) | `c_surf = c_vol + <s>_A j` |
| `"DC0"` | `c_surf = c_vol` |
| `"Yang"` | `c_surf = c_vol + gamma j`, `gamma = (l_d^2/(6 l_k^2) - l_d/(2 l_k)) l_k/(D F)` |

In DC1, `<s>_A` [mol.m-1.A-1] is the **closure variable**: the surface average
of the solution of a closure problem solved once on the electrode
microstructure (e.g. with the `solveclosure` tool, ref. 56 of the paper).
It depends only on the microstructure and the AM diffusivity, not on the
operating conditions. No assumption on particle shape is made. With
`"calculate surface concentration a priori": "true"`, `j` is replaced by its
electrode average, which removes one algebraic equation per mesh cell.

Yang and Tartakovsky's model uses a boundary layer of thickness
`l_d = a_d sqrt(D t)` (up to `l_k`) inside particles of length scale `l_k`,
taken from the particle radius parameter, with a fitting parameter `a_d`.

The DC model is a fully macroscale description: it has no radial dimension.
Radial output variables (e.g. `"Positive particle concentration [mol.m-3]"`)
are the volume average broadcast in r; use `"Positive particle surface
concentration [mol.m-3]"` for the surface concentration.

## Usage

```python
import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo.dual_continuum import parameter_sets

DualContinuum = zoo.load("DualContinuum")

# Defaults: PyBaMM's DFN parameter set plus the DC parameters below
solution = pybamm.Simulation(DualContinuum()).solve([0, 3600])

# Cathode half cell
model = DualContinuum({"working electrode": "positive"})

# LG M50 cell with the parameters of Paten et al. (2026), Table IV
solution = pybamm.Simulation(
    DualContinuum(), parameter_values=parameter_sets.paten2026_dc1()
).solve([0, 3600])

# Any parameter set written for the DFN (no DC parameters needed)
model = DualContinuum(
    dc_options={"closure variable": "isolated sphere", "surface area": "spherical"}
)
solution = pybamm.Simulation(
    model, parameter_values=pybamm.ParameterValues("Chen2020")
).solve([0, 3600])
```

### DC options

| Option | Values (first is default) |
| --- | --- |
| `"model type"` | `"DC1"`, `"DC0"`, `"Yang"` |
| `"calculate surface concentration a priori"` | `"false"`, `"true"` (DC1 only) |
| `"closure variable"` | `"parameter"`, `"isolated sphere"` (DC1 only) |
| `"dimensionless closure variable"` | `"false"`, `"true"` |
| `"surface area"` | `"from image"`, `"spherical"` |
| `"lithium foil surface porosity"` | `"false"`, `"true"` (half cells only) |

- `"closure variable": "isolated sphere"` uses `<s>_A = -R_eff / (5 D F)`, the
  analytical solution of the closure problem for an isolated sphere of radius
  `R_eff`, with `D` evaluated at the local volume-averaged concentration. With
  it, DC1 coincides with PyBaMM's `"quadratic profile"` particle for spherical
  geometry.
- `"surface area": "spherical"` uses `a = 3 eps_s / R`, as in the DFN.
- `"lithium foil surface porosity": "true"` multiplies the exchange-current
  density of the lithium foil by `"Separator surface porosity"`. Some DNS
  codes include the separator surface porosity in the kinetics at the foil,
  since the foil only reacts where it meets the separator's pores; turn this on
  when comparing with such a DNS. Leave it off otherwise: a foil
  exchange-current density taken from the literature is usually given per
  geometric area, so it already includes the contact with the separator.

### Specific surface area from images

With `"surface area": "from image"`, the reactive specific surface area of each
electrode is

    a = a_AM-electrolyte + eps_CBD_surf a_AM-CBD + eps_sep_surf a_AM-separator,

with each specific area (per unit electrode volume) measured on images of the
microstructure. The carbon-binder domain (CBD) and the separator are effective
media that contain electrolyte in their pores, so a reaction still takes place
at AM-CBD and AM-separator interfaces, albeit at a reduced rate. The surface
porosities `eps_CBD_surf` and `eps_sep_surf` express that reduction: they are
the fraction of those interfaces in contact with electrolyte. **We recommend
setting each surface porosity equal to the corresponding volume porosity.**

An effective particle radius `R_eff = 3 eps_s / a` is computed from the total
area and reported as `"Positive electrode effective particle radius [m]"`; it
is the radius used by the isolated-sphere closure.

### SEI and other DFN options

The DC model has no SEI model of its own: it accepts PyBaMM's `"SEI"` options
(e.g. `DualContinuum({"SEI": "reaction limited"})`) and the DFN's other
options, with their parameters. With `"surface area": "from image"`, the SEI
grows on the same total specific surface area `a` as the main reaction,
including the AM-CBD and AM-separator contributions.

### Parameters

| Parameter | Used when |
| --- | --- |
| `"{Domain} electrode s0 surface average"` [mol.m-1.A-1], the closure variable `<s>_A` | DC1, `"closure variable": "parameter"` |
| `"{Domain} electrode s0 surface average dimensionless"` `s*`, with `<s>_A = s* L / (D F)` | DC1, `"dimensionless closure variable": "true"` |
| `"{Domain} electrode specific surface area from image (AM-electrolyte) [m-1]"` | `"surface area": "from image"` |
| `"{Domain} electrode specific surface area from image (AM-CBD) [m-1]"` | `"surface area": "from image"` |
| `"{Domain} electrode specific surface area from image (AM-separator) [m-1]"` | `"surface area": "from image"` |
| `"CBD surface porosity"`, `"Separator surface porosity"` | `"surface area": "from image"` |
| `"{Domain} electrode Yang fitting parameter"` `a_d` | Yang |

`default_parameter_values` is the DFN's default set (Marquis2019 for a full
cell, Xu2019 for a half cell) plus:

- AM-electrolyte areas equal to the DFN's `3 eps_s / R`, no AM-CBD or
  AM-separator contact, separator surface porosity equal to the separator
  porosity, and a CBD surface porosity of 0.5;
- the closure variable set to the isolated-sphere value `-R/(5 D F)` at the
  reference state, so the defaults describe the same cell as the DFN's;
- a Yang fitting parameter of 1.

`parameter_sets.paten2026_dc1()` and `parameter_sets.paten2026_dfn()` return the
DC1 and DFN columns of Table IV (an LG M50 cell, fitted to the 1C discharge of
Chen et al. 2020), on top of PyBaMM's Chen2020 set for all other values. The
electrolyte tortuosities are given both as tortuosity factors and as the
equivalent Bruggeman coefficients, so either `"transport efficiency"` option
gives the same result.

## Validation

Pinned by `tests/test_dual_continuum.py`, for a full cell (Marquis2019) and a
cathode half cell (Xu2019):

- DC0 reproduces DFN `"uniform profile"` (rtol 1e-7).
- DC1 with the isolated-sphere closure variable reproduces DFN
  `"quadratic profile"` (rtol 1e-7), as do the default parameters, with
  dimensional and dimensionless closure variables.
- The image-based specific surface area and `R_eff` follow the formula above,
  and `"lithium foil surface porosity"` scales the foil exchange current by the
  separator surface porosity.
- Lithium in the active material is conserved to round-off.
- In a half cell with AM-CBD and AM-separator areas, surface porosities and
  the foil scaling, DC0, DC1 and DC1 a priori match reference voltages from
  the standalone implementation used in the paper to within 10 µV
  (`tests/data/standalone_reference.csv`, whose header records how it was
  generated).
- Splitting the specific surface area into AM-electrolyte, AM-CBD and
  AM-separator contributions gives the same result as the same total area.
- The Table IV parameter sets give identical results with the Bruggeman and
  tortuosity-factor transport options.
- With PyBaMM's `"SEI"` options, DC1 with the isolated-sphere closure variable
  reproduces the DFN with the `"quadratic profile"` particle (rtol 1e-7).

Not yet validated here: the Yang closure (only its dependence on the particle
radius is checked), and the experimental comparison of the paper. Not yet
ported from the standalone implementation: transient closure inputs and
minimum plating overpotential outputs. Particle-size
distributions, multiple particle phases and loss of active material are not
supported.

## Citation

See `CITATION.bib`. Please cite Paten2026 when using this model; it is
registered automatically, so `pybamm.print_citations()` credits it after use.

## Maintainer

Isaac Paten (@isaacbasil) — tier: community
