# DualContinuum

![status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm-model-zoo/badges/dual_continuum.json)

## Summary

Dual-continuum (DC) model of a lithium-ion cell (Paten et al., 2026), built as a
subclass of `pybamm.lithium_ion.DFN`. The electrolyte, solid-phase conduction,
kinetics, thermal and lithium-metal counter-electrode physics are PyBaMM's. Only
the particle problem changes: instead of solving radial diffusion at every
macroscale point, the model solves a mass balance for the volume-averaged
active-material concentration,

    d c_avg / dt = - a j / (eps_s F),

and closes the surface concentration seen by the kinetics with

| `"model type"` | closure |
| --- | --- |
| `"DC0"` | `c_surf = c_avg` |
| `"DC1"` | `c_surf = c_avg + s0 * j` (local `j`, implicit), or `c_surf = c_avg + s0 * j_avg` with `"calculate surface concentration a priori": "true"` |
| `"Yang"` | diffusion-length correction with a time-dependent length (Yang et al.) |

The specific surface area can be `3 eps_s / R` (`"surface area": "spherical"`) or
a parameter, e.g. measured on tomography images (`"surface area": "from parameter"`).

## Usage

```python
import pybamm
import pybamm_model_zoo as zoo

DualContinuum = zoo.load("DualContinuum")
model = DualContinuum(
    options={"working electrode": "positive"},  # cathode half cell
    dc_options={"model type": "DC1", "dimensionless closure variable": "false"},
)
parameter_values = model.default_parameter_values
parameter_values["Positive electrode s0 surface average"] = -1100.0  # [mol.A-1.m-1]
solution = pybamm.Simulation(model, parameter_values=parameter_values).solve([0, 3600])
print(solution["Voltage [V]"](1800))
```

New parameters:

| Parameter | When |
| --- | --- |
| `"{Domain} electrode s0 surface average"` [mol.A-1.m-1] | DC1, `"dimensionless closure variable": "false"` |
| `"{Domain} electrode s0 surface average dimensionless"`, with `s0 = s0* L / (D_s F)` | DC1, `"dimensionless closure variable": "true"` (default) |
| `"{Domain} electrode surface area to volume ratio [m-1]"` (function of x) | `"surface area": "from parameter"` |
| `"{Domain} electrode Yang fitting parameter"` | Yang |

The defaults set `s0 = -R / (5 D_s F)`, which makes DC1 identical to PyBaMM's
`"quadratic profile"` particle.

Options of the standalone `DCModelMyScripts` (BaseDC) model map as follows:
`"cell type"` → PyBaMM `"working electrode"`; `"effective properties"` → PyBaMM
`"transport efficiency"` (e.g. `"tortuosity factor"`); the AM–CBD and
AM–separator interfaces and the separator surface porosity → fold them into the
surface-area parameter and the lithium-metal exchange-current function.

## Validation

Pinned by `tests/test_dual_continuum.py`, for a full cell (Marquis2019) and a
cathode half cell (Xu2019):

- DC0 reproduces DFN `"uniform profile"` (rtol 1e-7).
- DC1 with `s0 = -R/(5 D_s F)`, dimensional and dimensionless, reproduces DFN
  `"quadratic profile"` (rtol 1e-7).
- `"surface area": "from parameter"` with `a = 3 eps_s / R` reproduces `"spherical"`.
- Lithium in the particles is conserved to round-off.
- Half cell: DC0, DC1 implicit and DC1 a priori match the standalone BaseDC
  implementation to within 10 µV (the difference shrinks with solver tolerance).

Not yet validated: the Yang closure (only checked to solve), full-cell
comparison against BaseDC, and calibrated closure coefficients against
pore-scale simulations. Not yet ported from BaseDC: Schneider2022 SEI, transient
closure inputs, minimum plating overpotential outputs. Particle-size
distributions and multiple particle phases are not supported.

## Citation

See `CITATION.bib`. Please cite Paten2026 when using this model; it is
registered automatically, so `pybamm.print_citations()` credits it after use.

## Maintainer

Isaac Paten (@isaacbasil) — tier: community
