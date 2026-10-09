"""Generate reference voltages for the dual_continuum tests with this model.

Runs the standalone implementation (DCModelMyScripts) on a cathode half cell
that exercises the AM-CBD and AM-separator areas, their surface porosities and
the lithium-foil scaling, and writes the voltages to
``dual_continuum/tests/data/standalone_reference.csv``. Run from the
repository root with ``uv run python <this file>``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pybamm
import pybamm_model_zoo as zoo

SOLVE_TIME = 3000
TIMES = np.linspace(0, SOLVE_TIME, 100)
AREA = "Positive electrode specific surface area from image"
#: Changes to DualContinuum's half-cell defaults; test_dual_continuum.py
#: applies the same ones.
REFERENCE_CHANGES = {
    f"{AREA} (AM-CBD) [m-1]": 5.0e4,
    f"{AREA} (AM-separator) [m-1]": 2.0e4,
    "CBD surface porosity": 0.3,
    "Separator surface porosity": 0.6,
}
CASES = {
    "DC0": {"model type": "DC0", "calculate surface concentration a priori": "true"},
    "DC1": {"model type": "DC1"},
    "DC1 a priori": {
        "model type": "DC1",
        "calculate surface concentration a priori": "true",
    },
}
OUTPUT = (
    Path(__file__).parents[1]
    / "dual_continuum"
    / "tests"
    / "data"
    / "standalone_reference.csv"
)


def standalone_parameters(pv):
    """Translate PyBaMM half-cell parameter values into this model's names."""
    eps_s = pv["Positive electrode active material volume fraction"]
    eps_e = pv["Positive electrode porosity"]
    eps_sep = pv["Separator porosity"]
    area = pv["Electrode height [m]"] * pv["Electrode width [m]"]
    j0_li = pv["Exchange-current density for lithium metal electrode [A.m-2]"]
    c_li_metal = 1 / pv["Lithium metal partial molar volume [m3.mol-1]"]
    brugg = "Bruggeman coefficient"
    same = [
        "Electrode height [m]",
        "Electrode width [m]",
        "Initial concentration in electrolyte [mol.m-3]",
        "Electrolyte diffusivity [m2.s-1]",
        "Electrolyte conductivity [S.m-1]",
        "Cation transference number",
        "Positive electrode porosity",
        "Maximum concentration in positive electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
        "Positive particle diffusivity [m2.s-1]",
        "Separator thickness [m]",
        "Positive electrode thickness [m]",
        "Positive electrode conductivity [S.m-1]",
        "Positive electrode OCP [V]",
        "Positive electrode exchange-current density [A.m-2]",
        "Positive electrode s0 surface average",
        "CBD surface porosity",
        "Separator surface porosity",
        f"{AREA} (AM-electrolyte) [m-1]",
        f"{AREA} (AM-CBD) [m-1]",
        f"{AREA} (AM-separator) [m-1]",
    ]
    values = {key: pv[key] for key in same}
    values.update(
        {
            "Ideal gas constant [J.K-1.mol-1]": pybamm.constants.R.value,
            "Faraday constant [C.mol-1]": pybamm.constants.F.value,
            # Negative current density is a discharge here
            "Current density [A.m-2]": -pv["Current function [A]"] / area,
            "Temperature [K]": pv["Ambient temperature [K]"],
            "Separator porosity": eps_sep,
            "Separator tortuosity (electrolyte)": eps_sep
            ** (1 - pv[f"Separator {brugg} (electrolyte)"]),
            "Thermodynamic factor": lambda c_e, T: 1.0 + 0 * c_e,
            "Positive electrode active material volume fraction": eps_s,
            "Positive electrode tortuosity (electrode)": eps_s
            ** (1 - pv[f"Positive electrode {brugg} (electrode)"]),
            "Positive electrode tortuosity (electrolyte)": eps_e
            ** (1 - pv[f"Positive electrode {brugg} (electrolyte)"]),
            "Positive electrode rate constant": 0.0,
            # This model passes c_Li = 1; PyBaMM passes 1 / (Li partial molar volume)
            "Exchange-current density for lithium metal electrode [A.m-2]": (
                lambda c_e, c_Li, T: j0_li(c_e, c_li_metal, T)
            ),
        }
    )
    return pybamm.ParameterValues(values)


def solve_standalone(dc_options, pv, points=20):
    options = {
        "cell type": "Cathode half cell",
        "effective properties": "false",
        "dimensionless closure variable": "false",
        "active material-CBD interface": "positive",
        "active material-separator interface": "positive",
        **dc_options,
    }
    model = zoo.load("DCModelMyScripts")(dc_options=options)
    geometry = {
        "separator": {model.x_s: {"min": pybamm.Scalar(0), "max": model.L_s}},
        "positive electrode": {
            model.x_p: {"min": model.L_s, "max": model.L_s + model.L_p}
        },
    }
    parameters = standalone_parameters(pv)
    parameters.process_model(model)
    parameters.process_geometry(geometry)
    mesh = pybamm.Mesh(
        geometry,
        dict.fromkeys(geometry, pybamm.Uniform1DSubMesh),
        {model.x_s: points, model.x_p: points},
    )
    discretisation = pybamm.Discretisation(
        mesh, {domain: pybamm.FiniteVolume() for domain in geometry}
    )
    discretisation.process_model(model)
    solver = pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-8)
    solution = solver.solve(model, [0, SOLVE_TIME], t_interp=TIMES)
    return solution["Voltage [V]"](TIMES)


def main():
    pv = zoo.load("DualContinuum")({"working electrode": "positive"})
    pv = pv.default_parameter_values
    pv.update(REFERENCE_CHANGES)
    columns = {"Time [s]": TIMES}
    for label, dc_options in CASES.items():
        columns[f"{label} voltage [V]"] = solve_standalone(dc_options, pv)

    header = "\n".join(
        [
            "Reference voltages from the standalone dual-continuum implementation",
            "(dc_model_my_scripts/BaseDC.py), generated by",
            "dc_model_my_scripts/generate_reference.py.",
            f"PyBaMM {pybamm.__version__}; DualContinuum half-cell defaults (Xu2019)",
            f"with {REFERENCE_CHANGES};",
            "20 + 20 finite volumes; IDAKLU rtol 1e-7, atol 1e-8.",
            ",".join(columns),
        ]
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    np.savetxt(
        OUTPUT,
        np.column_stack(list(columns.values())),
        delimiter=",",
        header=header,
        fmt="%.10g",
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
