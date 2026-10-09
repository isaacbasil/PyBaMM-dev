"""Compare dual-continuum closures with the full DFN for a cathode half cell."""

import numpy as np

import pybamm
import pybamm_model_zoo as zoo

HALF_CELL = {"working electrode": "positive"}
DualContinuum = zoo.load("DualContinuum")

parameter_values = DualContinuum(HALF_CELL).default_parameter_values
parameter_values["Current function [A]"] *= 2
times = np.linspace(0, 1500, 101)

models = {
    "DFN (Fickian)": pybamm.lithium_ion.DFN(HALF_CELL),
    "DC0": DualContinuum(HALF_CELL, {"model type": "DC0"}),
    "DC1 (implicit)": DualContinuum(HALF_CELL, {"model type": "DC1"}),
    "DC1 (a priori)": DualContinuum(
        HALF_CELL,
        {"model type": "DC1", "calculate surface concentration a priori": "true"},
    ),
}

voltages = {}
for label, model in models.items():
    solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
        [0, times[-1]], t_interp=times
    )
    voltages[label] = solution["Voltage [V]"](times)
    print(f"{label:16s} solve time {solution.solve_time}")

reference = voltages["DFN (Fickian)"]
for label, value in voltages.items():
    error = np.abs(value - reference).max() * 1e3
    print(f"{label:16s} max |V - V_DFN| = {error:6.2f} mV")
