#
# Tests for the "surface area to volume ratio": "from parameter" option
#
import numpy as np
import pytest

import pybamm

MODELS = [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN]
PARTICLES = ["Fickian diffusion", "quadratic profile"]
FROM_PARAMETER = {"surface area to volume ratio": "from parameter"}


def parameter_values_with_surface_area(factor):
    """Chen2020 with each electrode's surface area set to ``factor * 3 eps_s / R``."""
    parameter_values = pybamm.ParameterValues("Chen2020")
    for Domain in ["Negative", "Positive"]:
        eps_s = parameter_values[f"{Domain} electrode active material volume fraction"]
        R = parameter_values[f"{Domain} particle radius [m]"]
        parameter_values.update(
            {
                f"{Domain} electrode surface area to volume ratio [m-1]": factor
                * 3
                * eps_s
                / R
            },
            check_already_exists=False,
        )
    return parameter_values


def solve(model, parameter_values, t_eval=None):
    t_eval = np.linspace(0, 3000, 101) if t_eval is None else t_eval
    simulation = pybamm.Simulation(model, parameter_values=parameter_values)
    return simulation.solve([0, t_eval[-1]], t_interp=t_eval)


class TestSurfaceAreaFromParameter:
    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("particle", PARTICLES)
    def test_reduces_to_particle_shape(self, model_class, particle):
        # With a = 3 eps_s / R the source vanishes and the models coincide
        parameter_values = parameter_values_with_surface_area(1)
        t = np.linspace(0, 3000, 101)
        default = solve(model_class({"particle": particle}), parameter_values, t)
        specified = solve(
            model_class({"particle": particle, **FROM_PARAMETER}), parameter_values, t
        )
        np.testing.assert_allclose(
            specified["Voltage [V]"](t), default["Voltage [V]"](t), rtol=1e-6, atol=0
        )

    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("particle", PARTICLES)
    def test_conserves_lithium(self, model_class, particle):
        parameter_values = parameter_values_with_surface_area(2)
        solution = solve(
            model_class({"particle": particle, **FROM_PARAMETER}), parameter_values
        )
        t = solution.t
        lithium = solution["Total lithium in particles [mol]"](t)
        np.testing.assert_allclose(lithium, lithium[0], rtol=1e-8)
        # Lithium leaves the negative particles at the rate set by the current
        negative = solution["Total lithium in negative electrode [mol]"](t)
        charge = (
            solution["Discharge capacity [A.h]"](t) * 3600 / pybamm.constants.F.value
        )
        np.testing.assert_allclose(
            negative[0] - negative, charge, rtol=1e-6, atol=1e-10
        )

    def test_surface_area_changes_the_solution(self):
        t = np.linspace(0, 3000, 101)
        model = pybamm.lithium_ion.DFN(FROM_PARAMETER)
        doubled = solve(model, parameter_values_with_surface_area(2), t)
        default = solve(pybamm.lithium_ion.DFN(), pybamm.ParameterValues("Chen2020"), t)
        parameter_values = pybamm.ParameterValues("Chen2020")
        eps_s = parameter_values["Positive electrode active material volume fraction"]
        R = parameter_values["Positive particle radius [m]"]
        surface_area = doubled["Positive electrode surface area to volume ratio [m-1]"]
        np.testing.assert_allclose(surface_area(t), 2 * 3 * eps_s / R, rtol=1e-12)
        # A larger surface area lowers the reaction overpotential
        assert np.max(np.abs(doubled["Voltage [V]"](t) - default["Voltage [V]"](t))) > (
            1e-3
        )

    def test_half_cell_conserves_lithium(self):
        parameter_values = pybamm.ParameterValues("Xu2019")
        eps_s = parameter_values["Positive electrode active material volume fraction"]
        R = parameter_values["Positive particle radius [m]"]
        parameter_values.update(
            {
                "Positive electrode surface area to volume ratio [m-1]": 2
                * 3
                * eps_s
                / R
            },
            check_already_exists=False,
        )
        model = pybamm.lithium_ion.DFN(
            {"working electrode": "positive", **FROM_PARAMETER}
        )
        solution = solve(model, parameter_values, np.linspace(0, 1000, 51))
        t = solution.t
        lithium = solution["Total lithium in positive electrode [mol]"](t)
        charge = (
            solution["Discharge capacity [A.h]"](t) * 3600 / pybamm.constants.F.value
        )
        np.testing.assert_allclose(lithium - lithium[0], charge, rtol=1e-6, atol=1e-12)
