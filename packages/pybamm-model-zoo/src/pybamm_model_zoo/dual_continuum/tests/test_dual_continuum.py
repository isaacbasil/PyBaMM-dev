"""Physics tests for the DFN-based dual-continuum model.

The DC closures are pinned against PyBaMM's polynomial particle models, which
they reduce to exactly for particular closure coefficients, and against the
standalone ``DCModelMyScripts`` (BaseDC) reference implementation.
"""

import numpy as np
import pytest

import pybamm
import pybamm_model_zoo as zoo

SOLVE_TIME = 3000
TIMES = np.linspace(0, SOLVE_TIME, 100)
HALF_CELL = {"working electrode": "positive"}
CELLS = [pytest.param({}, id="full cell"), pytest.param(HALF_CELL, id="half cell")]


def dual_continuum(options=None, dc_options=None):
    return zoo.load("DualContinuum")(options, dc_options)


def voltage(model, parameter_values, solver=None):
    simulation = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=solver
    )
    return simulation.solve([0, SOLVE_TIME], t_interp=TIMES)["Voltage [V]"](TIMES)


class TestDualContinuum:
    @pytest.mark.parametrize("options", CELLS)
    def test_dc0_is_uniform_profile(self, options):
        parameter_values = dual_continuum(options).default_parameter_values
        dc0 = voltage(dual_continuum(options, {"model type": "DC0"}), parameter_values)
        uniform = voltage(
            pybamm.lithium_ion.DFN({**options, "particle": "uniform profile"}),
            parameter_values,
        )
        np.testing.assert_allclose(dc0, uniform, rtol=1e-7)

    @pytest.mark.parametrize("options", CELLS)
    @pytest.mark.parametrize("dimensionless", ["true", "false"])
    def test_dc1_is_quadratic_profile(self, options, dimensionless):
        # Default s0 = -R / (5 D F), the quadratic-profile closure (constant D)
        model = dual_continuum(
            options,
            {"model type": "DC1", "dimensionless closure variable": dimensionless},
        )
        parameter_values = model.default_parameter_values
        quadratic = voltage(
            pybamm.lithium_ion.DFN({**options, "particle": "quadratic profile"}),
            parameter_values,
        )
        np.testing.assert_allclose(
            voltage(model, parameter_values), quadratic, rtol=1e-7
        )

    @pytest.mark.parametrize("options", CELLS)
    def test_surface_area_parameter_matches_spherical(self, options):
        spherical = dual_continuum(options)
        from_parameter = dual_continuum(options, {"surface area": "from parameter"})
        np.testing.assert_allclose(
            voltage(from_parameter, from_parameter.default_parameter_values),
            voltage(spherical, spherical.default_parameter_values),
            rtol=1e-9,
        )

    def test_surface_area_parameter_is_used(self):
        model = dual_continuum(HALF_CELL, {"surface area": "from parameter"})
        parameter_values = model.default_parameter_values
        reference = voltage(model, parameter_values)
        parameter_values["Positive electrode surface area to volume ratio [m-1]"] *= 2
        assert np.abs(voltage(model, parameter_values) - reference).max() > 1e-3

    @pytest.mark.parametrize("options", CELLS)
    def test_a_priori_close_to_implicit_at_low_rate(self, options):
        model = dual_continuum(options)
        parameter_values = model.default_parameter_values
        a_priori = dual_continuum(
            options, {"calculate surface concentration a priori": "true"}
        )
        np.testing.assert_allclose(
            voltage(a_priori, parameter_values),
            voltage(model, parameter_values),
            atol=5e-3,
        )

    def test_lithium_conservation(self):
        solution = pybamm.Simulation(dual_continuum()).solve(
            [0, SOLVE_TIME], t_interp=TIMES
        )
        lithium = solution["Total lithium in particles [mol]"](TIMES)
        np.testing.assert_allclose(lithium, lithium[0], rtol=1e-10)

    # BaseDC still uses the pre-rename "... electrode diffusivity" parameter name
    @pytest.mark.filterwarnings("ignore:The parameter .* has been renamed")
    @pytest.mark.filterwarnings("ignore:Both the deprecated")
    @pytest.mark.parametrize(
        "dc_options",
        [
            {"model type": "DC0", "calculate surface concentration a priori": "true"},
            {"model type": "DC1", "dimensionless closure variable": "false"},
            {
                "model type": "DC1",
                "dimensionless closure variable": "false",
                "calculate surface concentration a priori": "true",
            },
        ],
    )
    def test_matches_standalone_basedc(self, dc_options):
        parameter_values = dual_continuum(HALF_CELL).default_parameter_values
        # Tight tolerances so that solver error does not mask model differences
        new = voltage(
            dual_continuum(HALF_CELL, dc_options),
            parameter_values,
            pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-8),
        )
        reference = solve_basedc(
            dc_options, parameter_values, pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-8)
        )
        np.testing.assert_allclose(new, reference, atol=1e-5)

    def test_yang_solves(self):
        solution = pybamm.Simulation(
            dual_continuum(HALF_CELL, {"model type": "Yang"})
        ).solve([0, SOLVE_TIME])
        assert np.all(np.isfinite(solution["Voltage [V]"](TIMES)))

    def test_options(self):
        with pytest.raises(pybamm.OptionError, match="not recognised"):
            dual_continuum(dc_options={"closure": "DC1"})
        with pytest.raises(pybamm.OptionError, match="Invalid value"):
            dual_continuum(dc_options={"model type": "DC2"})
        with pytest.raises(pybamm.OptionError, match="a priori"):
            dual_continuum(
                dc_options={
                    "model type": "Yang",
                    "calculate surface concentration a priori": "true",
                }
            )
        with pytest.raises(pybamm.OptionError, match="single particle phase"):
            dual_continuum({"particle phases": ("2", "1")})


def basedc_parameters(pv):
    """Translate PyBaMM half-cell parameter values into BaseDC's names."""
    eps_s = pv["Positive electrode active material volume fraction"]
    eps_e = pv["Positive electrode porosity"]
    eps_sep = pv["Separator porosity"]
    area = pv["Electrode height [m]"] * pv["Electrode width [m]"]
    j0_li = pv["Exchange-current density for lithium metal electrode [A.m-2]"]
    c_li_metal = 1 / pv["Lithium metal partial molar volume [m3.mol-1]"]
    brugg = "Bruggeman coefficient"
    return pybamm.ParameterValues(
        {
            "Ideal gas constant [J.K-1.mol-1]": pybamm.constants.R.value,
            "Faraday constant [C.mol-1]": pybamm.constants.F.value,
            "Electrode height [m]": pv["Electrode height [m]"],
            "Electrode width [m]": pv["Electrode width [m]"],
            # BaseDC: negative current density is a discharge
            "Current density [A.m-2]": -pv["Current function [A]"] / area,
            "Temperature [K]": pv["Ambient temperature [K]"],
            "Separator porosity": eps_sep,
            "Separator surface porosity": 1.0,
            "Separator tortuosity (electrolyte)": eps_sep
            ** (1 - pv[f"Separator {brugg} (electrolyte)"]),
            "Initial concentration in electrolyte [mol.m-3]": pv[
                "Initial concentration in electrolyte [mol.m-3]"
            ],
            "Thermodynamic factor": lambda c_e, T: 1.0 + 0 * c_e,
            "Electrolyte diffusivity [m2.s-1]": pv["Electrolyte diffusivity [m2.s-1]"],
            "Electrolyte conductivity [S.m-1]": pv["Electrolyte conductivity [S.m-1]"],
            "Cation transference number": pv["Cation transference number"],
            "Positive electrode active material volume fraction": eps_s,
            "Positive electrode porosity": eps_e,
            "Maximum concentration in positive electrode [mol.m-3]": pv[
                "Maximum concentration in positive electrode [mol.m-3]"
            ],
            "Initial concentration in positive electrode [mol.m-3]": pv[
                "Initial concentration in positive electrode [mol.m-3]"
            ],
            "Positive electrode diffusivity [m2.s-1]": pv[
                "Positive particle diffusivity [m2.s-1]"
            ],
            "Separator thickness [m]": pv["Separator thickness [m]"],
            "Positive electrode thickness [m]": pv["Positive electrode thickness [m]"],
            "Positive electrode conductivity [S.m-1]": pv[
                "Positive electrode conductivity [S.m-1]"
            ],
            "Positive electrode tortuosity (electrode)": eps_s
            ** (1 - pv[f"Positive electrode {brugg} (electrode)"]),
            "Positive electrode tortuosity (electrolyte)": eps_e
            ** (1 - pv[f"Positive electrode {brugg} (electrolyte)"]),
            "Positive electrode specific surface area from image (AM-electrolyte) "
            "[m-1]": 3 * eps_s / pv["Positive particle radius [m]"],
            "Positive electrode s0 surface average": pv[
                "Positive electrode s0 surface average"
            ],
            "Positive electrode rate constant": 0.0,
            "Positive electrode OCP [V]": pv["Positive electrode OCP [V]"],
            "Positive electrode exchange-current density [A.m-2]": pv[
                "Positive electrode exchange-current density [A.m-2]"
            ],
            # BaseDC passes c_Li = 1; PyBaMM passes 1 / (Li partial molar volume)
            "Exchange-current density for lithium metal electrode [A.m-2]": (
                lambda c_e, c_Li, T: j0_li(c_e, c_li_metal, T)
            ),
        }
    )


def solve_basedc(dc_options, pv, solver, points=20):
    options = {
        "cell type": "Cathode half cell",
        "effective properties": "false",
        **dc_options,
    }
    model = zoo.load("DCModelMyScripts")(dc_options=options)
    geometry = {
        "separator": {model.x_s: {"min": pybamm.Scalar(0), "max": model.L_s}},
        "positive electrode": {
            model.x_p: {"min": model.L_s, "max": model.L_s + model.L_p}
        },
    }
    parameters = basedc_parameters(pv)
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
    solution = solver.solve(model, [0, SOLVE_TIME], t_interp=TIMES)
    return solution["Voltage [V]"](TIMES)
