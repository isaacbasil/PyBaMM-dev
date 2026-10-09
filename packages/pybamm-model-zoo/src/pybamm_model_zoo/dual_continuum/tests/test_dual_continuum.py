"""Physics tests for the DFN-based dual-continuum model.

The closures are pinned against exact limits: DC0 is PyBaMM's uniform-profile
particle, and DC1 with the isolated-sphere closure variable -R/(5 D F) (the
analytical solution of the closure problem for a sphere) is PyBaMM's
quadratic-profile particle. The image-based surface area and the lithium-foil
scaling are pinned against the standalone implementation, dc_model_my_scripts.
"""

import numpy as np
import pytest

import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo.dual_continuum import parameter_sets

SOLVE_TIME = 3000
TIMES = np.linspace(0, SOLVE_TIME, 100)
HALF_CELL = {"working electrode": "positive"}
CELLS = [pytest.param({}, id="full cell"), pytest.param(HALF_CELL, id="half cell")]
#: DC options that reproduce the DFN's geometry and need no DC parameters.
DFN_LIKE = {"surface area": "spherical", "closure variable": "isolated sphere"}
AREA = "Positive electrode specific surface area from image"


def dual_continuum(options=None, dc_options=None):
    return zoo.load("DualContinuum")(options, dc_options)


def solve(model, parameter_values=None, solver=None):
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameter_values or model.default_parameter_values,
        solver=solver,
    )
    return simulation.solve([0, SOLVE_TIME], t_interp=TIMES)


def voltage(model, parameter_values=None, solver=None):
    return solve(model, parameter_values, solver)["Voltage [V]"](TIMES)


class TestDualContinuum:
    @pytest.mark.parametrize("options", CELLS)
    def test_dc0_is_uniform_profile(self, options):
        dfn = pybamm.lithium_ion.DFN({**options, "particle": "uniform profile"})
        np.testing.assert_allclose(
            voltage(dual_continuum(options, {**DFN_LIKE, "model type": "DC0"})),
            voltage(dfn),
            rtol=1e-7,
        )

    @pytest.mark.parametrize("options", CELLS)
    def test_isolated_sphere_closure_is_quadratic_profile(self, options):
        dfn = pybamm.lithium_ion.DFN({**options, "particle": "quadratic profile"})
        np.testing.assert_allclose(
            voltage(dual_continuum(options, DFN_LIKE)), voltage(dfn), rtol=1e-7
        )

    def test_default_parameters_describe_the_dfn_cell(self):
        # Full cell: defaults are the DFN's geometry and the isolated-sphere
        # closure at the reference state (constant diffusivities here)
        dfn = pybamm.lithium_ion.DFN({"particle": "quadratic profile"})
        for dc_options in [{}, {"dimensionless closure variable": "true"}]:
            np.testing.assert_allclose(
                voltage(dual_continuum(dc_options=dc_options)),
                voltage(dfn),
                rtol=1e-7,
            )

    def test_surface_area_from_image(self):
        model = dual_continuum(HALF_CELL)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                f"{AREA} (AM-electrolyte) [m-1]": 2.0e5,
                f"{AREA} (AM-CBD) [m-1]": 1.0e5,
                f"{AREA} (AM-separator) [m-1]": 4.0e4,
                "CBD surface porosity": 0.3,
                "Separator surface porosity": 0.4,
            }
        )
        solution = solve(model, parameter_values)
        expected = 2.0e5 + 0.3 * 1.0e5 + 0.4 * 4.0e4
        eps_s = parameter_values["Positive electrode active material volume fraction"]
        np.testing.assert_allclose(
            solution["Positive electrode surface area to volume ratio [m-1]"](
                t=TIMES[-1]
            ),
            expected,
        )
        np.testing.assert_allclose(
            solution["Positive electrode effective particle radius [m]"](t=TIMES[-1]),
            3 * eps_s / expected,
        )

    def test_lithium_foil_scaled_by_separator_surface_porosity(self):
        model = dual_continuum(HALF_CELL, {"closure variable": "isolated sphere"})
        parameter_values = model.default_parameter_values
        porosity = parameter_values["Separator surface porosity"]
        reference = dual_continuum(HALF_CELL, DFN_LIKE)
        reference_values = reference.default_parameter_values
        j0 = reference_values[
            "Exchange-current density for lithium metal electrode [A.m-2]"
        ]
        reference_values[
            "Exchange-current density for lithium metal electrode [A.m-2]"
        ] = lambda c_e, c_Li, T: porosity * j0(c_e, c_Li, T)
        np.testing.assert_allclose(
            voltage(model, parameter_values),
            voltage(reference, reference_values),
            rtol=1e-7,
        )

    @pytest.mark.parametrize("options", CELLS)
    def test_a_priori_close_to_implicit_at_low_rate(self, options):
        a_priori = dual_continuum(
            options, {"calculate surface concentration a priori": "true"}
        )
        np.testing.assert_allclose(
            voltage(a_priori), voltage(dual_continuum(options)), atol=5e-3
        )

    def test_lithium_conservation(self):
        solution = solve(dual_continuum())
        lithium = solution["Total lithium in particles [mol]"](TIMES)
        np.testing.assert_allclose(lithium, lithium[0], rtol=1e-10)

    def test_yang_uses_particle_radius(self):
        yang = dual_continuum(HALF_CELL, {"model type": "Yang"})
        dc1 = dual_continuum(HALF_CELL)
        parameter_values = yang.default_parameter_values
        larger = parameter_values.copy()
        larger["Positive particle radius [m]"] *= 2
        # With image-based areas the radius enters only Yang's closure
        assert (
            np.abs(voltage(yang, larger) - voltage(yang, parameter_values)).max() > 1e-3
        )
        np.testing.assert_allclose(
            voltage(dc1, larger), voltage(dc1, parameter_values), rtol=1e-9
        )

    def test_paten2026_parameter_sets(self):
        dc1 = parameter_sets.paten2026_dc1()
        assert dc1["Positive electrode s0 surface average"] == -500.38
        assert dc1[
            "Negative electrode specific surface area from image (AM-electrolyte) [m-1]"
        ] == pytest.approx(441818.3)
        for model_class, values in [
            (zoo.load("DualContinuum"), dc1),
            (pybamm.lithium_ion.DFN, parameter_sets.paten2026_dfn()),
        ]:
            # Bruggeman coefficients are the exact equivalent of the tortuosities
            np.testing.assert_allclose(
                voltage(model_class(), values),
                voltage(
                    model_class({"transport efficiency": "tortuosity factor"}), values
                ),
                rtol=1e-9,
            )

    # BaseDC still uses the pre-rename "... electrode diffusivity" parameter name
    @pytest.mark.filterwarnings("ignore:The parameter .* has been renamed")
    @pytest.mark.filterwarnings("ignore:Both the deprecated")
    @pytest.mark.parametrize(
        "dc_options",
        [
            {"model type": "DC0", "calculate surface concentration a priori": "true"},
            {"model type": "DC1"},
            {"model type": "DC1", "calculate surface concentration a priori": "true"},
        ],
    )
    def test_matches_standalone_implementation(self, dc_options):
        model = dual_continuum(HALF_CELL, dc_options)
        parameter_values = model.default_parameter_values
        # Exercise every surface term, and the foil scaling
        parameter_values.update(
            {
                f"{AREA} (AM-CBD) [m-1]": 5.0e4,
                f"{AREA} (AM-separator) [m-1]": 2.0e4,
                "CBD surface porosity": 0.3,
                "Separator surface porosity": 0.6,
            }
        )
        # Tight tolerances so that solver error does not mask model differences
        new = voltage(
            model, parameter_values, pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-8)
        )
        reference = solve_standalone(
            dc_options, parameter_values, pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-8)
        )
        np.testing.assert_allclose(new, reference, atol=1e-5)

    def test_options(self):
        with pytest.raises(pybamm.OptionError, match="not recognised"):
            dual_continuum(dc_options={"closure": "DC1"})
        with pytest.raises(pybamm.OptionError, match="Invalid value"):
            dual_continuum(dc_options={"surface area": "components"})
        with pytest.raises(pybamm.OptionError, match="a priori"):
            dual_continuum(
                dc_options={
                    "model type": "Yang",
                    "calculate surface concentration a priori": "true",
                }
            )
        with pytest.raises(pybamm.OptionError, match="single particle phase"):
            dual_continuum({"particle phases": ("2", "1")})


def standalone_parameters(pv):
    """Translate PyBaMM half-cell parameter values into BaseDC's names."""
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
            # BaseDC: negative current density is a discharge
            "Current density [A.m-2]": -pv["Current function [A]"] / area,
            "Temperature [K]": pv["Ambient temperature [K]"],
            "Separator porosity": eps_sep,
            "Separator tortuosity (electrolyte)": eps_sep
            ** (1 - pv[f"Separator {brugg} (electrolyte)"]),
            "Thermodynamic factor": lambda c_e, T: 1.0 + 0 * c_e,
            "Positive electrode active material volume fraction": eps_s,
            "Positive electrode diffusivity [m2.s-1]": pv[
                "Positive particle diffusivity [m2.s-1]"
            ],
            "Positive electrode tortuosity (electrode)": eps_s
            ** (1 - pv[f"Positive electrode {brugg} (electrode)"]),
            "Positive electrode tortuosity (electrolyte)": eps_e
            ** (1 - pv[f"Positive electrode {brugg} (electrolyte)"]),
            "Positive electrode rate constant": 0.0,
            # BaseDC passes c_Li = 1; PyBaMM passes 1 / (Li partial molar volume)
            "Exchange-current density for lithium metal electrode [A.m-2]": (
                lambda c_e, c_Li, T: j0_li(c_e, c_li_metal, T)
            ),
        }
    )
    return pybamm.ParameterValues(values)


def solve_standalone(dc_options, pv, solver, points=20):
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
    solution = solver.solve(model, [0, SOLVE_TIME], t_interp=TIMES)
    return solution["Voltage [V]"](TIMES)
