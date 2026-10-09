"""Physics tests for the DFN-based dual-continuum model.

The closures are pinned against exact limits: DC0 is PyBaMM's uniform-profile
particle, and DC1 with the isolated-sphere closure variable -R/(5 D F) (the
analytical solution of the closure problem for a sphere) is PyBaMM's
quadratic-profile particle. The image-based surface area and the lithium-foil
option are pinned against reference voltages from the standalone
implementation (tests/data/standalone_reference.csv).
"""

from pathlib import Path

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
REFERENCE = Path(__file__).parent / "data" / "standalone_reference.csv"
#: Changes to the half-cell defaults used for the reference voltages.
REFERENCE_CHANGES = {
    f"{AREA} (AM-CBD) [m-1]": 5.0e4,
    f"{AREA} (AM-separator) [m-1]": 2.0e4,
    "CBD surface porosity": 0.3,
    "Separator surface porosity": 0.6,
}


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

    @pytest.mark.parametrize("sei", ["reaction limited", "ec reaction limited"])
    def test_sei_matches_dfn(self, sei):
        # PyBaMM's SEI options apply unchanged to the DC model
        dfn = pybamm.lithium_ion.DFN({"SEI": sei, "particle": "quadratic profile"})
        parameter_values = dfn.default_parameter_values
        solution = solve(dual_continuum({"SEI": sei}, DFN_LIKE), parameter_values)
        np.testing.assert_allclose(
            solution["Voltage [V]"](TIMES), voltage(dfn, parameter_values), rtol=1e-7
        )
        thickness = solution["X-averaged negative SEI thickness [m]"](TIMES)
        assert thickness[-1] > thickness[0]

    @pytest.mark.parametrize("options", CELLS)
    def test_default_parameters_describe_the_dfn_cell(self, options):
        # Defaults are the DFN's geometry and the isolated-sphere closure at
        # the reference state (constant diffusivities here), with no foil scaling
        dfn = pybamm.lithium_ion.DFN({**options, "particle": "quadratic profile"})
        for dc_options in [{}, {"dimensionless closure variable": "true"}]:
            np.testing.assert_allclose(
                voltage(dual_continuum(options, dc_options)),
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

    def test_lithium_foil_surface_porosity(self):
        model = dual_continuum(
            HALF_CELL,
            {
                "closure variable": "isolated sphere",
                "lithium foil surface porosity": "true",
            },
        )
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

    def test_surface_area_enters_only_through_its_total(self):
        model = dual_continuum(HALF_CELL)
        split = model.default_parameter_values
        split.update(REFERENCE_CHANGES)
        total = split.copy()
        total.update(
            {
                f"{AREA} (AM-electrolyte) [m-1]": split[
                    f"{AREA} (AM-electrolyte) [m-1]"
                ]
                + 0.3 * 5.0e4
                + 0.6 * 2.0e4,
                f"{AREA} (AM-CBD) [m-1]": 0.0,
                f"{AREA} (AM-separator) [m-1]": 0.0,
            }
        )
        np.testing.assert_allclose(
            voltage(model, split), voltage(model, total), rtol=1e-9
        )

    @pytest.mark.parametrize(
        ("label", "dc_options"),
        [
            (
                "DC0",
                {
                    "model type": "DC0",
                    "calculate surface concentration a priori": "true",
                },
            ),
            ("DC1", {"model type": "DC1"}),
            (
                "DC1 a priori",
                {
                    "model type": "DC1",
                    "calculate surface concentration a priori": "true",
                },
            ),
        ],
    )
    def test_matches_standalone_reference(self, label, dc_options):
        # The standalone implementation scales the foil exchange current
        model = dual_continuum(
            HALF_CELL, {**dc_options, "lithium foil surface porosity": "true"}
        )
        parameter_values = model.default_parameter_values
        parameter_values.update(REFERENCE_CHANGES)
        reference = read_reference()
        # Tight tolerances so that solver error does not mask model differences
        new = voltage(
            model, parameter_values, pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-8)
        )
        np.testing.assert_allclose(TIMES, reference["Time [s]"])
        np.testing.assert_allclose(new, reference[f"{label} voltage [V]"], atol=1e-5)

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
        with pytest.raises(pybamm.OptionError, match="only applies to half cells"):
            dual_continuum(dc_options={"lithium foil surface porosity": "true"})
        with pytest.raises(pybamm.OptionError, match="single particle phase"):
            dual_continuum({"particle phases": ("2", "1")})


def read_reference():
    """Reference voltages by column name (the last comment line)."""
    with open(REFERENCE) as file:
        comments = [line for line in file if line.startswith("#")]
    names = comments[-1].lstrip("# ").strip().split(",")
    data = np.loadtxt(REFERENCE, delimiter=",")
    return dict(zip(names, data.T, strict=True))
