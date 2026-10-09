"""Dual-continuum (DC) model built on PyBaMM's Doyle-Fuller-Newman model."""

from __future__ import annotations

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo.dual_continuum.dc_active_material import (
    ConstantWithSurfaceArea,
)
from pybamm_model_zoo.dual_continuum.dc_options import DCModelOptions
from pybamm_model_zoo.dual_continuum.dc_particle import DCParticle

SLUG = "dual_continuum"


class DualContinuum(pybamm.lithium_ion.DFN):
    """Dual-continuum model of a lithium-ion cell.

    Electrolyte, solid-phase conduction, kinetics, thermal and counter-electrode
    physics are those of :class:`pybamm.lithium_ion.DFN`. The pseudo-2D particle
    problem is replaced by an upscaled mass balance for the volume-averaged
    active-material concentration plus a closure for the surface concentration
    (see :class:`DCParticle`).

    Parameters
    ----------
    options : dict, optional
        PyBaMM model options, as for :class:`pybamm.lithium_ion.DFN`. Use
        ``{"working electrode": "positive"}`` for a cathode half cell.
    dc_options : dict, optional
        Dual-continuum options, see :class:`DCModelOptions`.
    name : str, optional
        The model name.
    build : bool, optional
        Whether to build the model on instantiation.

    Examples
    --------
    >>> import pybamm_model_zoo as zoo
    >>> model = zoo.load("DualContinuum")(dc_options={"model type": "DC1"})
    """

    def __init__(
        self,
        options: dict | None = None,
        dc_options: dict | None = None,
        name: str = "Dual-continuum model",
        build: bool = True,
    ) -> None:
        self.dc_options = DCModelOptions(dc_options)
        super().__init__(options=options, name=name, build=build)
        pybamm_model_zoo.register_citation(SLUG, "PyBaMMModelZoo2026", "Paten2026")

    def set_particle_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            phases = self.options.phases[domain]
            if len(phases) > 1:
                raise pybamm.OptionError(
                    "The dual-continuum model supports a single particle phase"
                )
            for phase in phases:
                self.submodels[f"{domain} {phase} particle"] = DCParticle(
                    self.param, domain, self.options, self.dc_options, phase
                )
                self.submodels[f"{domain} {phase} total particle concentration"] = (
                    pybamm.particle.TotalConcentration(
                        self.param, domain, self.options, phase
                    )
                )

    def set_active_material_submodel(self):
        super().set_active_material_submodel()
        if self.dc_options["surface area"] == "spherical":
            return
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            if getattr(self.options, domain)["loss of active material"] != "none":
                raise pybamm.OptionError(
                    "'surface area': 'from parameter' is not compatible with "
                    "loss of active material"
                )
            for phase in self.options.phases[domain]:
                self.submodels[f"{domain} {phase} active material"] = (
                    ConstantWithSurfaceArea(self.param, domain, self.options, phase)
                )

    @property
    def default_parameter_values(self) -> pybamm.ParameterValues:
        """DFN defaults plus a DC1 closure equivalent to a quadratic profile."""
        values = super().default_parameter_values
        param = pybamm.LithiumIonParameters()
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            Domain = domain.capitalize()
            domain_param = param.domain_params[domain]
            phase_param = domain_param.prim
            c_ref = pybamm.Scalar(0.5) * phase_param.c_max
            D_ref = phase_param.D(c_ref, param.T_ref)
            # s0 = -R/(5 D F) reproduces the quadratic-profile closure
            s0 = values.evaluate(-phase_param.R_typ / (5 * D_ref * param.F))
            s0_star = values.evaluate(-phase_param.R_typ / (5 * domain_param.L))
            values.update(
                {
                    f"{Domain} electrode s0 surface average": float(s0),
                    f"{Domain} electrode s0 surface average dimensionless": float(
                        s0_star
                    ),
                    f"{Domain} electrode Yang fitting parameter": 2.0,
                },
                check_already_exists=False,
            )
            if self.dc_options["surface area"] == "from parameter":
                eps_s = values[f"{Domain} electrode active material volume fraction"]
                R = values[f"{Domain} particle radius [m]"]
                values.update(
                    {
                        f"{Domain} electrode surface area to volume ratio [m-1]": (
                            3 * eps_s / R
                        )
                    },
                    check_already_exists=False,
                )
        return values
