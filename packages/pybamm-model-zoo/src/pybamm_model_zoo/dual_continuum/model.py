"""Dual-continuum (DC) model built on PyBaMM's Doyle-Fuller-Newman model."""

from __future__ import annotations

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo.dual_continuum.dc_active_material import (
    ConstantFromImage,
    ConstantSpherical,
)
from pybamm_model_zoo.dual_continuum.dc_options import DCModelOptions
from pybamm_model_zoo.dual_continuum.dc_particle import DCParticle

SLUG = "dual_continuum"


class DualContinuum(pybamm.lithium_ion.DFN):
    """Dual-continuum model of a lithium-ion cell (Paten et al., 2026).

    Electrolyte, solid-phase conduction, kinetics, thermal and counter-electrode
    physics are those of :class:`pybamm.lithium_ion.DFN`. The pseudo-2D particle
    problem is replaced by a homogenised mass balance for the volume-averaged
    active-material concentration, with the surface concentration given by the
    DC0, DC1 or Yang definition (see :class:`DCParticle`).

    With the default DC options the model reads the closure variable and the
    image-based specific surface areas from the parameter set. To run it on a
    parameter set written for the DFN, use
    ``dc_options={"closure variable": "isolated sphere", "surface area":
    "spherical"}``.

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

    def _rebuild_param(self):
        dc_options = getattr(self, "dc_options", None)
        if dc_options is None or dc_options["lithium foil surface porosity"] == "false":
            super()._rebuild_param()
            return
        if self.options["working electrode"] == "both":
            raise pybamm.OptionError(
                "'lithium foil surface porosity' only applies to half cells"
            )
        self.param = LithiumIonParametersWithFoilPorosity(self.options)

    def set_active_material_submodel(self):
        super().set_active_material_submodel()
        if self.dc_options["surface area"] == "from image":
            submodel = ConstantFromImage
        else:
            submodel = ConstantSpherical
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            if getattr(self.options, domain)["loss of active material"] != "none":
                raise pybamm.OptionError(
                    "The dual-continuum model does not support loss of active material"
                )
            for phase in self.options.phases[domain]:
                self.submodels[f"{domain} {phase} active material"] = submodel(
                    self.param, domain, self.options, phase
                )

    @property
    def default_parameter_values(self) -> pybamm.ParameterValues:
        """The DFN's default parameter set plus the DC parameters.

        Image-based surface areas default to the DFN geometry (3 eps_s / R,
        no AM-CBD or AM-separator contact), and the closure variable to the
        isolated-sphere value -R / (5 D F) at the reference state, so the
        defaults describe the same cell as the DFN defaults.
        """
        values = super().default_parameter_values
        param = pybamm.LithiumIonParameters()
        values.update(
            {
                "CBD surface porosity": 0.5,
                "Separator surface porosity": values["Separator porosity"],
            },
            check_already_exists=False,
        )
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            Domain = domain.capitalize()
            domain_param = param.domain_params[domain]
            phase_param = domain_param.prim
            D_ref = phase_param.D(0.5 * phase_param.c_max, param.T_ref)
            s_sphere = values.evaluate(-phase_param.R_typ / (5 * D_ref * param.F))
            area = f"{Domain} electrode specific surface area from image"
            values.update(
                {
                    f"{Domain} electrode s0 surface average": float(s_sphere),
                    f"{Domain} electrode s0 surface average dimensionless": float(
                        s_sphere * values.evaluate(D_ref * param.F / domain_param.L)
                    ),
                    f"{area} (AM-electrolyte) [m-1]": float(
                        values.evaluate(phase_param.a_typ)
                    ),
                    f"{area} (AM-CBD) [m-1]": 0.0,
                    f"{area} (AM-separator) [m-1]": 0.0,
                    f"{Domain} electrode Yang fitting parameter": 1.0,
                },
                check_already_exists=False,
            )
        return values


class LithiumIonParametersWithFoilPorosity(pybamm.LithiumIonParameters):
    """Lithium-ion parameters with the lithium-foil reaction on the separator.

    The foil reacts only where the separator's pores meet it, so its exchange
    current density is scaled by the separator surface porosity.
    """

    def j0_Li_metal(self, c_e, c_Li, T):
        """Exchange-current density of the lithium foil [A.m-2]."""
        porosity = pybamm.Parameter("Separator surface porosity")
        return porosity * super().j0_Li_metal(c_e, c_Li, T)
