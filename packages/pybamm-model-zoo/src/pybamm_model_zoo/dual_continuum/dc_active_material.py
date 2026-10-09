"""Constant active material with a prescribed specific surface area."""

from __future__ import annotations

import pybamm


class ConstantWithSurfaceArea(pybamm.active_material.Constant):
    """Constant active material whose surface area to volume ratio is a parameter.

    Replaces :math:`a = 3\\varepsilon_s/R` by the function parameter
    "{Domain} electrode surface area to volume ratio [m-1]" (of x), e.g.
    measured on tomography images. The particle radius is kept for closures
    that need a length scale.
    """

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        variables = super().get_fundamental_variables()

        x = pybamm.SpatialVariable(
            f"x_{domain[0]}",
            domain=[f"{domain} electrode"],
            auxiliary_domains={"secondary": "current collector"},
            coord_sys="cartesian",
        )
        a = pybamm.FunctionParameter(
            f"{self.phase_param.phase_prefactor}{Domain} electrode "
            "surface area to volume ratio [m-1]",
            {"Through-cell distance (x) [m]": x},
        )
        a.print_name = f"a_{domain[0]}"
        variables.update(
            {
                f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]": a,
                f"X-averaged {domain} electrode {phase_name}"
                "surface area to volume ratio [m-1]": pybamm.x_average(a),
            }
        )
        return variables
