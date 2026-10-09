"""Options specific to the dual-continuum (DC) model."""

from __future__ import annotations

import pybamm


class DCModelOptions(pybamm.FuzzyDict):
    """Dual-continuum options; the first value of each list is the default.

    * "model type" : the surface-concentration definition
        - "DC1": :math:`c_{surf} = c_{vol} + \\langle s \\rangle_A j`
        - "DC0": :math:`c_{surf} = c_{vol}`
        - "Yang": boundary-layer closure of Yang and Tartakovsky
    * "calculate surface concentration a priori" : DC1 only. "false" uses the
      local reaction rate ``j`` (an algebraic equation for c_surf); "true" uses
      the electrode-averaged rate imposed by the applied current (explicit).
    * "closure variable" : DC1 only. "parameter" reads the closure variable
      :math:`\\langle s \\rangle_A` from the parameter set; "isolated sphere"
      uses the analytical closure solution for an isolated sphere,
      :math:`-R_{eff}/(5 D F)`.
    * "dimensionless closure variable" : with "closure variable": "parameter",
      "true" reads :math:`s^*` with :math:`\\langle s \\rangle_A = s^* L/(D F)`.
    * "surface area" : "from image" builds the specific surface area from the
      AM-electrolyte, AM-CBD and AM-separator areas and surface porosities;
      "spherical" uses :math:`3 \\varepsilon_s / R` as in the DFN.
    """

    possible_options = {
        "model type": ["DC1", "DC0", "Yang"],
        "calculate surface concentration a priori": ["false", "true"],
        "closure variable": ["parameter", "isolated sphere"],
        "dimensionless closure variable": ["false", "true"],
        "surface area": ["from image", "spherical"],
    }

    def __init__(self, extra_options=None):
        extra_options = dict(extra_options or {})
        options = {k: v[0] for k, v in self.possible_options.items()}

        for key, value in extra_options.items():
            if key not in self.possible_options:
                raise pybamm.OptionError(
                    f"DC option '{key}' not recognised. Possible options are "
                    f"{list(self.possible_options)}"
                )
            if isinstance(value, bool):
                value = "true" if value else "false"
            if value not in self.possible_options[key]:
                raise pybamm.OptionError(
                    f"Invalid value '{value}' for DC option '{key}'. "
                    f"Possible values are {self.possible_options[key]}"
                )
            options[key] = value

        if (
            options["model type"] == "Yang"
            and options["calculate surface concentration a priori"] == "true"
        ):
            raise pybamm.OptionError(
                "Cannot calculate the surface concentration a priori for Yang's model"
            )

        super().__init__(options)

    @property
    def implicit_surface_concentration(self):
        """Whether c_surf is an extra algebraic unknown."""
        if self["model type"] == "DC0":
            return False
        if self["model type"] == "Yang":
            return True
        return self["calculate surface concentration a priori"] == "false"
