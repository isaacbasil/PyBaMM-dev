"""Options specific to the dual-continuum (DC) model.

Only the options that change the *model equations* live here. Options of the
standalone ``BaseDC`` model that are really parameter choices are handled by
PyBaMM options or by the parameter set instead:

* "cell type"                 -> PyBaMM option ``"working electrode"``
* "effective properties"      -> PyBaMM option ``"transport efficiency"``
* "active material-separator interface" / "active material-CBD interface"
                              -> fold into the surface-area parameter
                                 (``"surface area": "from parameter"``)
"""

import pybamm


class DCModelOptions(pybamm.FuzzyDict):
    #: The first entry of each list is the default.
    possible_options = {
        # Closure for the surface concentration
        "model type": ["DC1", "DC0", "Yang"],
        # DC1 only: "true" uses the electrode-averaged current density
        # (explicit, no extra unknown); "false" uses the local current density
        # (one algebraic equation per mesh cell for c_surf)
        "calculate surface concentration a priori": ["false", "true"],
        # DC1 only: give s0 directly ("false") or as s0* with
        # s0 = s0* * L / (D_s F) ("true")
        "dimensionless closure variable": ["true", "false"],
        # How the specific surface area a [m-1] is obtained:
        #   "spherical"      -> a = 3 eps_s / R  (PyBaMM default)
        #   "from parameter" -> "{Domain} electrode surface area to volume
        #                        ratio [m-1]" (e.g. measured from images)
        "surface area": ["spherical", "from parameter"],
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
