"""Dual-continuum particle submodel."""

from __future__ import annotations

import pybamm


class DCParticle(pybamm.particle.BaseParticle):
    """Molar conservation in the active material for the dual-continuum model.

    The active material is described by its volume-averaged concentration,
    which obeys

    .. math::
        \\frac{\\partial \\bar{c}_s}{\\partial t} = -\\frac{a j}{\\varepsilon_s F},

    and the surface concentration seen by the kinetics is given by a closure:

    * ``DC0``: :math:`c_{surf} = \\bar{c}_s`;
    * ``DC1``: :math:`c_{surf} = \\bar{c}_s + s_0 j` (local ``j``, implicit) or
      :math:`c_{surf} = \\bar{c}_s + s_0 \\bar{j}` (electrode-averaged ``j``,
      explicit, "calculate surface concentration a priori");
    * ``Yang``: diffusion-length correction with a time-dependent length.

    A quadratic radial profile matching the average and surface concentrations
    is reconstructed for output variables only.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel.
    domain : str
        Either "negative" or "positive".
    options : dict
        PyBaMM model options.
    dc_options : :class:`DCModelOptions`
        Dual-continuum options.
    phase : str, optional
        Phase of the particle (default is "primary").
    """

    def __init__(self, param, domain, options, dc_options, phase="primary"):
        super().__init__(param, domain, options, phase)
        if self.size_distribution:
            raise pybamm.OptionError(
                "The dual-continuum particle does not support particle-size "
                "distributions"
            )
        self.dc_options = dc_options

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        c_max = self.phase_param.c_max

        c_s_rav = pybamm.Variable(
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, c_max),
            scale=c_max,
        )
        c_s_rav.print_name = f"c_s_{domain[0]}_rav"
        variables = {
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]": c_s_rav
        }

        if self.dc_options.implicit_surface_concentration:
            c_s_surf = pybamm.Variable(
                f"{Domain} {phase_name}particle surface concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, c_max),
                scale=c_max,
            )
            c_s_surf.print_name = f"c_s_{domain[0]}_surf"
            variables.update(self._get_profile_variables(c_s_rav, c_s_surf))

        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        if not self.dc_options.implicit_surface_concentration:
            if self.dc_options["model type"] == "DC0":
                c_s_surf = c_s_rav
            else:
                j_av = self._electrode_averaged_current_density(variables)
                c_s_surf = c_s_rav + self._closure_coefficient(variables) * j_av
            variables.update(self._get_profile_variables(c_s_rav, c_s_surf))

        c_s = variables[f"{Domain} {phase_name}particle concentration [mol.m-3]"]
        c_s_surf = variables[
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
        ]
        T = pybamm.PrimaryBroadcast(
            variables[f"{Domain} electrode temperature [K]"],
            [f"{domain} {phase_name}particle"],
        )
        current = variables["Total current density [A.m-2]"]
        D_eff = self._get_effective_diffusivity(c_s, T, current)
        R = variables[f"{Domain} {phase_name}particle radius [m]"]
        r = self._radial_variable()

        # Flux of the reconstructed quadratic profile (output only)
        N_s = -D_eff * 5 * (c_s_surf - c_s_rav) * r / R**2

        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_standard_flux_variables(N_s))
        variables.update(
            {
                f"{Domain} electrode DC correction term [mol.m-3]": c_s_surf - c_s_rav,
            }
        )
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        j = variables[
            f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
        ]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]
        self.rhs = {c_s_rav: -a * j / (eps_s * self.param.F)}

    def set_algebraic(self, variables):
        if not self.dc_options.implicit_surface_concentration:
            return

        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        c_s_surf = variables[
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
        ]
        j = variables[
            f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
        ]

        if self.dc_options["model type"] == "DC1":
            correction = self._closure_coefficient(variables) * j
        else:
            correction = self._yang_correction(variables, j)

        self.algebraic = {
            c_s_surf: (c_s_surf - c_s_rav - correction) / self.phase_param.c_max
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_init = pybamm.r_average(self.phase_param.c_init)
        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        self.initial_conditions = {c_s_rav: c_init}

        if self.dc_options.implicit_surface_concentration:
            # Initial guess for the algebraic solver
            c_s_surf = variables[
                f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
            ]
            self.initial_conditions[c_s_surf] = c_init

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _radial_variable(self):
        domain = self.domain
        return pybamm.SpatialVariable(
            f"r_{domain[0]}",
            domain=[f"{domain} {self.phase_name}particle"],
            auxiliary_domains={
                "secondary": f"{domain} electrode",
                "tertiary": "current collector",
            },
            coord_sys="spherical polar",
        )

    def _get_profile_variables(self, c_s_rav, c_s_surf):
        """Standard concentration variables from a quadratic profile in r."""
        domain = self.domain
        particle = [f"{domain} {self.phase_name}particle"]
        R = self.phase_param.R
        r = self._radial_variable()

        A = pybamm.PrimaryBroadcast(5 / 2 * c_s_rav - 3 / 2 * c_s_surf, particle)
        B = pybamm.PrimaryBroadcast(5 / 2 * (c_s_surf - c_s_rav), particle)
        c_s = A + B * r**2 / R**2

        return self._get_standard_concentration_variables(
            c_s, c_s_rav=c_s_rav, c_s_surf=c_s_surf
        )

    def _electrode_averaged_current_density(self, variables):
        """Average interfacial current density imposed by the applied current."""
        domain = self.domain
        a_av = variables[
            f"X-averaged {domain} electrode {self.phase_name}"
            "surface area to volume ratio [m-1]"
        ]
        i_cell = variables["Current collector current density [A.m-2]"]
        L = self.domain_param.L
        # Discharge (i_cell > 0) delithiates the negative, lithiates the positive
        sign = 1 if domain == "negative" else -1
        j_av = sign * i_cell / (a_av * L)
        return pybamm.PrimaryBroadcast(j_av, f"{domain} electrode")

    def _closure_coefficient(self, variables):
        """DC1 closure coefficient s0 [mol.A-1.m-1]."""
        domain, Domain = self.domain_Domain
        if self.dc_options["dimensionless closure variable"] == "true":
            s0_star = pybamm.Parameter(
                f"{Domain} electrode s0 surface average dimensionless"
            )
            c_s_rav = variables[
                f"R-averaged {domain} {self.phase_name}particle concentration [mol.m-3]"
            ]
            T = variables[f"{Domain} electrode temperature [K]"]
            D = self.phase_param.D(c_s_rav, T)
            return s0_star * self.domain_param.L / (D * self.param.F)
        return pybamm.Parameter(f"{Domain} electrode s0 surface average")

    def _yang_correction(self, variables, j):
        """Surface correction of Yang et al. with a growing diffusion length."""
        domain, Domain = self.domain_Domain
        c_s_rav = variables[
            f"R-averaged {domain} {self.phase_name}particle concentration [mol.m-3]"
        ]
        T = variables[f"{Domain} electrode temperature [K]"]
        D = self.phase_param.D(c_s_rav, T)
        l_s = variables[f"{Domain} {self.phase_name}particle radius [m]"]
        a_d = pybamm.Parameter(f"{Domain} electrode Yang fitting parameter")

        t_cut = l_s**2 / (D * a_d**2)
        # Offset avoids the sqrt singularity at t = 0
        l_dif_short = a_d * (D * (pybamm.t + 1e-200)) ** 0.5
        regime = pybamm.t < t_cut
        l_dif = l_dif_short * regime + l_s * (1 - regime)

        return -(l_dif / (2 * l_s) - l_dif**2 / (6 * l_s**2)) * (
            j * l_s / (D * self.param.F)
        )
