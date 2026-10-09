"""Parameter sets of Paten et al. (2026), Table IV, for an LG M50 cell.

Both sets start from :footcite:t:`Chen2020` ("Chen2020"), which provides every
value not listed in Table IV. The electrolyte tortuosities of Table IV are
given as "tortuosity factor (electrolyte)" parameters and, equivalently, as
Bruggeman coefficients, so the sets give the same result with PyBaMM's default
"transport efficiency": "Bruggeman" option and with "tortuosity factor".
"""

from __future__ import annotations

import math

import pybamm

#: Table IV, DFN column.
TABLE_IV_DFN = {
    "Initial concentration in negative electrode [mol.m-3]": 29541.5,
    "Initial concentration in positive electrode [mol.m-3]": 16842.2,
    "Maximum concentration in negative electrode [mol.m-3]": 32694.2,
    "Maximum concentration in positive electrode [mol.m-3]": 62266.0,
    "negative rate constant": 4.908e-7,
    "positive rate constant": 4.414e-6,
    "negative tortuosity (electrolyte)": 2.738,
    "positive tortuosity (electrolyte)": 2.802,
    "Negative particle diffusivity [m2.s-1]": 1.332e-13,
    "Positive particle diffusivity [m2.s-1]": 5.126e-15,
    "Negative particle radius [m]": 6.401e-6,
    "Positive particle radius [m]": 5.266e-6,
}

#: Table IV, DC1 column.
TABLE_IV_DC1 = {
    "Initial concentration in negative electrode [mol.m-3]": 29901.3,
    "Initial concentration in positive electrode [mol.m-3]": 18051.3,
    "Maximum concentration in negative electrode [mol.m-3]": 33304.8,
    "Maximum concentration in positive electrode [mol.m-3]": 57399.3,
    "negative rate constant": 9.480e-7,
    "positive rate constant": 2.053e-6,
    "negative tortuosity (electrolyte)": 2.735,
    "positive tortuosity (electrolyte)": 1.944,
    "negative specific surface area": 441818.3,
    "positive specific surface area": 412729.1,
    "Negative electrode s0 surface average": -335.06,
    "Positive electrode s0 surface average": -500.38,
}

#: Activation energies of the Chen2020 exchange-current densities [J.mol-1].
ACTIVATION_ENERGY = {"negative": 35000, "positive": 17800}


def paten2026_dc1() -> pybamm.ParameterValues:
    """Table IV, DC1 column, on top of Chen2020.

    Table IV's specific surface area is used as the AM-electrolyte area, with
    no AM-CBD or AM-separator contribution.
    """
    table = TABLE_IV_DC1
    values = _with_common_entries(table)
    values.update(
        {
            "Negative electrode s0 surface average": table[
                "Negative electrode s0 surface average"
            ],
            "Positive electrode s0 surface average": table[
                "Positive electrode s0 surface average"
            ],
            "CBD surface porosity": 0.5,
            "Separator surface porosity": values["Separator porosity"],
        },
        check_already_exists=False,
    )
    for domain in ["negative", "positive"]:
        prefix = f"{domain.capitalize()} electrode specific surface area from image"
        values.update(
            {
                f"{prefix} (AM-electrolyte) [m-1]": table[
                    f"{domain} specific surface area"
                ],
                f"{prefix} (AM-CBD) [m-1]": 0.0,
                f"{prefix} (AM-separator) [m-1]": 0.0,
            },
            check_already_exists=False,
        )
    return values


def paten2026_dfn() -> pybamm.ParameterValues:
    """Table IV, DFN column, on top of Chen2020, for comparison with DC1."""
    table = TABLE_IV_DFN
    values = _with_common_entries(table)
    for key in [
        "Negative particle diffusivity [m2.s-1]",
        "Positive particle diffusivity [m2.s-1]",
        "Negative particle radius [m]",
        "Positive particle radius [m]",
    ]:
        values[key] = table[key]
    return values


def _with_common_entries(table):
    """Chen2020 with the Table IV entries shared by both columns."""
    values = pybamm.ParameterValues("Chen2020")
    for key in [
        "Initial concentration in negative electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
        "Maximum concentration in negative electrode [mol.m-3]",
        "Maximum concentration in positive electrode [mol.m-3]",
    ]:
        values[key] = table[key]

    for domain in ["negative", "positive"]:
        Domain = domain.capitalize()
        values[f"{Domain} electrode exchange-current density [A.m-2]"] = (
            _exchange_current_density(
                table[f"{domain} rate constant"], ACTIVATION_ENERGY[domain]
            )
        )
        porosity = values[f"{Domain} electrode porosity"]
        tortuosity = table[f"{domain} tortuosity (electrolyte)"]
        brugg_solid = values[f"{Domain} electrode Bruggeman coefficient (electrode)"]
        values.update(
            {
                # eps**b = eps / tau
                f"{Domain} electrode Bruggeman coefficient (electrolyte)": 1
                - math.log(tortuosity) / math.log(porosity),
                f"{Domain} electrode tortuosity factor (electrolyte)": tortuosity,
                f"{Domain} electrode tortuosity factor (electrode)": values[
                    f"{Domain} electrode active material volume fraction"
                ]
                ** (1 - brugg_solid),
            },
            check_already_exists=False,
        )

    separator_porosity = values["Separator porosity"]
    values.update(
        {
            "Separator tortuosity factor (electrolyte)": separator_porosity
            ** (1 - values["Separator Bruggeman coefficient (electrolyte)"])
        },
        check_already_exists=False,
    )
    return values


def _exchange_current_density(rate_constant, activation_energy):
    """Chen2020's exchange-current density with Table IV's rate constant."""

    def exchange_current_density(c_e, c_s_surf, c_s_max, T):
        arrhenius = pybamm.exp(
            activation_energy / pybamm.constants.R * (1 / 298.15 - 1 / T)
        )
        return (
            rate_constant
            * arrhenius
            * c_e**0.5
            * c_s_surf**0.5
            * (c_s_max - c_s_surf) ** 0.5
        )

    return exchange_current_density
