"""Dual Continuum Model."""

from __future__ import annotations

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo import _compat

SLUG = "dual_continuum"


class DualContinuum(pybamm.BaseModel):
    """Dual continuum model.

    #TODO: finish docstring
    Parameters
    ----------
    options : dict, optional
        Model options...
    name : str, optional
        The model name.
    build : bool, optional
        Whether to build the model on instantiation.

    Examples
    --------
    >>> import pybamm_model_zoo as zoo
    >>> model = zoo.load("DualContinuum")()
    """

    def __init__(
        self,
        options: dict | None = None,
        name: str = "Dual Continuum Model",
        build: bool = True,
    ) -> None:
        super().__init__(name=name)
        # TODO: sort citation
        # pybamm_model_zoo.register_citation(
        #     SLUG, "PyBaMMModelZoo2026", "WeppnerHuggins1977"
        # )

