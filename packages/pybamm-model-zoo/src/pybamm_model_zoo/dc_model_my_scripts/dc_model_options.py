
import pybamm

class DCModelOptions(pybamm.FuzzyDict):
    possible_options = {
        "cell type": ["Cathode half cell", "Anode half cell", "Full cell"],
        "model type": ["DC1", "DC0", "Yang"],
        "effective properties": ["true", "false"], # TODO: maybe not necessary, could do in a cleaner way? 
        "dimensionless closure variable": ["true", "false"],
        "calculate surface concentration a priori": ["false", "true"],
        "calculate effective properties": ["false", "true"],
        "active material-separator interface": ["none", "positive", "negative", "both"],
        "active material-CBD interface": ["none", "positive", "negative", "both"],
        "transient inputs": ["false", "true"],
        "SEI": ["none", "Schneider2022"],
        "minimum plating overpotential": ["none", "no correction", "with correction"], 
    }

    def __init__(self, extra_options=None):
        extra_options = dict(extra_options or {})
        options = {k: v[0] for k, v in self.possible_options.items()}

        for key, value in extra_options.items():
            if key not in self.possible_options:
                raise pybamm.OptionError(
                    f"Option '{key}' not recognised. Best matches are "
                    f"{self.get_best_matches(key)}"
                )
            # Accept Python bools/ints for convenience, store as strings
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, int):
                value = str(value)
            if value not in self.possible_options[key]:
                raise pybamm.OptionError(
                    f"Invalid value '{value}' for option '{key}'. "
                    f"Possible values are {self.possible_options[key]}"
                )
            options[key] = value

        if options["model type"] == "DC0":
            options["Calculate surface concentration a priori"] = "true" # No correction between surface and average concentration in DC0

        if options["model type"] == "Yang" and options["Calculate surface concentration a priori"] == "true":
            raise pybamm.OptionError(
                "Cannot calculate surface concentration a priori for Yang's model"
            )

        super().__init__(options)