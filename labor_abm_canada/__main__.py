import json
from argparse import ArgumentParser
from pathlib import Path

from labor_abm_canada.runner import run_model

FILE_PATH = Path(__file__).parent
PROJECT_PATH = FILE_PATH.parent
DATA_PATH = PROJECT_PATH / "data"

GDP_SEASONAL_BAROMETER = DATA_PATH / "u_v_gdp_seasonal_barometer.csv"
DEFAULT_MODEL_PARAMS = FILE_PATH / "model-params.yaml"


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default=None,
        help="Path to the scenario file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./labour_model_results.json",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="National",
        help="Region to run the model for.",
    )

    parser.add_argument(
        "--parameters",
        type=str,
        default=str(DEFAULT_MODEL_PARAMS),
        help="Model parameters file.",
    )

    parser.add_argument("--aggregate", action="store_true", help="If flagged, returns aggregate data")

    return parser


if __name__ == "__main__":

    # Create the parser
    parser = setup_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Run the model
    results = run_model(scenario_filename=args.scenario, region=args.region, model_parameters=args.parameters)

    # Save the results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
