# imports
import click

import src.data.LDoGI as ldogi


@click.command()
@click.option("--dataset", default="LDoGI", help="Number of greetings.")
def main(dataset: str):
    """Process a raw dataset into a more convenient format."""

    match dataset.lower():
        case "ldogi":
            ldogi.process_raw_data()
        case _:
            raise ValueError(f"Dataset value of '{dataset}' is unsupported!")


if __name__ == "__main__":
    main()
