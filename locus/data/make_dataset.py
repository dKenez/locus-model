# imports
import click

import locus.data.LDoGI as ldogi


@click.command()
@click.option("--dataset", default="LDoGI", help="Name of the dataset to process.")
def main(dataset: str):
    """Process a raw dataset into a more convenient format.

    Args:
        dataset (str): Name of the dataset to process.

    Raises:
        ValueError: If dataset is not supported.
    """

    match dataset.lower():
        case "ldogi":
            ldogi.process_raw_data()
        case _:
            raise ValueError(f"Dataset value of '{dataset}' is unsupported!")


if __name__ == "__main__":
    main()
