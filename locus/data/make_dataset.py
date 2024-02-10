# imports
import click

import locus.data.LDoGI as ldogi


@click.command()
@click.option("--delete-existing", is_flag=True, help="Delete the existing data if present.", default=False)
def main(delete_existing: bool):
    """Process a raw dataset into a more convenient format.

    Args:
        delete-existing (bool): Delete the existing data if present.
    """

    ldogi.process_raw_data(delete_existing=delete_existing)


if __name__ == "__main__":
    main()
