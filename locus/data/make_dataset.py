# imports
import click
from dotenv import dotenv_values

import locus.data.LDoGI as ldogi


@click.command()
@click.option("--delete-existing", is_flag=True, help="Delete the existing data if present.", default=False)
def main(delete_existing: bool):
    """Process a raw dataset into a more convenient format.

    Args:
        delete-existing (bool): Delete the existing data if present.
    """

    config = dotenv_values(".env")

    ldogi.process_raw_data(
        delete_existing=delete_existing,
        db_host=config["DB_HOST"] or "",  # Add type hint to indicate it expects a string
        db_port=config["DB_PORT"] or 0,
        db_name=config["DB_NAME"] or "",
        db_user=config["DB_USER"] or "",
        db_password=(config["DB_PASSWORD"] or ""),
    )


if __name__ == "__main__":
    main()
