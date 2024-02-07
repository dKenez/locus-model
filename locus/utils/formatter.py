# Formatters for various data types

# imports
from locus.utils.constants import data_units
from locus.utils.converter import convert_data_size, pick_auto_data_size


def format_data_size(size: int | float, to: str = "auto", base: str = "B", *, precision: int = 0) -> str:
    """Format a data size from one unit into another int a human readable format.

    Args:
        size (int): size of data in base unit
        to (str, optional): Unit of the output. Defaults to "auto".
        base (str, optional): Base unit. Defaults to "B".
        precision (int, optional): Precision of the output. Defaults to 0.

    Returns:
        str: _description_
    """

    # if to is auto, detect which unit it is
    if to == "auto":
        to = pick_auto_data_size(size)

    # check if the base is valid
    if base not in data_units.keys():
        raise ValueError(f"Invalid base unit: {base}")

    # check if the to is valid
    if to not in data_units.keys():
        raise ValueError(f"Invalid target unit: {to}")

    # convert the size
    size = convert_data_size(size, to, base)

    # format the size
    return f"{size:.{precision}f} {to}"


if __name__ == "__main__":
    # imports
    from locus.utils.console import console

    # test if format_data_size works
    print(format_data_size(1024))
    print(format_data_size(1024, "MB", "GB"))

    # test if format_data_size throws an error
    try:
        format_data_size(1024, "foo")
    except ValueError:
        console.print("[green]ValueError thrown as expected[/green]")
    else:
        console.print("[red]ValueError not thrown[/red]")
