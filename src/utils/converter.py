# Converters for various things

# imports
from src.utils.constants import data_units


def pick_auto_data_size(size: int | float, base: str = "B") -> str:
    """Pick the best unit for a data size.

    Args:
        size (int | float): size of data in base unit
        base (str, optional): Base unit. Defaults to "B".

    Returns:
        str: _description_
    """
    data_size = "B"
    # loop through the units
    for unit in data_units.keys():
        # check if the size is bigger than the unit
        if size * data_units[base] >= data_units[unit] and len(unit) < 3:
            # set the base
            data_size = unit

    return data_size


def convert_data_size(size: int | float, to: str = "auto", base: str = "B") -> float:
    """Convert a data size from one unit into another int a human readable format.

    Args:
        size (int): size of data in base unit
        to (str, optional): Unit of the output. Defaults to "auto".
        base (str, optional): Base unit. Defaults to "B".

    Returns:
        int: size of data in the new unit
    """

    # check if the base is valid
    if base not in data_units.keys():
        raise ValueError(f"Invalid base unit: {base}")

    # if to is auto, detect which unit it is
    if to == "auto":
        to = pick_auto_data_size(size)

    # check if the to is valid
    if to not in data_units.keys():
        raise ValueError(f"Invalid target unit: {to}")

    # convert the size
    return size * data_units[base] / data_units[to]


if __name__ == "__main__":
    # imports
    from src.utils.console import console

    # test if format_data_size works
    print(convert_data_size(1024))
    print(convert_data_size(1024, "MB", "GB"))

    # test if convert_data_size throws an error
    try:
        convert_data_size(1024, "foo")
    except ValueError:
        console.print("[green]ValueError thrown as expected[/green]")
    else:
        console.print("[red]ValueError not thrown[/red]")
