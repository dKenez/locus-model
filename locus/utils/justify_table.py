def justify_table(data: list[str], widths: list[int]) -> str:
    return "".join(f"{data[i].center(widths[i])}" for i in range(len(data)))
