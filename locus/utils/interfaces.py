from typing import TypedDict


class DescribeFileEntry(TypedDict):
    """Structure of a file entry in the describe JSON file.

    Params:
        name (str): name of the file
        count (int): number of records in the file
        min_index (int): minimum index of record in the file
        max_index (int): maximum index of record in the file
    """

    name: str
    count: int
    min_index: int
    max_index: int


class DescribeJsonStructure(TypedDict):
    """Structure of the describe JSON file.

    Params:
        count (int): number of files
        files (list[DescribeFileEntry]): list of file entries

    See Also:
        DescribeFileEntry
    """

    count: int
    files: list[DescribeFileEntry]
