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


class QuadTreeItemParams(TypedDict):
    """Structure of the parameters of a QuadTree item.

    Params:
        tau_min (int): minimum tau value
        tau_max (int): maximum tau value
        shards (list[int]): list of shard IDs used to generate the QuadTree
    """

    tau_min: int
    tau_max: int
    shards: list[int]


class QuadTreeItem(TypedDict):
    """Structure of a QuadTree item.

    Params:
        name (str): name of the QuadTree item
        params (QuadTreeItemParams): parameters used to generate the QuadTree
        excluded_ids (list[int]): list of IDs with no corresponding cells in the QuadTree
    """

    name: str
    params: QuadTreeItemParams
    excluded_ids: list[int]


class QuadTreeManifest(TypedDict):
    """Structure of the QuadTree manifest JSON file.

    Params:
        quadtrees (list[QuadTreeItem]): list of QuadTree items
    """

    quadtrees: list[QuadTreeItem]
