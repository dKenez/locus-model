import torch
from torch.utils.data import DataLoader

from locus.models.dataset import LDoGIDataset

train_data = LDoGIDataset("quadtree.gml", from_id=1, to_id=1_000)
test_data = LDoGIDataset("quadtree.gml", from_id=1_001, to_id=2_000)


def cf(*args, **kwargs):
    ids = [i[0] for i in args[0]]
    image_tensors = [i[1] for i in args[0]]
    labels = [i[2] for i in args[0]]
    images = torch.cat(image_tensors, dim=0)
    return ids, images, labels


train_dataloader = DataLoader(
    train_data,
    collate_fn=cf,
    shuffle=True,
    batch_size=64,
)
for i, (ids, images, labels) in enumerate(train_dataloader):
    print(ids)
    print(i, images.shape, len(labels))
    # print(i)
