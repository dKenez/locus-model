from typing import Literal

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler


class LDoGIDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        *,
        batch_size=1,
        shuffle=True,
        fetch_mode: Literal["individual", "batched"] = "batched",
        generator=None,
        drop_last=False,
        **kwargs,
    ):
        _sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)

        if fetch_mode == "individual":

            def cf(*mini_batches):
                ids_out = np.concatenate([i[0] for i in mini_batches[0]])
                images_out = torch.cat([i[1] for i in mini_batches[0]])
                labels_out = torch.cat([i[2] for i in mini_batches[0]])
                label_names_out = np.concatenate([i[3] for i in mini_batches[0]])

                return ids_out, images_out, labels_out, label_names_out

            super(LDoGIDataLoader, self).__init__(
                dataset=dataset, batch_size=batch_size, sampler=_sampler, collate_fn=cf, **kwargs
            )

        elif fetch_mode == "batched":

            def cf(*mini_batches):
                return mini_batches[0][0]

            super(LDoGIDataLoader, self).__init__(
                dataset=dataset,
                sampler=BatchSampler(_sampler, batch_size=batch_size, drop_last=drop_last),
                collate_fn=cf,
                **kwargs,
            )

        else:
            raise ValueError(f"Fetch mode {fetch_mode} not recognized")
