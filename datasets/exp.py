import os
import pickle

import torch
from ..data import Dataset, Data, Batch


class EXP(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EXP, self).__init__(root, transform, pre_transform, pre_filter)
        self.data_batch = torch.load(self.processed_paths[0])
        self.indices = torch.arange(len(self.data_batch))

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # EXP dataset doesn't provide a download link
        pass

    def process(self):
        data_list = pickle.load(open(os.path.join(self.raw_dir, "GRAPHSAT.pkl"), "rb"))
        data_list = [Data(**g.__dict__) for g in data_list]
        data_list = [Data(x=g.x, edge_index=g.edge_index, y=g.y) for g in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(Batch.from_data_list(data_list), self.processed_paths[0])
