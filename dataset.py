from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    """

    """

    def __init__(self, data_list, is_tensor=False):
        self.data_list = data_list
        self.is_tensor = is_tensor

    def __getitem__(self, index):
        return self.data_list[index]
        # input_ids = self.data_list[index].strip()
        # input_ids = [int(token_id) for token_id in input_ids.split()]
        # return input_ids

    def __len__(self):
        if self.is_tensor:
            return self.data_list.size(0)
        return len(self.data_list)
