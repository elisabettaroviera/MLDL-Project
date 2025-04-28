from torch.utils.data import Dataset

# TODO: implement here your custom dataset class for Cityscapes

# FIRST PROBLEM: What we have to do HERE?? 
# 1. Upload the dataset?
# 2. Modify it in order to use it?

"""
Method       | What it does                                               | When it is called
__init__     | Prepares the dataset (paths, file lists, transformations)  | Once, when the dataset object is created
__getitem__  | Returns the image and corresponding mask at a given index  | Every time the DataLoader requests a batch
__len__      | Returns the total number of samples in the dataset         | At the beginning and during epoch creation

"""


class CityScapes(Dataset):
    def __init__(self):
        super(CityScapes, self).__init__()
        # TODO

        pass

    def __getitem__(self, idx):
        # TODO

        pass

    def __len__(self):
        # TODO

        pass
