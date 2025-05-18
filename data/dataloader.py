# We have to do something else?
# We have to move to datasets folder?

from torch.utils.data import DataLoader

# Create a DataLoader
def dataloader(dataset_train, dataset_val, batch_size, shuffle_train=True, shuffle_val=True):
    if dataset_val is None:
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train)
        return dataloader_train, None
    elif dataset_train is None:
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val)
        return dataloader_train, dataloader_val
    else:
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val)
        return dataloader_train, dataloader_val