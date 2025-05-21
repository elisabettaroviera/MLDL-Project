from torch.utils.data import DataLoader

# Create a DataLoader
def dataloader(dataset_train, dataset_val, batch_size, shuffle_train=True, shuffle_val=True, drop_last_bach=False):
    if dataset_val is None:
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last_bach)
        return dataloader_train, None
    elif dataset_train is None:
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val, drop_last=drop_last_bach)
        return None, dataloader_val
    else:
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last_bach)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val, drop_last=drop_last_bach)
        return dataloader_train, dataloader_val