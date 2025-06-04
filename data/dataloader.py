from torch.utils.data import DataLoader

# Create a DataLoader
def dataloader(dataset_train, dataset_val, batch_size, shuffle_train=True, shuffle_val=True, drop_last_bach=False, num_workers=2):
    # drop_last_bach = Ture: if the last bach has NOT batch_size element, just remove it

    if dataset_val is None: # Only to train
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last_bach, num_workers=num_workers, pin_memory=True)
        return dataloader_train, None
    
    elif dataset_train is None: # Only to validate
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val, drop_last=drop_last_bach, num_workers=num_workers, pin_memory=True)
        return None, dataloader_val
    
    else: # To train and validate at the same time
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last_bach, num_workers=num_workers, pin_memory=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val, drop_last=drop_last_bach, num_workers=num_workers, pin_memory=True)
        return dataloader_train, dataloader_val