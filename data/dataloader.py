from torch.utils.data import DataLoader
import os

def dataloader(dataset_train, dataset_val, batch_size, shuffle_train=True, shuffle_val=True, drop_last_bach=False):
    num_workers = min(8, os.cpu_count() // 2)  # Tieni 4-8 worker al massimo

    kwargs = {
        "batch_size": batch_size,
        "drop_last": drop_last_bach,
        "pin_memory": True,
        "persistent_workers": True,
        "num_workers": num_workers,
    }

    if dataset_val is None:  # Only to train
        dataloader_train = DataLoader(dataset_train, shuffle=shuffle_train, **kwargs)
        return dataloader_train, None
    
    elif dataset_train is None:  # Only to validate
        dataloader_val = DataLoader(dataset_val, shuffle=shuffle_val, **kwargs)
        return None, dataloader_val
    
    else:  # To train and validate at the same time
        dataloader_train = DataLoader(dataset_train, shuffle=shuffle_train, **kwargs)
        dataloader_val = DataLoader(dataset_val, shuffle=shuffle_val, **kwargs)
        return dataloader_train, dataloader_val
    

    """| Parametro            | Cosa fa                                             |
| -------------------- | --------------------------------------------------- |
| `num_workers`        | Carica i dati in parallelo con più processi         |
| `persistent_workers` | Tiene vivi i worker tra le epoche → meno overhead   |
| `pin_memory`         | Sposta i dati in memoria pinned (più veloce su GPU) |

"""


"""

# Create a DataLoader
def dataloader(dataset_train, dataset_val, batch_size, shuffle_train=True, shuffle_val=True, drop_last_bach=False):
    # drop_last_bach = Ture: if the last bach has NOT batch_size element, just remove it

    if dataset_val is None: # Only to train
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last_bach)
        return dataloader_train, None
    
    elif dataset_train is None: # Only to validate
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val, drop_last=drop_last_bach)
        return None, dataloader_val
    
    else: # To train and validate at the same time
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last_bach)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle_val, drop_last=drop_last_bach)
        return dataloader_train, dataloader_val"""