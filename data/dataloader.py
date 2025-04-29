from torch.utils.data import DataLoader

# Create a DataLoader
def dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_test):
    dataloader_train = DataLoader(tiny_imagenet_dataset_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(tiny_imagenet_dataset_test, batch_size=64, shuffle=True)

    return dataloader_train, dataloader_test