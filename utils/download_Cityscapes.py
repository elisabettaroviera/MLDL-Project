import requests
from zipfile import ZipFile
from io import BytesIO
import gdown

# Personal Google Drive links for the datasets - CHOOSE one of them
# Link 1 - 1HmmcWizKfXW8h7VzNzZ4c2cTkiMojR5J
# Link 2 - 1qKP1oqImbrMfDI9R9dJpnlLYn_hgS8Ht
# Link 3 - 10XPV2_VJ0kqSPhEHyleYwsalVdVSl1uI
# Link 4 - 11JDxO0DX7MV5DVB225DuIW7vXP-tZGmb
# Link 5 - 1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2

cityscapes_id = '1HmmcWizKfXW8h7VzNzZ4c2cTkiMojR5J'  # Google Drive file ID
cityscapes_url = f'https://drive.google.com/uc?id={cityscapes_id}' # Construct the download URL
cityscapes_output = 'cityscapes.zip' # Name of the output file

gdown.download(cityscapes_url, cityscapes_output, quiet=False) # Download the file using gdown

# Open the downloaded bytes and extract them
with ZipFile("cityscapes.zip", "r") as zip_ref:
    zip_ref.extractall('./datasets')
print('Download of the dataset Cityscapes and extraction complete!')
