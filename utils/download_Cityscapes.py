# Import necessary libraries
import requests
from zipfile import ZipFile
from io import BytesIO
import gdown
"""
cityscapes_id = '1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2'  # Google Drive file ID
cityscapes_url = f'https://drive.google.com/uc?id={cityscapes_id}' # Construct the download URL"""



#Personal Google Drive links for the datasets:
cityscapes_id = '10XPV2_VJ0kqSPhEHyleYwsalVdVSl1uI'  # Google Drive file ID
# 10XPV2_VJ0kqSPhEHyleYwsalVdVSl1uI
# 11JDxO0DX7MV5DVB225DuIW7vXP-tZGmb
# 1HmmcWizKfXW8h7VzNzZ4c2cTkiMojR5J

cityscapes_url = f'https://drive.google.com/uc?id={cityscapes_id}' # Construct the download URL


cityscapes_output = 'cityscapes.zip' # Name of the output file

gdown.download(cityscapes_url, cityscapes_output, quiet=False) # Download the file using gdown

# Open the downloaded bytes and extract them
with ZipFile("cityscapes.zip", "r") as zip_ref:
    zip_ref.extractall('./datasets')
print('Download of the dataset Cityscapes and extraction complete!')