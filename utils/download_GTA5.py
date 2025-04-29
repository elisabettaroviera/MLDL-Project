# Import necessary libraries
import requests
from zipfile import ZipFile
from io import BytesIO
import gdown

gta5_id = '1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23'  # Google Drive file ID
gta5_url = f'https://drive.google.com/uc?id={gta5_id}' # Construct the download URL

gta5_output = 'gta5.zip' # Name of the output file

gdown.download(gta5_url, gta5_output, quiet=False) # Download the file using gdown

# Open the downloaded bytes and extract them
with ZipFile("gta5.zip", "r") as zip_ref:
    zip_ref.extractall('./datasets')
print('Download of the dataset GTA5 and extraction complete!')