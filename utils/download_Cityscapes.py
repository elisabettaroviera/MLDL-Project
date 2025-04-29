# Import necessary libraries
import requests
from zipfile import ZipFile
from io import BytesIO
import gdown

# Define the path to the dataset
cityscapes_id = '1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2'  # Google Drive file ID
# cityscapes_path = 'https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing'  
cityscapes_url = f'https://drive.google.com/uc?id={cityscapes_id}'
# output = './datasets/cityscapes.zip'
output = 'cityscapes.zip' # Name of the output file

gdown.download(cityscapes_url, output, quiet=False)

# Send a GET request to the URL
# response = requests.get(cityscapes_path)

# Check if the request was successful
# if response.status_code == 200:
# Open the downloaded bytes and extract them
with ZipFile("cityscapes.zip", "r") as zip_ref:
    zip_ref.extractall('./datasets')
print('Download of the dataset Cityscapes and extraction complete!')