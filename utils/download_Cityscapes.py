# Import necessary libraries
import requests
from zipfile import ZipFile
from io import BytesIO

# Define the path to the dataset
cityscapes_path = 'https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing'  

# Send a GET request to the URL
response = requests.get(cityscapes_path)

# Check if the request was successful
# if response.status_code == 200:
# Open the downloaded bytes and extract them
with ZipFile(BytesIO(response.content)) as zip_file:
    zip_file.extractall('./datasets')
print('Download of the dataset Cityscapes and extraction complete!')