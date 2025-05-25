# Import necessary libraries
import requests
from zipfile import ZipFile
from io import BytesIO
import gdown
#gta5_id = '1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23'  # Google Drive file ID
#gta5_id = '1U5LpT_jGqTp2wuDvGHewTOnjbCCjVWE2' #link auro
#gta5_id = '1eL0EhtP_TYs8QY0WBRkfC665jYiBW3Fl' #link luci
#gta5_id = '1291JV6dHH0YkT0jI2lgRZXtAXSWB5m90' #new link luci
#gta5_id =  '1T4sLX0HLpI6kqe4V2F2oVAzur5fsp4Uc' #newwwissimo id 
gta5_id = '1dJ8HS9Z6XQtDwI1RBDQm89orz69npVTM'
#gta5_id = '1QcNgt8xsDpU0lsjiJjCkZK_N9WuOytD8'
gta5_url = f'https://drive.google.com/uc?id={gta5_id}' # Construct the download URL
"""

Personal Google Drive links for the datasets:
gta5_id = '1W7lXYYeRl30jDvq01Kt9BlojFXKQglKO'  # Google Drive file ID

"""

gta5_output = 'gta5.zip' # Name of the output file

gdown.download(gta5_url, gta5_output, quiet=False) # Download the file using gdown

# Open the downloaded bytes and extract them
with ZipFile("gta5.zip", "r") as zip_ref:
    zip_ref.extractall('./datasets')
print('Download of the dataset GTA5 and extraction complete!')