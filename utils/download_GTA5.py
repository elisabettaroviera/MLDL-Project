# Import necessary libraries
import requests
from zipfile import ZipFile
from io import BytesIO
import gdown

# Personal Google Drive links for the datasets - CHOOSE one of them
# Link 1 - 1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23
# Link 2 - 1U5LpT_jGqTp2wuDvGHewTOnjbCCjVWE2
# Link 3 - 1eL0EhtP_TYs8QY0WBRkfC665jYiBW3Fl
# Link 4 - 1291JV6dHH0YkT0jI2lgRZXtAXSWB5m90 
# Link 5 - 1T4sLX0HLpI6kqe4V2F2oVAzur5fsp4Uc
# Link 6 - 1dJ8HS9Z6XQtDwI1RBDQm89orz69npVTM
# Link 7 - 1W7lXYYeRl30jDvq01Kt9BlojFXKQglKO
gta5_id ='1T4sLX0HLpI6kqe4V2F2oVAzur5fsp4Uc'
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