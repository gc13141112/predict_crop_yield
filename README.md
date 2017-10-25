# predict_crop_yield
Python scripts to download image data from MODIS satellite to Google Drive, then process the images, and predict crop yield using Deep Learning. (Currently suffering from a serious lack of data)

# Files:
1. data_for_image_download.csv : CSV file with householdID, yield (kg/meter squared), lattitude, longitude
2. modis.py : (Requires Google Earth Engine) Data downloads in google drive. Copy to a folder called "image_data"
3. image_to_input.py : Gets pixel values and does dimensionality reduction on the input images.
4. process.py: (Requires Tensor Flow) Performs Machine Learning using Tensor Flow's Deep Learning and tests model. 