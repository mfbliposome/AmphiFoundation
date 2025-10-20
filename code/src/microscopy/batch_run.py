

import cv2
import pandas as pd
import numpy as np
from Preprocess import Enhance_contrast
from Vesicle_detection import Multi_template_match
import os

def run_batch_files_templates(file_folder, templates, PlateName, 
                    min_scale=0.2, max_scale=1.5, intervals=10, threshold = 0.5, Preprocess=True, sigma=50):
    
    '''
    Process a batch of image files to detect vesicles using template matching and save the results.
    
    Parameters:
    -----------
    file_folder : str
        The path to the folder containing the image files.
    templates : list of numpy.ndarray
        A list of templates to be used for matching vesicles in the images.
    PlateName : str
        The name of the plate, used for saving results.
    min_scale : float, optional, default=0.2
        The minimum scale factor for resizing templates.
    max_scale : float, optional, default=1.5
        The maximum scale factor for resizing templates.
    intervals : int, optional, default=10
        The number of intervals between min_scale and max_scale.
    threshold : float, optional, default=0.5
        The threshold for template matching. Only matches with a value above this threshold are considered.
    Preprocess : bool, optional, default=True
        Whether to preprocess the images before template matching.
    sigma : int, optional, default=50
        The sigma value for the Gaussian filter used in preprocessing.

    Returns:
    --------
    None
        The function saves the results to CSV files and prints status messages.

    Notes:
    ------
    - This function processes each image and detects vesiclesin the specified folder,
      and saves the detection results and summary statistics to CSV files.
    - For each image, a CSV file containing the detection results is saved, as well as a summary CSV file
      containing the file names, the number of vesicles, the total area of detected vesicles 
      and their percentage relative to the image area.
    '''
    
    filenames = []
    vesicle_numbers = []

    # Check if the file_folder exists
    if os.path.exists(file_folder):
        # Get the list of files in the folder
        files = os.listdir(file_folder)
        # Remove '.DS_Store' from the list of files if it exists
        # This file will accidently added when copy and paste files into folder
        if '.DS_Store' in files:
            files.remove('.DS_Store')

        dfs = []
        for file in files:
            filepath = os.path.join(file_folder, file)
            if Preprocess:
                image_analysis, image_ori, filename = Enhance_contrast(filepath, PlateName, image_type='RGB', sigma_size=sigma)
            else:
                image_ori = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                filename = os.path.basename(filepath)
                image_analysis = image_ori

            results, number_vesicles = Multi_template_match(filename, image_ori, image_analysis, templates, PlateName,\
                                    min_scale=min_scale, max_scale=max_scale, intervals=intervals, threshold = threshold)
            
            # Define the column names
            columns1 = ["x_center", "y_center", "box_length", "score"]
            filenames.append(file)
            print(file)
            vesicle_numbers.append(number_vesicles)

            # Create a DataFrame using the data and column names
            df1 = pd.DataFrame(results, columns=columns1)

            # Save the DataFrame to a CSV file
            df1.to_csv(f"../../MicroscopyImage/VesicleDetection/Results_{PlateName}/"+f"{filename}.csv", index=False)
            # Calculate vesicles area here?
            if number_vesicles == 0.:
                total_area = 0.
                area_percent = 0.
                num_rows = len(df1)
                # Create a DataFrame containing filename, total area, number of rows, and area percent
                area_df = pd.DataFrame({
                    'filename': [filename],
                    'num_vesicles': [num_rows],
                    'area_vesicles': [total_area],  
                    'area_percent': [area_percent]
                })
                dfs.append(area_df)
                
            else:
                df1['area'] = np.pi * (df1['box_length'] / 2)**2
                total_area = df1['area'].sum()
                area_percent = total_area / (image_analysis.shape[0] * image_analysis.shape[1])
                num_rows = len(df1)
                area_df = pd.DataFrame({
                    'filename': [filename],
                    'num_vesicles': [num_rows],
                    'area_vesicles': [total_area],  
                    'area_percent': [area_percent]
                })
                dfs.append(area_df)

        df2 = pd.concat(dfs, ignore_index=True)
        df2.to_csv(f"../../MicroscopyImage/VesicleDetection/Results_{PlateName}/"+f"{PlateName}_VesiclesSummary.csv", index=False)
    else:
        print(f"Folder '{file_folder}' does not exist.")
    
