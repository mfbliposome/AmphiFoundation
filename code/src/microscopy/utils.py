
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import os

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


def show_plot(image):
    plt.imshow(image, cmap='gray')

def Image_blur(image, filter_size):

    image_blur = cv2.GaussianBlur(image, (filter_size, filter_size), 0)

    return image_blur

def Image_enhance(image, filter_size):

    image_blur = Image_blur(image, filter_size)
    image_enhance = image - image_blur
    image_enhance[image_enhance < 0] = 0

    return  image_enhance, image_blur

    
def get_template(filename, image, x, y, w, h):
    '''
    Specify the bounding box position, x, y, w, h represents
    the left corner of postion and the width and height of 
    the bounding box.
    e.g.
    x=309
    y=33
    w=55
    h=55
    '''
    template_image = image[y:y+h, x:x+w]
    # show_plot(template_image)
    
    plt.savefig('Results/'+f'{filename}_SelectTemplate_{formatted_datetime}.png')
    plt.close()
    np.save('Templates/'+filename, template_image)

    return template_image

def load_template(filename):

    load_template_array = np.load(filename)

    return load_template_array

def calculate_area(df, filename):
    '''
    calculating vesicles area percent over image area
    '''

    # Calculate the area for each object
    if 'box_length' in df.columns:
        df['area'] = np.pi * (df['box_length'] / 2)**2
    
    # Count the number of rows
    num_rows = len(df)
    
    # Sum up the areas to get the total area for the file
    total_area = df['area'].sum()
    
    # Calculate the area percent
    area_percent = total_area / (1024 * 1024)
    
    # Create a DataFrame containing filename, total area, number of rows, and area percent
    area_df = pd.DataFrame({
        'filename': [filename],
        'num_vesicles': [num_rows],
        'area_vesicles': [total_area],  
        'area_percent': [area_percent]
    })
    
    return area_df

def template_generate(path, x, y, w, h):
    
    file_template1_ori = np.load(path)
    plt.imshow(file_template1_ori, cmap='gray')
    filename = os.path.basename(path)
    x, y, w, h = x, y, w, h
    template1 = get_template(filename, file_template1_ori, x, y, w, h)
    plt.imshow(template1, cmap='gray')
    return template1

def get_dispense_volume(concentrations_df, unique_solutes, allow_zero=False):

    stocks = [[50, 10, 2], [50, 10, 2], [50, 10, 2], [50, 10, 2], [50, 10, 2], [15, 3], [10, 2]]
    valid_dispense_volumes = [[(4, 20), (4, 20), (0, 20)], [(4, 20), (4, 20), (0, 20)], [(4, 20), (4, 20), (0, 20)],
                            [(4, 20), (4, 20), (0, 20)], [(4, 20), (4, 20), (0, 20)], [(4, 20), (0, 20)], [(4, 20), (0, 20)]]

    # Total volume
    total_volume = 200

    # Initialize empty DataFrame to store dispense volumes
    column_names = []
    for i, (col, stock) in enumerate(zip(concentrations_df.columns, stocks), start=1):
        for j, concentration in enumerate(stock, start=1):
            column_names.append(f'{col} ({concentration} mM)')

    dispense_df = pd.DataFrame(columns=column_names)

    # Iterate through each row of concentrations dataframe
    for index, row in concentrations_df.iterrows():
        dispense_volumes = []
        
        # Calculate required volume for each stock solution
        for i, (conc, stock, valid_vol) in enumerate(zip(row, stocks, valid_dispense_volumes), start=1):
            total_mass = conc * total_volume
            volumes = [total_mass / s for s in stock]
            
            # Initialize a list to store volumes for this stock solution
            stock_dispense_volumes = []
            
            # Flag to track if a valid volume has been encountered
            valid_volume_found = False
            
            # Check all possible volumes within the valid dispense volume range
            for volume, vol_range in zip(volumes, valid_vol):
                if not valid_volume_found and vol_range[0] <= volume <= vol_range[1]:
                    stock_dispense_volumes.append(volume)
                    # Set the flag to True to indicate a valid volume has been found
                    valid_volume_found = True
                else:
                    stock_dispense_volumes.append(0)
            
            # Append the volumes for this stock solution to dispense_volumes
            dispense_volumes.extend(stock_dispense_volumes)
            
        # Append dispense volumes for this row to the dataframe
        dispense_df.loc[index] = dispense_volumes
        # Round all numbers in the DataFrame to one decimal place
        dispense_df = dispense_df.round(1)
        # Apply custom function to each element of the DataFrame
        # dispense_df = dispense_df.applymap(discretize)

    # Iterate over the rows and remove those where all columns for the same solute are zero
    if allow_zero:
        return dispense_df
    else:
        rows_to_remove = []
        for index, row in dispense_df.iterrows():
            all_zero = True
            for solute in unique_solutes:
                solute_cols = [col for col in dispense_df.columns if solute in col]
                if all(row[col] == 0 for col in solute_cols):
                    all_zero = all_zero and True
                    rows_to_remove.append(index)
                    # print(index)
                else:
                    all_zero = False

        # Remove the identified rows
        dispense_df = dispense_df.drop(rows_to_remove)
        return dispense_df

def generate_excel(df):
    # Repeat the dataframe to have 96 rows
    df_repeated = pd.concat([df] * 2, ignore_index=True)

    # Create Labware_Deck_Slot column
    df_repeated['Labware_Deck_Slot'] = [1] * 48 + [2] * 48

    # Create Destination_Well column based on the well pattern
    destination_well = []

    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        for i in range(1, 13):
            destination_well.append(f"{letter}{i}")

    df_repeated['Destination_Well'] = destination_well

    # Rearrange columns
    df_repeated = df_repeated[['Labware_Deck_Slot', 'Destination_Well'] + list(df.columns)]

    # Save to Excel with two sheets
    with pd.ExcelWriter('dispense_df_20240429.xlsx') as writer:
        # Write df_repeated to sheet Plate1
        df_repeated.to_excel(writer, sheet_name='Plate1', index=False)
        
        # Write df_repeated to sheet Plate2
        df_repeated.to_excel(writer, sheet_name='Plate2', index=False)

    return df_repeated

# Define a custom sorting function to sort by alphabetical and numerical order
def custom_sort_key(filename):
    letter = filename[0]
    number = int(filename[1:])
    return (letter, number)

# Plot pixel intensity
def plot_histogram(image_path, save_folder):
    '''
    This function is used to plot pixel intensity of images
    '''
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image is read correctly
    if img is None:
        print(f"Error: Unable to read image {image_path}.")
        return
    
    # Flatten the image array to a 1D array
    pixel_values = img.flatten()
    print(pixel_values.max())
    
    # Plot the histogram
    plt.hist(pixel_values, bins=256, color='black', alpha=0.75)
    plt.title(f'Histogram of Pixel Intensities for {os.path.basename(image_path)}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    # Save the plot
    base_filename = os.path.basename(image_path)
    save_path = os.path.join(save_folder, f"{os.path.splitext(base_filename)[0]}_histogram.png")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
    print(f"Histogram saved to {save_path}")

def plot_histograms_for_folder(folder_path, save_folder):
    '''
    This function is to get batch pixel intensity histograms
    '''
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        
        # Check if it is a file (and optionally, if it has a .tiff extension)
        if os.path.isfile(file_path) and file_path.lower().endswith('.tiff'):
            plot_histogram(file_path, save_folder)

def transform_uint8(image_path):
    '''
    Transform the original uint16 image to uint8 image
    '''
    # Read the image as is (uint16)
    image_uint16 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Normalize the image to the range 0-255
    image_normalized = cv2.normalize(image_uint16, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    image_uint8 = image_normalized.astype(np.uint8)

    return image_uint8

# Function to plot horizontal bars for each sample
def plot_samples(path, df, title, color_map):
    num_samples = df.shape[0]
    features = df.columns

    plt.figure(figsize=(15, num_samples * 0.5))

    for i in range(num_samples):
        sample = df.iloc[i, :]
        left = 0
        for feature in features:
            plt.barh(i, sample[feature], left=left, color=color_map[feature], label=feature if i == 0 else "")
            left += sample[feature]
        
    plt.xticks(fontsize=70)
    plt.yticks(fontsize=70)
    plt.xlim(0, 3.2)
    plt.tight_layout()
    plt.savefig(path+f"{title}.pdf", dpi=600, bbox_inches='tight', format='pdf')

# Function to plot horizontal bars for each sample with percentages
def plot_samples_percentage(path, df, title, color_map):
    num_samples = df.shape[0]
    features = df.columns

    plt.figure(figsize=(15, num_samples * 0.5))

    for i in range(num_samples):
        sample = df.iloc[i, :]
        total_concentration = sample.sum()
        left = 0
        for feature in features:
            percentage = (sample[feature] / total_concentration) * 100
            plt.barh(i, percentage, left=left, color=color_map[feature], label=feature if i == 0 else "")
            left += percentage

    # plt.xlabel('Percentage of Total Concentration (%)', fontsize=20)
    # plt.ylabel('Sample Index', fontsize=20)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.title(title)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    plt.tight_layout()
    plt.savefig(path+f"{title}.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()