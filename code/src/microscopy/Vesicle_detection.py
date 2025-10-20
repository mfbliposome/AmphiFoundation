import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.patches as patches
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H")

def Multi_template_match(filename, original_image, image, templates, PlateName, \
                         min_scale=0.5, max_scale=1.2, intervals=10, threshold = 0.6):
    '''
    Perform multi-template multi-scale matching on a microscopy image to detect and locate vesicles.

    Parameters:
    -----------
    filename : str
        The name of the image file.
    original_image : numpy.ndarray
        The original image to be used for visualization of results.
    image : numpy.ndarray
        The image on which vesicle detection is to be performed.
    templates : list of numpy.ndarray
        A list of templates to be matched against the image.
    PlateName : str
        The name of the plate, used for saving results.
    min_scale : float, optional, default=0.5
        The minimum scale factor for resizing templates.
    max_scale : float, optional, default=1.2
        The maximum scale factor for resizing templates.
    intervals : int, optional, default=10
        The number of intervals between min_scale and max_scale.
    threshold : float, optional, default=0.6
        The threshold for template matching. Only matches with a value above this threshold are considered.

    Returns:
    --------
    match_results : numpy.ndarray or None
        A 2D array where each row corresponds to a detected object. Each row contains:
        [x_center, y_center, detecting_box_length, match_score].
        If no objects are detected, returns None.
    len_match_results : int
        The number of detected objects. Returns 0 if no objects are detected.

    Notes:
    ------
    - This function uses multi-scale template matching to detect objects of varying sizes.
    - Overlapping bounding boxes are removed based on the match score, keeping only the highest value matches.
    - The results are visualized and saved as an image file.
    
    '''
    # Generate a linear space of scales
    scales = np.linspace(min_scale, max_scale, intervals)
    scales = np.round(scales, decimals=2)
    h_o, w_o = image.shape[0], image.shape[1]

    x_center=[]
    y_center=[]
    scale_select=[]
    match_value=[]

    # Iterate different templates
    for template in templates:
        for scale in scales:
            template_scale = cv2.resize(template, None, fx=scale, fy=scale)
            h, w = template_scale.shape[0], template_scale.shape[1]
            if w >= w_o or h >= h_o:
                break

            result = cv2.matchTemplate(image, template_scale, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)

            x_center.extend(loc[1] + w/2)
            y_center.extend(loc[0] + h/2)
            scale_select.extend(w + 0*loc[0])
            match_value.extend(result[loc[0], loc[1]])
   
    x_center=np.array(x_center)
    y_center=np.array(y_center)
    scale_select=np.array(scale_select)
    match_value=np.array(match_value)

    # Remove overlapping bounding boxes
    mask = np.zeros(image.shape, dtype = float)
    index = np.argsort(match_value)
    match_sort = match_value[index[::-1]]
    x_s = [int(x) for x in x_center[index[::-1]]]
    y_s = [int(x) for x in y_center[index[::-1]]]
    bbox = [int(x) for x in scale_select[index[::-1]]]

    x_center_n = []
    y_center_n=[]
    bbox_n=[]
    match_value_n=[]

    for x, y, b, m in zip(x_s, y_s, bbox, match_sort):
        if mask[y,x] == 0:
            y_u=int(y-b/2)
            y_d=int(y+b/2)
            x_l=int(x-b/2)
            x_r=int(x+b/2)
            # cope with boundaries
            if y_u<0: y_u=0
            if x_l<0: x_l=0
            if y_d>mask.shape[0]: y_d=mask.shape[0]
            if x_r>mask.shape[1]: x_r=mask.shape[1]

            mask[y_u:y_d, x_l:x_r] = m
            x_center_n.append(x)
            y_center_n.append(y)
            bbox_n.append(b)
            match_value_n.append(m)
    # Filtered values of center in x,y and bounding box size
    if len(match_value_n) < 1:
        match_results = None
        len_match_results = 0.
        print('None vesicles found')
    else: 
        match_results = np.stack((np.array(x_center_n), np.array(y_center_n),
                                np.array(bbox_n), np.array(match_value_n)), axis = 1)
        print(f'{len(match_results)} vesicles found')         
        len_match_results = len(match_results)

        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(original_image, cmap='gray')
        ax.set_title(filename)

        # Plot each detection box
        for box in match_results:
            x_center, y_center, length = box[:3]
            x_min = x_center - length / 2
            y_min = y_center - length / 2
            rect = patches.Rectangle((x_min, y_min), length, length, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.savefig(f'../../MicroscopyImage/VesicleDetection/Results_{PlateName}/'+f'{filename}_TemplateMatch_{formatted_datetime}.png')
        plt.close(fig)

    return match_results, len_match_results