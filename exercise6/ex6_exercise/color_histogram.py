import cv2
import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    """
    Calculate the normalized histogram of RGB colors in the bounding box.

    Parameters:
    - xmin, ymin: The top left coordinates of the bounding box.
    - xmax, ymax: The bottom right coordinates of the bounding box.
    - frame: The current frame from the video.
    - hist_bin: The number of bins for the histogram for each color channel.

    Returns:
    - hist: The normalized color histogram as a flat array.
    """
    # Crop the frame to the bounding box
    cropped_frame = frame[ymin:ymax, xmin:xmax]
    
    # Initialize histogram
    hist_channels = []
    
    # Calculate the histogram for each color channel
    for i in range(3):  # Assuming frame is in RGB or BGR format
        channel_hist = cv2.calcHist([cropped_frame], [i], None, [hist_bin], [0, 256])
        hist_channels.append(channel_hist)

    # Concatenate histograms for all channels and normalize
    hist = np.concatenate(hist_channels).flatten()
    hist /= np.sum(hist)

    return hist