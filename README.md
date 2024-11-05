# droplet-in-droplet

Python Scripts for:

1. Extracting metadata from .lif files acquired with Leica microscopes (lif_metadata.py)
2. Creating required local directories (preprocess.py)
3. Visualizing stack images (show_stack.py)
4. Analyzing droplet-in-droplet structures using ROI detection (set_roi.py)

The ROI analysis and measurements take approximately 4 seconds per field. Total processing time can be estimated by multiplying this by the number of frames and positions on a standard desktop computer.  
Usage examples are provided in the Jupyter notebook (example notebook.ipynb).  
Sample raw data is included in the zip file for testing purposes.  
Expected outputs can be verified against the source data provided in the paper.  
System Requirements:

macOS Sonoma 14.3.1  
Python 3.11  
OpenCV 4.6.0  
AICSImageIO 4.11.0  
Library installation should take less than 10 minutes on a standard desktop computer.  

The codes are available under a CC-BY-NC-ND 4.0 International license.  
Kanji Tomohara, 2024
