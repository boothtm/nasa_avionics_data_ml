# NASA_avionics_data_ML

## FYI:
- The python and notebooks in this project were used as for research towards writing a paper ("A Gradient Descent Multi-Algorithm Grid Search Optimization of Deep Learning for Sensor Fusion") and are not stellar examples of proper software development, but they worked for me.
  - Link to paper: https://par.nsf.gov/servlets/purl/10448779
- There may also be some files missing since i couldn't upload my entire 18GB folder that I used to perform this research.



## Getting started

- You need to download the NASA matlab files first.  There is a powershell script `download_687_1_flight_data.ps1` that provides a good example if your on a windows machine.
- unzip the file/script
- open the `ETL.ipynb` notebook to see how to convert the matlab (*.m) files into a metadata JSON file and a time-series flight data parquet file.  This should run if everything is named correctly.

Once all the files you want to convert to parquet are done you can explore the data with the following files:
- `Altitude_visualization.ipynb`
- `NASA_data_exploration.ipynb`

The DTED data that is required for a few of the RNN functions can be downloaded from my NexusDE Google Drive:
- https://drive.google.com/file/d/15yxI1QQluxYLN2YGn64ISKCPyZd2X3By/view?usp=drive_link

Once DTED is downloaded you can visualize it with `elevation_visualization.ipynb`
