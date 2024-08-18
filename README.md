# Space Weather Instrument Scientist Computing Task

This code analyses the vorticity values determined from measurements of ionospheric velocity made pairs of radars (in 
the Northern Hemisphere) of the Super Dual Auroral Radar Network (SuperDARN). 

As outputs, this code generates images which display the 

1. mean, 
2. median, 
3. number of data points, 
4. standard deviation,
5. Maximum Magnitude of Vorticity
6. Minimum Magnitude of Vorticity

at different Magnetic Local Times (MLT) and Altitude Adjustment Corrected Geomagnetic (AACGM) latitudes.

1 set of images displays these statistics for the entire dataset, while the other set displays the above statistics at 
different seasons

Apart from the above, the code also produces:

7. A plot showing the average vorticity vs. MLT in the R1 vorticity system
8. A Histogram showing the distribution of the area of the loops used to calculate the vorticities
9. A log file for miscellaneous statistics, for curious minds

## Table of Contents

- [Usage](#usage)
- [Installation](#installation)
- [Contact](#contact)

## Usage

1. Download the data from https://doi.org/10.5285/8eedc594-730b-4aad-b9ce-827912320c3a, and put the files in a directory 
of your choosing. The code can handle as many files as one desires, not just wg_v_00_05.txt.


2. Go to the **_params_** directory in the code, and adjust the parameters for the run (stored in the .json files) 
according to your preferences.

Recommendations:
- Change 'abs_data_txt_dir' to the directory where **_only_** the text file(s) with the vorticity data is stored.
- Change 'output_dir' to the directory in where you want the output files (e.g. plots, logs, etc.) to be stored.
- Only set 'txt_files_to_json' to 'true' for your first run, and false in subsequent runs. The code turns the data 
into Json files according to the radars involved, and the timestamps of the data, so that it can read 
them in parallel later.
- Other parameters work best with their default values. If you would like to change them, go to the **_classes_** 
directory and see the main_runparams_cls.py and plot_params_cls.py for reference.

3. Run the following:
```bash
python main.py
```

## Installation

Step-by-step instructions on how to get a development environment running.

### Python Version
This code has been optimised for Python 3.10.14. For installation details, see
https://www.python.org/downloads/release/python-31014/

P.S. For Macs, it is more convenient to install it via homebrew, with the following:
```bash
# install Homebrew (one of the best package managers for Macs)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.10
```

### Setting up the Environment 

After going to **your desired directory** to clone the repository, do the following:

#### For Linux/ Macs
```bash
# Clone the repository
git clone https://github.com/renzolam/instrument_scientist_task.git

# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate 

# Install the dependencies
pip install -r requirements.txt
```

#### For Windows

1. Download the repository throught the Github Desktop all, or just download the project via:
https://github.com/renzolam/instrument_scientist_task.git

2. Run the following with cmd as **_the administrator_**

```
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
```

## Contact
This code has been written entirely by Pak Yin (Renzo) Lam, Space Weather Research Assistant at the British Antarctic 
Survey (BAS)

- Email: paklam@bas.ac.uk
