PET_CSF (Non-spatial)
===========================
This document is used to show a detailed description of the Non-spatial_PET_CSF project.

****
 
| Project Name | PET_CSF |
|--------------|---------|
| Author       | --      |
| Version      | --      |

****
# Catalogue
* [TODO](#todo)
* [Introduction to files](#introduction-to-files)
* [Start](#start)
* [Contact](#contact)

****

# TODO
1. Edit file parameters.py to modify parameters
2. Edit file config.py to modify the Laplacian matrix (it's now set as an identity matrix)
3. Edit file config.py to modify the start conditions for each line (`8*160+3*1=1283` in total)
4. Please help check if ADTruth.pend() function is correct

****
# Introduction to files
1. parameters.py: set parameters (k_xxx, n_xxx, K_xxx, d_xxx)
2. config.py: set matrix size (160), the Laplacian matrix, and the start conditions
3. ode_truth.py: solve the ode and draw the trend curves
4. utils.py: functions for plotting

****
# Start
See `https://github.com/chenm19/PET_CSF` or
```shell
$ git clone https://github.com/chenm19/PET_CSF.git
```

Set virtual environment and install packages: (Python 3.7+ (3.7, 3.8, 3.9 or higher) is preferred)
```shell
$ python3 -m venv ./venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

Run
```shell
(venv) $ python3 ode_truth.py
```

Exit virtual environment
```shell
(venv) $ deactivate
```
****

# Contact
If you have any questions, suggestions, or improvements, please get in touch with xue20@wfu.edu
****