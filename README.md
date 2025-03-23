PET_CSF (Non-spatial)
===========================
This document is used to show a detailed description of the Non-spatial_PET_CSF project.

****
 
| Project Name | Non-spatial_PET_CSF |
|--------------|---------------------|
| Author       | --                  |
| Version      | --                  |

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
3. Edit file config.py to modify the start conditions for each line
4. Please help check if ADTruth.pend() function is correct

****
# Introduction to files
1. parameters.py: set parameters (k_xxx, n_xxx, K_xxx, d_xxx)
2. config.py: set matrix size (160), the Laplacian matrix, and the start conditions
3. ode_truth.py: solve the ode and draw the trend curves
4. utils.py: functions for plotting

****
# Start
See `https://github.com/EnzeXu/Non-Spatial_PET_CST` or
```shell
$ git clone https://github.com/EnzeXu/Non-Spatial_PET_CST.git
```

Set virtual environment and install packages: (Python 3.7+ (3.7, 3.8, 3.9 or higher) is preferred)
```shell
$ python3 -m venv ./venv # MacOS, Linux or Unix
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

Run
```shell
(venv) $ python test_nsga.py --model_name new_test --generation 10 --dataset chosen_0 --start ranged --pop_size 100 --option option1 --tcsf_scaler 0.7
#  -h, --help            show this help message and exit
#  --dataset DATASET     dataset strategy
#  --start START         start strategy
#  --generation GENERATION
#                        generation
#  --pop_size POP_SIZE   pop_size
#  --model_name MODEL_NAME
#                        model_name
#  --option {option1,option2}
#                        option
#  --tcsf_scaler TCSF_SCALER
#                        tcsf_scaler
```

Exit virtual environment
```shell
(venv) $ deactivate
```
****

# Contact
If you have any questions, suggestions, or improvements, please get in touch with xue20@wfu.edu
****