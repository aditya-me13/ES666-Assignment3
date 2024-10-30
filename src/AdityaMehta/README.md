# ES666-Assignment3

### Prerequisites:

To run the project, it is advised to make a virtual environment and install the required packages. To do so, follow the steps below:

1. Create a virtual environment using the following command:
``` bash
conda create --name CV python=3.10
```
2. Activate the virtual environment using the following command:
``` bash
conda activate CV
```
3. Install the required packages using the following command:
``` bash
pip install -r requirements.txt
```

### Important Note:
- Running the code was computationally expensive, so the images has been scaled down by a factor of ```0.3``` in order to reduce the computational time.
- To see the actual results without any scaling, navigate to ```src/AdityaMehta/Helpers/Master.py``` and change the value of ```SCALE``` to ```100```.
- The ```Helpers``` module also has python files for each step in the pipeline of Panorama Stitching. Running individual files will give you the results of each step saved in the respective folders.

