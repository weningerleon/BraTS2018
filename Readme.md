# BraTS Challenge 2018 Contribution

Our contribution to the BraTS Challenge 2018, both for the segmentation and survival prediction task. 
We are happy to announce that the straight-forward survival prediction approach ranked 3rd place.

This repository is divided in code for the segmentation and survival prediction task.

Please note that you can also try out the docker images, found on:  
https://hub.docker.com/r/leonweninger/brats18_segmentation/  
https://hub.docker.com/r/leonweninger/brats18_survival/

For any questions concerning the code or submission, please feel free to open an issue.

## Prerequisites

Depending on the task you want to check out, the following libraries may be needed:  
- Python 3.6
- Numpy  
- PyTorch 0.4.0   
- Dipy  
- Scikit-image  
- Scikit-learn  


## Segmentation

Before starting, you need to set up your paths in the file Segmentation/directories.py.
For training, you can run the file train_segmentation.sh, which does the preprocessing as well as the final training
For prediction, you should run the run_segmentation.sh file, eventually adapting the parameters to your needs

To change between training and testing set, you can change the parameters in the run_segmentation.sh file from
"validation" to "test"

## Survival Prediction
There are three different python files in this directory. If you want to reproduce the results from the challenge, you'll need to run survival_analysis.py. Please note that both, the training and testing .csv file are necessary.
For reproduction of the leave-one-out cross-validation results with different input features, check out the survival_analysis_cv.py file.

