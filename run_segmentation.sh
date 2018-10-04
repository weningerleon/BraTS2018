python ./Segmentation/directories.py
python ./Segmentation/preproc1.py --mode validation
python ./Segmentation/predict1.py --mode validation --net ./Nets/net_step1.pth -d 2
python ./Segmentation/predict2.py --mode validation --net ./Nets/net_step2.pth -d 2
