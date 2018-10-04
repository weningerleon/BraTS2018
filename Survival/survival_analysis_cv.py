import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from skimage.measure import label, regionprops
from sklearn.ensemble import RandomForestRegressor

from Segmentation import load_brats
from directories import *

survival_sheet_loc = '/images/brainMRI/brats2018/train/survival_data.csv'
savedir = '/work/local_data/brats/results/survival/'

def survival_analysis(dataset, survival_data, save_location):

    # Take only the those with resection status == GTR
    # Match data from csv file with segmentations
    matches = []
    for i in range(survival_data.__len__()):
        id = survival_data[i]['name']

        if survival_data[i]['resection'] != 'GTR':
            continue

        for x in dataset:
            id_data = x['name']
            if id == id_data:
                matches.append([i, x])

    inputs = np.zeros((matches.__len__(),5))
    outputs = np.zeros(matches.__len__())

    for i, (id_csvsheet, mri_data) in enumerate(matches):

        csv_info = survival_data[id_csvsheet]

        age = csv_info['age']
        survival_days = csv_info['survival_days']

        data = mri_data['data']

        segmentation = data[5]
        segmentation = segmentation.astype(np.int16)

        tumorsize_necrotic = np.count_nonzero(segmentation==1)
        tumorsize_edema = np.count_nonzero(segmentation==2)
        tumorsize_enhancing = np.count_nonzero(segmentation==3)


        # Calculate distance of tumor from brain center

        t1 = data[0]
        brainmask = (t1!=0)

        region_brainmask = 0
        biggest_size = 0
        for region in regionprops(label(brainmask)):
            if region.area > biggest_size:
                biggest_size = region.area
                region_brainmask = region

        x1 = np.asarray(region_brainmask.centroid)

        tumormask = (segmentation!=0)

        region_tumormask = 0
        biggest_size = 0
        for region in regionprops(label(tumormask)):
            if region.area > biggest_size:
                biggest_size = region.area
                region_tumormask = region

        x2 = np.asarray(region_tumormask.centroid)

        dist = np.sqrt(np.sum(np.square(x2-x1)))

        ################################################################################################################
        ############################ Change the inputs here for testing of different features ###########################
        inputs[i, 0] = age
        inputs[i, 1] = tumorsize_necrotic
        inputs[i, 2] = tumorsize_edema
        inputs[i, 3] = tumorsize_enhancing
        inputs[i, 4] = dist

        outputs[i] = survival_days


    # Evaluate cross-validation
    regr = linear_model.LinearRegression()

    print('Age')
    myinputs = inputs.copy()
    myinputs[:, 1:5] = 0
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    print('Age + GDE')
    myinputs = inputs.copy()
    myinputs[:, (1,2,4)] = 0
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    print('Age + ED')
    myinputs = inputs.copy()
    myinputs[:, (1,3,4)] = 0
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    print('Age + NEC')
    myinputs = inputs.copy()
    myinputs[:, (2,3,4)] = 0
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    print('Age + DIST')
    myinputs = inputs.copy()
    myinputs[:, (1,2,3)] = 0
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    print('No age')
    myinputs = inputs.copy()
    myinputs[:, 0] = 0
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    print('All')
    myinputs = inputs.copy()
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    print('All - SVR')
    myinputs = inputs.copy()
    regr = RandomForestRegressor(n_estimators=25)
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_mean_squared_error')
    print("Mean squared error: %.2f" % -scores.mean())
    scores = cross_val_score(regr, myinputs, outputs, cv=59, scoring='neg_median_absolute_error')
    print("Mean Median error: %.2f" % -scores.mean())

    return 1


########################################################################################################################
########################################################################################################################

def main(savedir=savedir):
    data_set = load_brats.load_normalized_data(dir=savedir_preproc_train1)
    # For this cross-evalutation, we use the groundtruth-segmentations
    # so that the results are not biased through our segmentation algorithm

    survival_data = np.loadtxt(survival_sheet_loc, delimiter=',', skiprows=1,
                                dtype={'names': ('name', 'age', 'survival_days', 'resection'),
                                       'formats': ('U20', 'f4', 'i4', 'U10')
                                       })

    survival_analysis(data_set, survival_data, savedir)

    return 0


if __name__ == "__main__":
    main()
