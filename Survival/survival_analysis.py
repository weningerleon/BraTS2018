import numpy as np
from sklearn import linear_model

survival_sheet_loc_train = '/images/brainMRI/brats2018/train/survival_data.csv'
survival_sheet_loc_val = '/images/brainMRI/brats2018/validation/survival_evaluation.csv'
survival_sheet_loc_test = '/images/brainMRI/brats2018/test/survival_evaluation.csv'

survival_sheet_save_train = '/work/weninger/brats/train/results/survival.csv'
survival_sheet_save_val = '/work/weninger/brats/val/results/survival.csv'
survival_sheet_save_test = '/work/weninger/brats/test/results/survival.csv'

########################################################################################################################
########################################################################################################################


def train(train_data):

    t = np.asarray(train_data)
    input = t[:,1].astype(np.float).reshape(-1, 1)  # Age
    label = t[:,2].astype(np.float)  # Survival

    regr = linear_model.LinearRegression()
    regr.fit(input, label)

    # Check results on the train set
    pred = regr.predict(input)
    pred = pred.astype(np.int)

    save_data = t[:,0:2].copy()
    save_data[:,1] = pred

    np.savetxt(survival_sheet_save_train, save_data, fmt='%s', delimiter=',')

    return regr


def predict(test_data, regr, survival_sheet):

    t = np.asarray(test_data)
    input = t[:,1].astype(np.float).reshape(-1, 1)  # Age

    pred = regr.predict(input)
    pred = pred.astype(np.int)

    save_data = t[:, 0:2].copy()
    save_data[:, 1] = pred

    np.savetxt(survival_sheet, save_data, fmt='%s', delimiter=',')


def load_train_data(survival_sheet_loc):

    survival_data = np.loadtxt(survival_sheet_loc, delimiter=',', skiprows=1,
                                dtype={'names': ('name', 'age', 'survival_days', 'resection'),
                                       'formats': ('U20', 'f4', 'i4', 'U10')
                                       })

    train_data = []

    for i in range(survival_data.__len__()):
        if survival_data[i]['resection'] != 'GTR':
            continue

        name = survival_data[i]['name']
        age = survival_data[i]['age']
        survival = survival_data[i]['survival_days']

        train_data.append(np.array([name, age, survival]))

    return train_data


def load_test_data(survival_sheet_loc):

    survival_data = np.loadtxt(survival_sheet_loc, delimiter=',', skiprows=1,
                                dtype={'names': ('name', 'age', 'resection'),
                                       'formats': ('U20', 'f4', 'U10')
                                       })

    test_data = []

    for i in range(survival_data.__len__()):
        if survival_data[i]['resection'] != 'GTR':
            continue

        name = survival_data[i]['name']
        age = survival_data[i]['age']

        test_data.append(np.array([name, age, 0]))

    return test_data


def main():
    train_data = load_train_data(survival_sheet_loc=survival_sheet_loc_train)
    regr = train(train_data)
    print("Training finished")

    val_data = load_test_data(survival_sheet_loc=survival_sheet_loc_val)
    predict(val_data, regr, survival_sheet_save_val)
    print("Prediction validation set finished")

    test_data = load_test_data(survival_sheet_loc=survival_sheet_loc_test)
    predict(test_data, regr, survival_sheet_save_test)
    print("Prediction test set finished")

    return 0


if __name__ == "__main__":
    main()
