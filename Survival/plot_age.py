

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

survival_sheet_loc_train = '/images/brainMRI/brats2018/train/survival_data.csv'
survival_sheet_loc_val = '/images/brainMRI/brats2018/validation/survival_evaluation.csv'
survival_sheet_loc_test = '/images/brainMRI/brats2018/test/survival_evaluation.csv'

########################################################################################################################
########################################################################################################################


def main():
    survival_data = np.loadtxt(survival_sheet_loc_train, delimiter=',', skiprows=1,
                               dtype={'names': ('name', 'age', 'survival_days', 'resection'),
                                      'formats': ('U20', 'f4', 'i4', 'U10')
                                      })

    gtr_data = []

    for i in range(survival_data.__len__()):
        name = survival_data[i]['name']
        age = survival_data[i]['age']
        survival = survival_data[i]['survival_days']

        if survival_data[i]['resection'] == 'GTR':
            gtr_data.append(np.array([name, age, survival]))

    t1 = np.asarray(gtr_data)
    input_gtr = t1[:, 1].astype(np.float).reshape(-1, 1)  # Age
    label_gtr = t1[:, 2].astype(np.float)  # Survival

    regr_gtr = linear_model.LinearRegression()
    regr_gtr.fit(input_gtr, label_gtr)

    plt.plot(input_gtr, label_gtr / 365, '.', color='r', label='Train Set')
    plt.axhline(y=10/12, color='g', linewidth=1, label='Classification Boundaries')
    plt.axhline(y=15/12, color='g', linewidth=1)

    x = np.asarray(range(15,95)).reshape(-1, 1)

    regr_line = regr_gtr.predict(x)

    plt.plot(x, regr_line / 365, color='b', label='Prediction')
    plt.axis([20, 90, 0, 5])

    plt.text(21,0.55, 'short-survivers', color='g')
    plt.text(21,0.98, 'mid-survivers', color='g')
    plt.text(21,1.4, 'long-survivers', color='g')
    plt.legend()
    plt.xlabel('Age (years)')
    plt.ylabel('Survival (years)')
    plt.title('Linear Regressor Survival / Age')
    plt.savefig('reg.png', format='png')
    plt.show()

    return gtr_data


if __name__ == "__main__":
    main()
