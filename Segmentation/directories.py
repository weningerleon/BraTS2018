import os
from os.path import join as opj

#####################################################
# Checks and creates the necessary folder structure #
#####################################################

basedir = '/work/weninger/brats/'

#%% TRAIN
raw_dir_train = '/images/brainMRI/brats2018/train'

savedir_train = opj(basedir,'train')

savedir_results_train = opj(savedir_train, 'results')
savedir_results_train1 = opj(savedir_results_train, 'step1')
savedir_results_train2 = opj(savedir_results_train, 'step2')

savedir_preproc_train = opj(savedir_train, 'preproc')
savedir_preproc_train1 = opj(savedir_preproc_train, 'step1')
savedir_preproc_train2 = opj(savedir_preproc_train, 'step2')


#%% VALIDATION
raw_dir_validate = '/images/brainMRI/brats2018/validation'

val_dir = opj(basedir, 'val')
savedir_preproc_val = opj(val_dir, 'preproc')
savedir_preproc_val1 = opj(val_dir, 'preproc/step1')
savedir_preproc_val2 = opj(val_dir, 'preproc/step2')

savedir_results_val = opj(val_dir, 'results')
savedir_results_val1 = opj(val_dir, 'results/step1')
savedir_results_val2 = opj(val_dir, 'results/step2')

#%% TEST
raw_dir_test = '/images/brainMRI/brats2018/test'

test_dir = opj(basedir, 'test')

savedir_preproc_test = opj(test_dir, 'preproc')
savedir_preproc_test1 = opj(test_dir, 'preproc/step1')
savedir_preproc_test2 = opj(test_dir, 'preproc/step2')

savedir_results_test = opj(test_dir, 'results')
savedir_results_test1 = opj(test_dir, 'results/step1')
savedir_results_test2 = opj(test_dir, 'results/step2')


# OTHER

savedir_nets = opj(basedir, 'nets')
savedir_nets1 = opj(savedir_nets, 'step1')
savedir_nets2 = opj(savedir_nets, 'step2')
temp_dir = opj(basedir, 'temp')


def main():
    try:
        os.stat(raw_dir_train)
        os.stat(raw_dir_validate)
        os.stat(raw_dir_test)
    except:
        print("Brats dataset not found!!")
        return

    print("creating folder structure")

    list_dirs = [basedir, savedir_nets, savedir_nets1, savedir_nets2, savedir_train, savedir_results_train, savedir_results_train1,
                 savedir_results_train2, savedir_preproc_train, savedir_preproc_train2, temp_dir, savedir_preproc_train1, savedir_preproc_train2,
                 val_dir, savedir_preproc_val, savedir_preproc_val1, savedir_preproc_val2, savedir_results_val,
                 savedir_results_val1, savedir_results_val2,
                 test_dir, savedir_preproc_test, savedir_preproc_test1, savedir_preproc_test2, savedir_results_test,
                 savedir_results_test1, savedir_results_test2]

    for i in list_dirs:
        try:
            os.stat(i)
        except:
            os.mkdir(i)


if __name__ == "__main__":
    main()
