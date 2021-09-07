import shutil
import os


####### Parameters #######

# datapath directory should only contain a 'train', 'test' and 'valid' (code is case sensitive) folders. Each fodler should only contain folders with each category to classify. Each category fodler must only contain image files of that category.
datapath = "" 




dirs = os.listdir(datapath + '/train/')

for cat_name in dirs:

    os.makedirs(datapath + '/' + cat_name + '/', exist_ok=True)
    dirs_train = os.listdir(datapath + '/train/' + cat_name + '/')
    dirs_valid = os.listdir(datapath + '/valid/' + cat_name + '/')
    dirs_test = os.listdir(datapath + '/test/' + cat_name + '/')



    for files in dirs_train:
        shutil.move(datapath + '/train/' + cat_name + '/' + files, datapath + '/' + cat_name + '/') 
    for files in dirs_valid:
        shutil.move(datapath + '/valid/' + cat_name + '/' + files, datapath + '/' + cat_name + '/') 
    for files in dirs_test:
        shutil.move(datapath + '/test/' + cat_name + '/' + files, datapath + '/' + cat_name + '/') 
        
shutil.rmtree(datapath + '/train/', ignore_errors=True)
shutil.rmtree(datapath + '/valid/', ignore_errors=True)
shutil.rmtree(datapath + '/test/', ignore_errors=True)

