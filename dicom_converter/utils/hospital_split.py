'''
    Splits a large dataset (grouped by category) into small datasets with custom individual sizes
'''

import shutil
import os
import random
import numpy as np

def hospital_split(number_of_datasplits = 3, split_percentages = np.array([0.5,0.3,0.2]), seed = 3, datapath = "" ):
    '''

    Parameters
    ----------
    number_of_datasplits : integer
        Number of data splits (hospitals needed). The default is 3.
    split_percentages : array
        split percentage by dataset (hospital) the number of components must be equal to 'number_of_datasplits'. Be careful that the sum of them is exactly = 1. The default is np.array([0.5,0.3,0.2]).
    seed : integer
        Random seed used. The default is 3.
    datapath : string
        Datapath directory should only contain folders with each category to classify. Each category fodler must only contain image files of that category. The default is "".

    Returns
    -------
    None.

    '''

    
    
    ####### Parameters #######
    
    # datapath directory should only contain folders with each category to classify. Each category fodler must only contain image files of that category.
    # datapath = "" 
    
    # Number of data splits and percentages
    # number_of_datasplits = 3
    # c = np.array([0.5,0.3,0.2]) # be careful that the sum of them is exactly = 1
    number_of_datasplits_array = np.arange(1,number_of_datasplits+1)
    
    split_array = np.array([number_of_datasplits_array,split_percentages])
    
    #Randomization seed
    # seed =3
    
    ####### find number of categories #######
    
    dirs = os.listdir(datapath)
    
    # fixed parameters
        
    for cat_name in dirs:
        x=0
        cat_set_min = 0
        cat_set_max = 0
        for x in range (0,number_of_datasplits):
            cat_path  = datapath + '/' + cat_name + '/'
            os.makedirs(datapath + '/' + cat_name + str(int(split_array[0,x])) + '/', exist_ok=True)    
            cat_files = os.listdir(cat_path)
            random.Random(seed).shuffle(cat_files) #to shuffle category data
            if x == 0:
                cat_total_files = len(cat_files)
            cat_set_max = int(round(split_array[1,x] * cat_total_files))       
            cat_set = cat_files[cat_set_min: cat_set_min + cat_set_max]
            cat_set_min = cat_set_min + cat_set_max
            for files in cat_set:
                shutil.copy(cat_path + files, datapath + '/' + cat_name + str(int(split_array[0,x])) + '/')
        shutil.rmtree(cat_path)                

        
