'''
    Divides a directory containing only contain folders with each category to classify
    into 'train', 'valid' and 'test' folders contain each category for ML with custom splitting percentages

'''

import shutil
import os
import random

def cat2trainvalidtest(datapath ="",trainset_percentage = 0.7,validset_percentage = 0.2,testset_percentage = 0.1,seed =3):
    '''    

    Parameters
    ----------
    datapath : string
         Datapath directory should only contain folders with each category to classify. Each category fodler must only contain image files of that category.. The default is "".
    trainset_percentage : float
        Training set percentage (trainset_percentage + validset_percentage + testset_percentage should be = 1). The default is 0.7.
    validset_percentage : float
        Validation set percentage (trainset_percentage + validset_percentage + testset_percentage should be = 1). The default is 0.2.
    testset_percentage : float
        Test set percentage (trainset_percentage + validset_percentage + testset_percentage should be = 1). The default is 0.1.
    seed : integer
        Random seed. The default is 3.

    Returns
    -------
    None.

    '''
    

    
    ####### Parameters #######
    
    # datapath directory should only contain folders with each category to classify. Each category fodler must only contain image files of that category.
    # datapath = "" 
    
    
    
    
    
    #trainset_percentage + validset_percentage + testset_percentage should be = 1 !
    # trainset_percentage = 0.7
    # validset_percentage = 0.2
    # testset_percentage = 0.1
    
    #Randomization seed
    # seed =3
    
    ####### find number of categories #######
    
    dirs = os.listdir(datapath)
    
        
    for cat_name in dirs:
          cat_path  = datapath + '/' + cat_name + '/'
          os.makedirs(datapath + '/train/', exist_ok=True)
          os.makedirs(datapath + '/valid/', exist_ok=True)
          os.makedirs(datapath + '/test/', exist_ok=True)
          os.makedirs(datapath + '/train/' + cat_name + '/')
          os.makedirs(datapath + '/valid/' + cat_name + '/')
          os.makedirs(datapath + '/test/' + cat_name + '/')      
          cat_files = os.listdir(cat_path)
          random.Random(seed).shuffle(cat_files) #to shuffle category data
          cat_total_files = len(cat_files)
          cat_trainset_max = round(trainset_percentage * cat_total_files)
          cat_validset_max = round(validset_percentage * cat_total_files)
          cat_testset_max = round(testset_percentage * cat_total_files)
          cat_trainset = cat_files[0:cat_trainset_max]
          cat_validset = cat_files[cat_trainset_max:cat_trainset_max + cat_validset_max]
          cat_testset = cat_files[cat_trainset_max + cat_validset_max:]
          for files in cat_trainset:
              shutil.move(cat_path + files, datapath + '/train/' + cat_name + '/')
          for files in cat_validset:
              shutil.move(cat_path + files, datapath + '/valid/' + cat_name + '/')  
          for files in cat_testset:
              shutil.move(cat_path + files, datapath + '/test/' + cat_name + '/')  
          os.rmdir(cat_path)
            
    
    
