#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 13:58:39 2021

@author: eloyen
"""
import os
import pydicom
import json
import shutil
import argparse


def get_tag_from_json(json_path,tag):
    '''
    Get tag value from .json fiel containing DICOM metadata
    Parameters
    ----------
    json_path : string
        /.../dicominfo.json
    tag : tuple of two elements
        DICOM tag, must be in hexagonal format. ex: (0x10,0x20)

    Returns
    -------
    value : Value stored in tag
    '''
    
    ds_json=json.load(open(json_path+'.json'))
    ds = pydicom.dataset.Dataset.from_json(ds_json)
    value=ds[tag].value
    return value

# Classer les fichiers en: Covid vs NON-Covid sur base du tag dans le .json
def classify_in_labelled_folders(inputFolder, labelTag, outputDir):
    '''
    Classify the IMAGES vs METADATA in folders according to the label tag   
    Parameters
    ----------
    inputFolder : string
        /.../dicoms/
    labelTag : tuple of two elements
        DICOM tag, must be in hexagonal format. ex: (0x10,0x20)
    outputDir : string
        /.../outputs
    Returns
    -------
    None.
    '''
    # lire uniquement les .json et classer en folders de types et labellisés 
    for file in os.listdir(inputFolder):
        if file.endswith(".json"):
            jsonFilePath = inputFolder + file
            associatedPngFilePath = jsonFilePath[:-5] + '.png'
            labelValue = get_tag_from_json(jsonFilePath[:-5], labelTag)
            
            
            newPathJson = outputDir + '/METADATA/JSON-' + str(labelValue) + '/'
            if not os.path.exists(newPathJson):
                os.makedirs(newPathJson)                
            newPathPng = outputDir + '/IMAGES/PNG-' + str(labelValue) + '/'
            if not os.path.exists(newPathPng):
                os.makedirs(newPathPng)
            
            shutil.copy(jsonFilePath, newPathJson)
            shutil.copy(associatedPngFilePath, newPathPng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('inputFolder', help = 'Path to the input Folder')
    parser.add_argument('tag', help = 'Tag to add search for in the JSON')
    parser.add_argument('outputFolder', help= 'Path where to create the new folders')
    args = parser.parse_args()
    
    classify_in_labelled_folders(args.inputFolder, eval(args.tag), args.outputFolder)
    #classify_in_labelled_folders('/Users/eloyen/Desktop/folderTRAIL/', [0x0014,0x2018], '/Users/eloyen/Desktop/folderTRAIL/')

