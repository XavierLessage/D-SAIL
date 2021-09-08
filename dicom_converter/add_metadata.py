#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:06:08 2021

@author: eloyen
"""
import os
import pydicom
import argparse

## Add a new label 'Indication Label' for the classificatiion task
def add_label_in_dcm(dcmFile, label, tag):
    '''
    Add the label metadata in the DICOM file

    Parameters
    ----------
    dcmFile : string
        /.../dicominfo.json
    label : short string
        ex: '0', '1'
    tag : tuple of two elements
        DICOM tag, must be in hexagonal format. ex: (0x10,0x20)

    Returns
    -------
    dcmFile : string 
        dcmFile with the new added metadata
    '''
    dcmFile.add_new(tag, "SH", label) # [0x0014,0x2016] = Indication Label
    return dcmFile

def go_through_folder(folderPath, label, tag):
    '''
    Go trough the folder to add the label tag metadata

    Parameters
    ----------
    folderPath : string
        /.../dicoms/
    label : short string
        ex: '0', '1'
    tag : tuple of two elements
        DICOM tag, must be in hexagonal format. ex: (0x10,0x20)
    
    Returns
    -------
    None.
    '''
    for file in os.listdir(folderPath):
        #print('file', file)

        dcmFile = pydicom.dcmread(folderPath+file)
        newDcmFile = add_label_in_dcm(dcmFile, label, tag)
        newDcmFile.save_as(folderPath+file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('inputFolder', help = 'Path to the input Folder')
    parser.add_argument('label', help='label to classify')
    parser.add_argument('tag', help = 'Tag to add in the DCM file')
    args = parser.parse_args()
    
    go_through_folder(args.inputFolder, args.label, eval(args.tag))
    


