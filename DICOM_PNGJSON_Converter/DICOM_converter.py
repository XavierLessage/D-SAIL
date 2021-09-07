# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:44:11 2021
@author: to fill
Module with functions used to convert DICOM files (.dcm) to .png/.bmp and .json files and from .png/.bmp to DICOM. 
The function 'compressToPng' calls executables from OpenJPEG (https://www.openjpeg.org/) available at 'https://github.com/uclouvain/openjpeg/releases/tag/v2.4.0'. 
"""

import os
import random
import string
import json
import subprocess
import cv2
import pydicom
import numpy as np

def imgFromDICOM(ds):
    '''
    Extract array from dicom dataset 'dcm' with [0,256] pixel intensities.
    Parameters
    ----------
    dcm : FileDataset object of pydicom.dataset module
    Returns
    -------
    d : array
        Image array of the dicom dataset
    '''

    data=ds.pixel_array
    
    if 'WindowWidth' in ds:
    
        # Uses window levels written in dicom header    
        data = pydicom.pixel_data_handlers.util.apply_voi_lut(ds.pixel_array, ds)
        
    img = np.round(cv2.normalize(data,  None, 0, 255, cv2.NORM_MINMAX)).astype('uint8')
    
    return img
    
def decomposeDICOM(file_path,output_path,img_format='bmp',removeImgInJson=False):
    '''
    Divides dicom file into a .json file with the dicom metadata and a 
    .'img_format' file containing the image.
    Parameters
    ----------
    file_path : string
        /.../filename.dcm
    output_path : string
        /.../foldername/
    img_format : string, optional
        Image file format : bmp, png, ... The default is 'bmp'.
    removeImgInJson : True/False, optional
        Removes PixelData from dicom metadata. The default is False.
    Returns
    -------
    None.
    '''
    
    filename=file_path.rsplit("/")[-1]
    if filename.endswith('.dcm'):
        filename=filename[:-4]
    
    # Open DICOM
    
    ds = pydicom.dcmread(file_path,force=True)
    
    img = imgFromDICOM(ds)
    
    if removeImgInJson==True:
        ds.PixelData=None
    
    metadata=ds.to_json_dict()
    
    cv2.imwrite(output_path+filename+'.'+img_format, img)
    with open(output_path+filename+'.json','w') as outfile:
        json.dump(metadata, outfile)
    
def dicomFromImgOrJson(file_path,output_folder,metadata_path=None,
                       randomizeName=False,verbose=False):
    '''
    Creates a dicom from a .json file or a .png/.bmp file
    Parameters
    ----------
    file_path : string
        /.../filename.png|.json
    output_path : string
        /.../foldername/
    metadata_path : string, optional
        path to reference dicom file. The default is dcm_file_path.
    randomizeName : True/False        
        Creates a random patientID, optional
    verbose : True/False, optional
        Raises warnings. The default is False.
    Returns
    -------
    None.
    '''
    
    file_path_ex=file_path
    
    suffix=file_path.split(".")[-1]
    file_path=file_path.split(".")[:-1]
    file_path='.'.join(file_path)
    
    filename=file_path.rsplit("/")[-1]
    
    if metadata_path is not None and os.path.exists(metadata_path):
        ds = pydicom.dcmread(metadata_path,force=True)
        ds.PixelData=None
    elif suffix=='json' or os.path.exists(file_path+'.json'):
        ds_json=json.load(open(file_path+'.json'))
        ds = pydicom.dataset.Dataset.from_json(ds_json)
    elif verbose:
        raise RuntimeWarning('No .json file ('+
                             file_path_ex+'.json) or metadata_path found')

    if suffix=='png' or suffix=='bmp' or ds.PixelData==None:
        if ds.PixelData==None:
            suffixList=['png','bmp']
        else:
            suffixList=[suffix]
        for suff in suffixList:
            if os.path.exists(file_path+'.'+suff):
                im=cv2.imread(file_path+'.'+suff)
                img=im[:,:,0].flatten()

                #ds.PixelData=bytes(img)
                ds.PixelData=img
                
                # To conform with uint8 greyscale png format
                ds.BitsAllocated=8      # bit depth of image
                ds.BitsStored=8
                ds.HighBit = ds.BitsStored - 1
                ds.Rows=im.shape[0]
                ds.Columns=im.shape[1]
                ds.SamplesPerPixel = 1 # 1 for greyscale, 3 for RGB
                ds.PixelRepresentation = 0 # 0 for unsigned, 1 for signed data
                ds.WindowWidth=np.max(im)-np.min(im)
                ds.WindowCenter=np.round(np.max(im)-ds.WindowWidth/2)
                break
                
            elif verbose:
                raise RuntimeWarning('No pixelData in json or .png file found')
    
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    # Necessary to create metadata and pixel_array tag
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    
    if randomizeName:       #Improv on Patient name
        letters = string.ascii_lowercase
        ds.PatientID=''.join(random.choice(letters) for i in range(10))
    
    ds.save_as(output_folder+filename+'.dcm')

def compressToPng(file_path,software_root,compress_ratio=1):
    '''
    Compresses an image to a .png with a specified 'compress_ratio'
    Parameters
    ----------
    file_path : string
        /.../filename.png
    software_root : string
        Path to the folder containing the openjpeg .exe programs
    compress_ratio : int, optional
        Best if multiple of 8. The default is 1.
    Returns
    -------
    None.
    '''
    
    subprocess.call(software_root+'opj_compress'+
              ' -i '+file_path+
              ' -o '+file_path.rsplit(".",1)[0]+'.j2k'+
              ' -r '+str(compress_ratio))
    subprocess.call(software_root+'opj_decompress'+
              ' -i '+file_path.rsplit(".",1)[0]+'.j2k'+
              ' -o '+file_path.rsplit(".",1)[0]+'.png')
    os.system('rm '+file_path.rsplit(".",1)[0]+'.j2k')
    
def getTagFromJson(json_path,tag):
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
