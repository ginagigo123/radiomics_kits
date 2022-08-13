# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:49:36 2022

@author: Gina
"""

import radiomics
from radiomics import featureextractor  # This module is used for interaction with pyradiomics

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import six


def print_image_info(img_path):
    image = sitk.ReadImage(img_path)

    print(image.GetSize())

    # 像素類型
    print(image.GetPixelIDValue())
    print(image.GetPixelIDTypeAsString())
    print(image.GetNumberOfComponentsPerPixel())

    # 其他屬性
    print(image.GetOrigin())
    print(image.GetSpacing())
    print(image.GetDirection())
    

case = 'case_00001'
dataDir = 'C:\\Users\\Gina\\Lab\\kidney\\nnUNet-1\\DataSet\\nnUnet_raw\\nnUNet_raw_data\\Task135_KiTS2021'

imagePath = os.path.join(dataDir, 'imagesTr', case + '_0000.nii.gz')
labelPath = os.path.join(dataDir, 'labelsTr', case + '.nii.gz')

testPath = os.path.join('C:\\Users\\Gina', 'test.nii.gz')

print('imagePath:', imagePath)
print('labePath:', labelPath)

print_image_info(imagePath)
print_image_info(labelPath)
print_image_info(testPath)


#%%

# read parameter
paramsFile = os.path.abspath(r'exampleCT.yaml')

# feature extract
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
extractor.enableAllFeatures()

print("Extraction parameters:\n\t", extractor.settings)
print("Enabled filters:\n\t", extractor.enabledImagetypes)
print("Enabled features:\n\t", extractor.enabledFeatures)

#%%

records = []

export_path = 'C:\\Users\\Gina\\Lab\\kidney\\radiomics\\result1'
idx = 0

for i in range(0, 300):

    case = f'case_{i:05d}'
    idx += 1
    
    export_case_path = export_path + '\\' + case
    print(case)
    if os.path.exists(export_case_path) is False:
        os.mkdir(export_case_path)
    
    imagePath = os.path.join(dataDir, 'imagesTr', case + '_0000.nii.gz')
    labelPath = os.path.join(dataDir, 'labelsTr', case + '.nii.gz')
    
    result = extractor.execute(imagePath, labelPath) # , voxelBased=True
    record = {}
    for featureName, featureValue in six.iteritems(result):
      if isinstance(featureValue, sitk.Image):
        sitk.WriteImage(featureValue, export_case_path + '\\%s_%s.nrrd' % (case, featureName))
        print('Computed %s, stored as "%s_%s.nrrd"' % (featureName, case, featureName))
      else:
        print('%s: %s' % (featureName, featureValue))
        record[featureName] = featureValue
    records.append(record)

df = pd.DataFrame(records)
df.to_csv(f"C:\\Users\\Gina\\Lab\\kidney\\radiomics\\result\\{i}_radiomics_feature.csv")


#%%
# test image transform to numpy array

def image_to_np(path):

    label_image = sitk.ReadImage(path)
    label_np = sitk.GetArrayFromImage(label_image) # 深度複製（deep copy）
    return label_np

def write_np_to_image(image_np):
    # convert numpy into SimpleITK's image
    image_np = np.transpose(image_np, (2, 1, 0))
    img = sitk.GetImageFromArray(image_np)
    writer = sitk.ImageFileWriter()
    writer.SetFileName('test.nii.gz')
    writer.Execute(img)

# convert to numpy array
nda_copy = image_to_np('C:\\Users\\Gina\\Lab\\kidney\\radiomics\\pyradiomics\\examples\\1_feature_map\\output_original_firstorder_Energy.nrrd')
np.argwhere(np.isnan(nda_copy))
