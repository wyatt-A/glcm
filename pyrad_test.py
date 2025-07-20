import radiomics
from radiomics import glcm
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np

#image volume (3x3x3)
data = [1,1,2,2,4,5,3,4,2,3,2,1,4,2,5,3,1,1,5,3,2,1,1,5,4,4,1]

arr = np.array(data, dtype=np.int16).reshape((3, 3, 3), order='F')  # 3 rows, 9 columns
vol = np.asfortranarray(arr)
img = sitk.GetImageFromArray(vol)
img.SetSpacing((1.0, 1.0, 1.0))

#image mask (3x3x3)
msk_data = [0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
mask = np.array(msk_data, dtype=np.uint8).reshape((3, 3, 3), order='F')  # 3 rows, 9 columns
mask_img = sitk.GetImageFromArray(mask)

extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=1)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("glcm")   # co‑occurrence only

glcm_calc = glcm.RadiomicsGLCM(img, mask_img, **extractor.settings)
matrix = glcm_calc._calculateMatrix()
glcm_calc.P_glcm = matrix
glcm_calc._calculateCoefficients()  

print("calculated GLCM features:")
x = glcm_calc.getAutocorrelationFeatureValue()
print(f"1: {x[0]}")
x = glcm_calc.getJointAverageFeatureValue()
print(f"2: {x[0]}")
x = glcm_calc.getClusterProminenceFeatureValue()
print(f"3: {x[0]}")
x = glcm_calc.getClusterShadeFeatureValue()
print(f"4: {x[0]}")
x = glcm_calc.getClusterTendencyFeatureValue()
print(f"5: {x[0]}")
x = glcm_calc.getContrastFeatureValue()
print(f"6: {x[0]}")
x = glcm_calc.getCorrelationFeatureValue()
print(f"7: {x[0]}")
x = glcm_calc.getDifferenceAverageFeatureValue()
print(f"8: {x[0]}")
x = glcm_calc.getDifferenceEntropyFeatureValue()
print(f"9: {x[0]}")
x = glcm_calc.getDifferenceVarianceFeatureValue()
print(f"10: {x[0]}")
x = glcm_calc.getJointEnergyFeatureValue()
print(f"11: {x[0]}")
x = glcm_calc.getJointEntropyFeatureValue()
print(f"12: {x[0]}")
x = glcm_calc.getImc1FeatureValue()
print(f"13: {x[0]}")
x = glcm_calc.getImc2FeatureValue()
print(f"14: {x[0]}")
x = glcm_calc.getIdmFeatureValue()
print(f"15: {x[0]}")
x = glcm_calc.getMCCFeatureValue()
print(f"16: {x[0]}")
x = glcm_calc.getIdmnFeatureValue()
print(f"17: {x[0]}")
x = glcm_calc.getIdFeatureValue()
print(f"18: {x[0]}")
x = glcm_calc.getIdnFeatureValue()
print(f"19: {x[0]}")
x = glcm_calc.getInverseVarianceFeatureValue()
print(f"20: {x[0]}")
x = glcm_calc.getMaximumProbabilityFeatureValue()
print(f"21: {x[0]}")
x = glcm_calc.getSumAverageFeatureValue()
print(f"22: {x[0]}")
x = glcm_calc.getSumEntropyFeatureValue()
print(f"23: {x[0]}")
x = glcm_calc.getSumSquaresFeatureValue()
print(f"24: {x[0]}")