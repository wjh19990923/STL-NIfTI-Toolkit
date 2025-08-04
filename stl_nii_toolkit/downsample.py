from nilearn.image import resample_img, image, crop_img, math_img
import pylab as plt
import nibabel as nb
import numpy as np

niiFilePath = r"C:\Users\ETH\source\dupla\dupla_renderers\test_files\cr_tib_modular_3_r_narrow_HU10000_filled.nii"
orig_nii = nb.load(niiFilePath)
orig_nii2 = orig_nii
print(orig_nii.header)
print(orig_nii.shape)
print(orig_nii.header.get_zooms())
print(np.min(orig_nii.dataobj[:, :, 0]))
print(np.max(orig_nii.dataobj[:, :, 0]))

masked_img = math_img("img * (img > -500)", img=orig_nii, copy_header_from="img")
orig_nii = crop_img(masked_img, copy_header=True)

scalingFactor = 2
affine = orig_nii.affine.copy()
affine[0, 0] = affine[0, 0] * scalingFactor
affine[1, 1] = affine[1, 1] * scalingFactor
affine[2, 2] = affine[2, 2] * scalingFactor

downsampled_nii = resample_img(orig_nii,
                               target_affine=affine,
                               interpolation='continuous', copy=True, copy_header=True)


def crop_img2(nii):
    startIndex = -1
    endIndex = -1
    for i in range(0, nii.shape[2]):
        min_val = np.min(nii.dataobj[:, :, i])
        max_val = np.max(nii.dataobj[:, :, i])
        print("Slice", i, "min =", min_val, "max =", max_val)
        if startIndex < 0 and max_val > 0:
            startIndex = i
        if startIndex >= 0 and max_val <= 0:
            endIndex = i
            break

    if nii.ndim == 4:
        cropped_img = image.index_img(nii, slice(startIndex, endIndex))
    else:
        data = nii.get_fdata()
        cropped_img = nb.Nifti1Image(data[:, :, startIndex:endIndex], nii.affine, nii.header)
    return cropped_img


cropped_img = crop_img2(downsampled_nii)
cropped_orig = crop_img2(orig_nii)
cropped_orig2 = crop_img2(orig_nii2)


def scale_img(img):
    img = img.astype(np.float32)
    img = img - np.min(img)  # Normalize to [0, max]
    img = img / np.max(img)  # Normalize to [0, 1]
    img = img * 255  # Scale to [0, 255]
    return img.astype(np.uint8)  # Convert to uint8


fig, ax = plt.subplots(1, 3)
myAx0 = ax[0].imshow(np.ones((cropped_img.shape[0], cropped_img.shape[1])), cmap='gray', vmin=0, vmax=255)
myAx1 = ax[1].imshow(np.ones((cropped_orig.shape[0], cropped_orig.shape[1])), cmap='gray', vmin=0, vmax=255)
myAx2 = ax[2].imshow(np.ones((cropped_orig2.shape[0], cropped_orig2.shape[1])), cmap='gray', vmin=0, vmax=255)
for i in range(0, cropped_img.shape[2]):
    min_val = np.min(cropped_img.dataobj[:, :, i])
    max_val = np.max(cropped_img.dataobj[:, :, i])
    if max_val < 1:
        print(f"Slice {i}: min = {min_val}, max = {max_val} (skipping)")
        continue
    print(f"Slice {i}: min = {min_val}, max = {max_val}")

    img2Show = scale_img(cropped_img.dataobj[:, :, i])

    # cropped_img.dataobj[:, :, i] = img2Show  # Update the data object with the new image
    myAx0.set_data(img2Show)
    if i*scalingFactor < cropped_orig.shape[2]:
        orig_img2Show = scale_img(cropped_orig.dataobj[:, :, i*scalingFactor])
        myAx1.set_data(orig_img2Show)
        myAx2.set_data(scale_img(cropped_orig2.dataobj[:, :, i*scalingFactor]))
    else:
        myAx1.set_data(np.zeros((cropped_orig.shape[0], cropped_orig.shape[1])))
        myAx2.set_data(np.zeros((cropped_orig2.shape[0], cropped_orig2.shape[1])))
    fig.canvas.draw()
    plt.pause(0.1)  # Pause to allow the image to be displayed

data = cropped_img.get_fdata()
# substitute <= 0 with -1024
num_zero_voxels = np.sum(data <= 0)
data[data <= 0.1] = -1024
print(f"Replaced {num_zero_voxels} voxels with value 0 to -1024.")
cropped_img = nb.Nifti1Image(data, affine=cropped_img.affine, header=cropped_img.header)

resampledNiiFilePath = str.replace(niiFilePath, ".nii", "_resampled.nii")
cropped_img.to_filename(resampledNiiFilePath)

# read again and check the result
orig_nii2 = nb.load(resampledNiiFilePath)
print(orig_nii2.header)
print(orig_nii2.shape)
print(orig_nii2.header.get_zooms())
print(np.min(orig_nii2.dataobj[:, :, 0]))
print(np.max(orig_nii2.dataobj[:, :, 0]))
