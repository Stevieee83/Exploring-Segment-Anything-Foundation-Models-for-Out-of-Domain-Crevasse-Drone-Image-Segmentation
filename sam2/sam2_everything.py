import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

from metrics import Metrics
from helper_functions import HelperFunctions

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from skimage.io import imread
from skimage.transform import resize

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument("--file_number", type=int, default=1)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--sam2_checkpoint", type=str, default='./checkpoints/sam2_hiera_large.pt')
parser.add_argument("--model_cfg", type=str, default='sam2_hiera_l.yaml')
parser.add_argument("--model_name", type=str, default='ViT-L-Hiera')
# ------------------------------------------------------------------------

# Main Python method for the program
def main():

    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()

    ground_truth = 'gt.png'                                                                         # Output ground truth segemtnaiotn mask
    ground_truth_plot = 'gt_plot.png'                                                               # Output ground truth segemtnaiotn mask as a plot
    combined_mask = f'borebreen_crop_drone_{str(args.file_number)}.png'                             # Combined final output mask
    combined_mask_plot = f'borebreen_crop_drone_{str(args.file_number)}_plot.png'                   # Combined final output mask as a plot
    img_ground_truth_plot = f'borebreen_crop_drone_{str(args.file_number)}_img_gt.png'              # Combined final output ground truth over input image
    img_mask_plot = f'borebreen_crop_drone_{str(args.file_number)}_img_mask.png'                    # Combined final output mask over input image
    # --------------------------------------------------------------------------
    
    # Input and output file path directoruies
    dir_image = f'./data/images/borebreen_crop_drone_{str(args.file_number)}.png'                   # Input image file path directory
    gt_name = f'./data/masks/borebreen_crop_drone_{str(args.file_number)}.png'                      # Ground truth file path directory
    output_dir = f'./results/everything/{args.model_name}/borebreen_image_{str(args.file_number)}_Everything/'   # Creates the file path for the output results
    output_file_1 = 'instances/'                                                                    # Output directory 1
    output_file_2 = 'instances_0_1/'                                                                # Output directory 2
    output_file_3 = 'end_of_script/'                                                                # Output directory 3
    input_image_save = f'image_{str(args.file_number)}.png'                                         # Saves plot 1 as an image
    image_mask = f'mask_{str(args.file_number)}.png'                                                # Saves plot 2 as an image
    # --------------------------------------------------------------------------

    # Use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Sets the device and the random seeds
    if torch.cuda.is_available():
        # Sets the device to CUDA GPU
        device = 'cuda'
        # Set random seed for GPU
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    else:
        # Sets the device to CUDA GPU
        device = 'cpu'
        # Set random seed for CPU
        torch.cuda.manual_seed(42)
    
    # Defines the Metrics and HelperFunctions Python objects
    metric = Metrics()
    helper = HelperFunctions()
    
    # Reads in the input image to the runtime
    image = cv2.imread(dir_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prints out the set image/mask height and width to the screen
    print("\nImage/mask width set: ", args.width)
    print("Image/mask height set: ", args.height)

    fig_width= args.width / 100      # Sets the width of Matplotlig figures
    fig_height = args.height / 100   # Sets the height of Matplotlig figures
    
    # Plots the imput image
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_dir + input_image_save)
    plt.show()
    
    # Builds the SAM 2 model
    sam2 = build_sam2(args.model_cfg, args.sam2_checkpoint, device ='cuda', apply_postprocessing=False)
    
    # Inputs the parameters and calls the SAM2AutomaticMaskGenerator
    mask_generator = SAM2AutomaticMaskGenerator(sam2,
                                                points_per_side=32,
                                                pred_iou_thresh=0.86,
                                                stability_score_thresh=0.92,
                                                crop_n_layers=1,
                                                crop_n_points_downscale_factor=2)
    
    # Generates the segemntaiton mask from the SAM 2 model
    masks = mask_generator.generate(image)
    
    # Prints out the length of the masks dictionary to the screen and the number of keys at index 0
    print('')
    print(len(masks), 'Instances generated by the Mask Generator module')
    
    # Saves the image with the segmentation instances over the top of the image
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(image)
    helper.show_anns(masks)
    plt.axis('off')
    plt.savefig(output_dir + image_mask)
    plt.show()
    
    # Stores the mask instances in a list
    masks_list = []
    
    # Key to be removed
    key_to_remove = 'segmentation'
    
    # Iterate through each dictionary in the list
    for dictionary in masks:
        # Remove the key if it exists in the current dictionary
        if key_to_remove in dictionary:
            masks_list.append(dictionary[key_to_remove])
    
    # Iterate through each array in the list and convert it to an integer array
    list_of_integer_arrays = [arr.astype(int) for arr in masks_list]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir + output_file_1, exist_ok=True)
    
    # Iterate through each array in the list
    for idx, arr in enumerate(list_of_integer_arrays):
        # Convert the boolean array to an integer array (0 and 255 for binary image)
        int_array = (arr.astype(np.uint8)) * 255
    
        # Define the output file path
        output_path = os.path.join(output_dir + output_file_1, f'borebreen_image_1_{idx + 1}.png')
    
        # Save the array as an image using OpenCV
        cv2.imwrite(output_path, int_array)
    
    print(f"\nImages saved to {output_dir + output_file_1}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir + output_file_2, exist_ok=True)
    
    # Iterate through each array in the list
    for idx, arr in enumerate(list_of_integer_arrays):
    
        # Define the output file path
        output_path = os.path.join(output_dir + output_file_2, f'borebreen_image_1_{idx + 1}.png')
    
        # Save the array as an image using OpenCV
        cv2.imwrite(output_path, arr)
    
    print(f"Images saved to {output_dir + output_file_2}")
    
    # Defines a 2D NumPy array of zeros to store the combined segemntation masks
    mask = np.zeros((args.height, args.width, 1), dtype=bool)
    
    # Loads the training labels to the runtime
    for mask_file in next(os.walk(output_dir + output_file_2))[2]:
        mask_ = imread(output_dir + output_file_2 + mask_file)
        mask_ = np.expand_dims(resize(mask_, (args.height, args.width), mode='constant',
                                          preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    
    # Displays the combined segemtnaiton mask to the screen
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mask)
    plt.axis('off')
    plt.show()
    
    # Outputs the combined segemntaiton mask to the screen
    cv2.imwrite(output_dir + combined_mask, mask)
    
    # Reads back in the combined segmentaiton mask to the screen
    cv2.imread(output_dir + combined_mask, 0)
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mask)
    plt.axis('off')
    plt.show()
    
    # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
    gt = cv2.imread(gt_name, 0)

    # Converts the values of the SAM 2 model output segmenation mask
    old_value = 1
    new_value = 2
    
    helper.post_process_mask(old_value, new_value, mask)
    
    old_value = 0
    new_value = 1
    
    helper.post_process_mask(old_value, new_value, mask)
    
    old_value = 2
    new_value = 0
    
    helper.post_process_mask(old_value, new_value, mask)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir + output_file_3, exist_ok=True)
    
    # Displays the converted SAM 2 model segemtnaiton mask to the screen
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(output_dir + output_file_3 + combined_mask_plot)
    plt.show()
    
    # Saves the output mask to the output mask file path directory
    cv2.imwrite(output_dir + output_file_3 + combined_mask, mask)
    
    # Reads back in the output segemtnaiotn mask to the runtime
    mask = cv2.imread(output_dir + output_file_3 + combined_mask, 0)
    
    # Outputs the segemtniaotn mask over the input image to the screen
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(image)
    helper.show_mask(mask, plt.gca())
    plt.axis('off')
    plt.savefig(output_dir + output_file_3 + img_mask_plot)
    plt.show()
    
    # Displays the converted ground truth mask to the screen
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(gt)
    plt.axis('off')
    plt.savefig(output_dir + output_file_3 + ground_truth_plot)
    plt.show()
    
    # Saves teh output mask to the output mask file path directory
    cv2.imwrite(output_dir + output_file_3 + ground_truth, gt)
    
    # Reads back in the output segemtnaiotn mask to the runtime
    gt = cv2.imread(output_dir + output_file_3 + ground_truth, 0)
    
    # Outputs the segemtniaotn mask over the input image to the screen
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(image)
    helper.show_mask(gt, plt.gca())
    plt.axis('off')
    plt.savefig(output_dir + output_file_3 + img_ground_truth_plot)
    plt.show()
    
    # Stores the mask and ground truth as a Pytorch tensor from the to_tensor helper function
    tensor_mask, tensor_gt = helper.to_tensor(mask, gt)
    
    # Reshape the PyTroch tensors
    tensor_mask = tensor_mask.reshape(1, 1, args.width, args.height)
    tensor_gt = tensor_gt.reshape(1, 1, args.width, args.height)
    
    # Find maximum value and its index and replaces with the new value
    mask_max_value, mask_max_index = tensor_mask.max(), tensor_mask.argmax()
    new_value = 1
    
    # Replace the maximum value in the tensor
    tensor_mask[tensor_mask == mask_max_value] = new_value
    
    # Find the maximum pixel value and its index
    gt_max_value, gt_max_index = tensor_gt.max(), tensor_gt.argmax()
    new_value = 1
    
    # Replace the maximum pixel value in the tensor
    tensor_gt[tensor_gt == gt_max_value] = new_value
    
    # Segemntation metric calculations
    ds = metric.dsc(tensor_mask, tensor_gt.long())
    iou = metric.iou(tensor_mask, tensor_gt.long())
    
    # Prints out the segmentation metric results to the screen
    print('\nDice Score Coefficient: ',ds.item())
    print('Intersection Over Union: ', iou.item())

# Executes the main method from the main.py Python script
if __name__ == '__main__':
    # Calls the main function for the SAM 2 model script
    main()
