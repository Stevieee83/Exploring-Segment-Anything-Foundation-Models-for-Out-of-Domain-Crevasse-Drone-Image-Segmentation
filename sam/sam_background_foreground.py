import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

from segment_anything import sam_model_registry, SamPredictor

from helper_functions import HelperFunctions
from metrics import Metrics

import torch

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument("--file_number", type=int, default=1)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--multi_mask", type=bool, default=False)
parser.add_argument("--sam_checkpoint", type=str, default='./checkpoints/sam_vit_h_4b8939.pth')
parser.add_argument("--model_type", type=str, default='vit_h')
# ------------------------------------------------------------------------

# Main Python method for the program
def main():

    # Input prompt points and image number to input to the SAM model
    input_point = np.array([[161.3008658, 92.86796537], [47.70779221, 192.60822511]])       # Image 1 promps
    #input_point = np.array([[803.05374098, 713.20155388], [952.95637196, 671.94394902]])    # Image 2 promps
    #input_point = np.array([[489.49594408, 522.04131804], [611.89350515, 517.91555756]])    # Image 3 promps
    #input_point = np.array([[335.46755261, 802.59303106], [233.69879397, 743.45713077]])    # Image 4 promps
    #input_point = np.array([[510.1247465, 513.78979707], [620.14502612, 343.25836366]])     # Image 5 promps
    #input_point = np.array([[257.07810339,  69.58291812], [331.34179213, 132.8445789 ]])    # Image 6 promps
    #input_point = np.array([[533.50405592, 710.45104688], [698.53447535, 541.29486697]])    # Image 7 promps
    #input_point = np.array([[489.49594408, 931.86685961], [617.39451913, 930.49160612]])    # Image 8 promps
    #input_point = np.array([[702.66023583, 720.07782135], [578.88742127, 777.83846815]])    # Image 9 promps
    #input_point = np.array([[459.24036718, 695.32325844], [592.63995622, 633.43685115]])     # Image 10 promps

    # Input prompt labels to the SAM model
    input_label = np.array([1, 0])

    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()

    # File path directory for the input images
    dir_images=f'./data/images/borebreen_crop_drone_{str(args.file_number)}.png'        # Input image file path directory
    path = f'./results/multi_mask_single_point/{args.model_type}/borebreen_image_{str(args.file_number)}_1BG_1FG/'  # Creates the file path for the output results
    output_mask_1 = f"borebreen_image_{str(args.file_number)}_mask_1.png"               # Output segmentation mask file path directory
    output_mask_2 = f"borebreen_image_{str(args.file_number)}_mask_2.png"               # Output segmentation mask file path directory
    output_mask_3 = f"borebreen_image_{str(args.file_number)}_mask_3.png"               # Output segmentation mask file path directory
    
    # Segmentation metrics filepath directories
    gt_name = f'./data/masks/borebreen_crop_drone_{str(args.file_number)}.png'           # Image ground truth masks file path directory
    output_path = f'./results/single_mask_single_point/{args.model_type}/borebreen_image_{str(args.file_number)}_1BG_1FG/output_masks/' # Output filepath directory for the segmentation mask results
    # ------------------------------------------------------------------------

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
    
    # Defiens the HelperFunctions object
    helper = HelperFunctions()
    
    # Load the dataset
    print('\nLoading image to the SAM model')
    
    # Loads the dataset
    image = cv2.imread(dir_images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Appends the file path directory for the SAM model
    sys.path.append("..")
    
    # Prints out the device being used to the screen
    print("\nViT: ", args.model_type)
    print("Device: ", device)
    
    # Stores the SAM model in the sam variable
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    
    # Sends the SAM model to the device (GPU or CPU)
    sam.to(device=device)
    
    # Creates the SAM object and stores it in the predictor variable
    predictor = SamPredictor(sam)
    
    # Processes the image through the SAM model
    predictor.set_image(image)
    
    # Stores the masks, scores and the logits from the predict SAM method
    masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=args.multi_mask,
    )

    ######################################## Multi Mask 1 ##########################################
    if args.multi_mask:

        # Calls the makedirs Python OS module function
        os.makedirs(path, exist_ok=True)

        # Stores the output masks from the SAM model inside a Python list
        out_masks = []

        # Iterates through the SAM model predictions to output images to the screen as a figure plot
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            helper.show_mask(mask, plt.gca())
            helper.show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('on')
            plt.savefig(path + 'output_image_' + str(i + 1) + '.png')
            out_masks.append(mask)
            plt.show()

        # Prints out the number of output masks saved to the out_masks Python list
        print("\nNumber of Output Masks: ", len(out_masks))

        # Displays the first output mask from the SAM model and saves to a plot figure and an image
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(out_masks[0])
        plt.axis('off')
        plt.savefig(path + 'mask_1.png')
        segmented = out_masks[0].astype(int)
        cv2.imwrite(path + output_mask_1, segmented)
        plt.show()

        # Setting values to rows and column variables
        rows = 1
        columns = 3

        # Creates the subplot figure
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(image)
        plt.axis('off')

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)

        # Segmentation mask subplot
        plt.imshow(out_masks[0])
        plt.axis('off')

        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)

        # Segmentation mask over image subplot
        image0 = cv2.imread(path + 'output_image_1.png')
        image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        plt.imshow(image0)
        plt.axis('off')

        # Saves the Matplotlib figure to the outputs file directory
        plt.savefig(path + 'output_image_all.png')

        # Displays the subplot figure to the screen
        plt.show()

        # Prints got to the end of the prediction iteration to the screen
        print('\nMask 1')
        print("SAM Model Predictions Complete")

        # Defines the Metrics and HelperFunctions Python objects
        metric = Metrics()

        # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
        image = cv2.imread(dir_images)
        mask = cv2.imread(path + output_mask_1, 0)
        gt = cv2.imread(gt_name, 0)

        # Displays the input image
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        plt.show()

        # Displays the segmentation mask output from the model
        plt.imshow(mask)
        plt.title('Model Segmentation Mask')
        plt.axis('off')
        plt.show()

        # Displays the ground truth segmentation mask
        plt.imshow(gt)
        plt.title('Ground Truth Mask')
        plt.axis('off')
        plt.show()

        # Displays the segmentation mask output from the model
        plt.imshow(mask)
        plt.title('Model Segmentation Mask')
        plt.axis('off')
        plt.show()

        # Displays the ground truth segmentation mask
        plt.imshow(gt)
        plt.title('Ground Truth Mask')
        plt.axis('off')
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
        iou = metric.iou(tensor_mask, tensor_gt.long())
        ds = metric.dsc(tensor_mask, tensor_gt.long())

        # Prints out the segmentation metric results to the screen
        print('Dice Score Coefficient (DSC): ', ds.item())
        print('Intersection Over Union (IoU): ', iou.item())

        # Display the model segmentation mask on the screen and save it as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(mask)
        plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
        plt.axis('off')
        plt.savefig(path + 'mask_only.png')
        plt.show()

        # Display the ground truth mask on the screen and save it as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(gt)
        plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
        plt.savefig(path + 'gt_only.png')
        plt.show()

        ######################################## Multi Mask 2 ##########################################
        print("\nMask 2")
    
        # Displays the first output mask from the SAM model and saves to a plot figure and an image
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(out_masks[1])
        plt.axis('off')
        plt.savefig(path + 'mask_2.png')
        segmented = out_masks[1].astype(int)
        cv2.imwrite(path + output_mask_2, segmented)
        plt.show()
    
        # Creates the sublot figure
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(image)
        plt.axis('off')
    
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
    
        # Segmentation mask subplot
        plt.imshow(out_masks[1])
        plt.axis('off')
    
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)
    
        # Segmentation mask over image subplot
        image0 = cv2.imread(path + 'output_image_2.png')
        image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        plt.imshow(image0)
        plt.axis('off')
    
        # Saves the Matplotlib figure to the outputs file directory
        plt.savefig(path + 'output_image_all_2.png')
    
        # Displays the subplot figure to the screen
        plt.show()
    
        # Prints got to the end of the prediction iteration to the screen
        print("SAM Model Predictions Complete")
    
        # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
        image = cv2.imread(dir_images)
        mask = cv2.imread(path + output_mask_2, 0)
        gt = cv2.imread(gt_name, 0)
    
        # Stores the mask and ground truth as a Pytorch tensor from teh to_tensor helper function
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
        iou = metric.iou(tensor_mask, tensor_gt.long())
        ds = metric.dsc(tensor_mask, tensor_gt.long())
    
        # Prints out the segmentation metric results to the screen
        print('Dice Score Coefficient (DSC): ', ds.item())
        print('Intersection Over Union (IoU): ', iou.item())
    
        # Display the ground truth mask to the screen and saves as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(mask)
        plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
        plt.axis('off')
        plt.savefig(path + 'mask_only_2.png')
        plt.show()
    
        # Displays the segmentation mask on the screen and saves it as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(gt)
        plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
        plt.savefig(path + 'gt_only_2.png')
        plt.show()
    
        ######################################## Multi Mask 3 ##########################################
        print("\nMask 3")
    
        # Displays the first output mask from the SAM model and saves to a plot figure and an image
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(out_masks[2])
        plt.axis('off')
        plt.savefig(path + 'mask_3.png')
        segmented = out_masks[2].astype(int)
        cv2.imwrite(path + output_mask_3, segmented)
        plt.show()
    
        # Creates the sublot figure
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(image)
        plt.axis('off')
    
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
    
        # Segmentation mask subplot
        plt.imshow(out_masks[2])
        plt.axis('off')
    
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)
    
        # Segmentation mask over image subplot
        image0 = cv2.imread(path + 'output_image_3.png')
        image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        plt.imshow(image0)
        plt.axis('off')
    
        # Saves the Matplotlib figure to the outputs file directory
        plt.savefig(path + 'output_image_all_3.png')
    
        # Displays the subplot figure to the screen
        plt.show()
    
        # Prints got to the end of the prediction iteration to the screen
        print("SAM Model Predictions Complete")
    
        # Defines the Metrics and HelperFunctions Python objects
        metric = Metrics()
    
        # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
        image = cv2.imread(dir_images)
        mask = cv2.imread(path + output_mask_3, 0)
        gt = cv2.imread(gt_name, 0)
    
        # Stores the mask and ground truth as a Pytorch tensor from teh to_tensor helper function
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
        iou = metric.iou(tensor_mask, tensor_gt.long())
        ds = metric.dsc(tensor_mask, tensor_gt.long())
    
        # Prints out the segmentation metric results to the screen
        print('Dice Score Coefficient (DSC): ', ds.item())
        print('Intersection Over Union (IoU): ', iou.item())
    
        # Display the ground truth mask to the screen and saves as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(mask)
        plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
        plt.axis('off')
        plt.savefig(path + 'mask_only_3.png')
        plt.show()
    
        # Displays the segmentation mask on the screen and saves it as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(gt)
        plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
        plt.savefig(path + 'gt_only_3.png')
        plt.show()

    else:

        # Calls the makedirs Python OS module function
        os.makedirs(output_path, exist_ok=True)

        # Stores the output masks from the SAM model inside a Python list
        out_masks = []

        # Iterates through the SAM model predictions to output images to the screen as a figure plot
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            helper.show_mask(mask, plt.gca())
            helper.show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('on')
            plt.savefig(output_path + 'output_image_' + str(i + 1) + '.png')
            out_masks.append(mask)
            plt.show()

        # Prints out the number of output masks saved to the out_masks Python list
        print("\nNumber of Output Masks: ", len(out_masks))

        # Displays the first output mask from the SAM model and saves to a plot figure and an image
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(out_masks[0])
        plt.axis('off')
        plt.savefig(output_path + 'mask_1.png')
        segmented = out_masks[0].astype(int)
        cv2.imwrite(output_path + output_mask_1, segmented)
        plt.show()

        # Setting values to rows and column variables
        rows = 1
        columns = 3

        # Creates the subplot figure
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(image)
        plt.axis('off')

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)

        # Segmentation mask subplot
        plt.imshow(out_masks[0])
        plt.axis('off')

        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)

        # Segmentation mask over image subplot
        image0 = cv2.imread(output_path + 'output_image_1.png')
        image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        plt.imshow(image0)
        plt.axis('off')

        # Saves the Matplotlib figure to the outputs file directory
        plt.savefig(output_path + 'output_image_all.png')

        # Displays the subplot figure to the screen
        plt.show()

        # Prints got to the end of the prediction iteration to the screen
        print('\nMask 1')
        print("SAM Model Predictions Complete")

        # Defines the Metrics and HelperFunctions Python objects
        metric = Metrics()

        # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
        image = cv2.imread(dir_images)
        mask = cv2.imread(output_path + output_mask_1, 0)
        gt = cv2.imread(gt_name, 0)

        # Displays the input image
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        plt.show()

        # Displays the segmentation mask output from the model
        plt.imshow(mask)
        plt.title('Model Segmentation Mask')
        plt.axis('off')
        plt.show()

        # Displays the ground truth segmentation mask
        plt.imshow(gt)
        plt.title('Ground Truth Mask')
        plt.axis('off')
        plt.show()

        # Displays the segmentation mask output from the model
        plt.imshow(mask)
        plt.title('Model Segmentation Mask')
        plt.axis('off')
        plt.show()

        # Displays the ground truth segmentation mask
        plt.imshow(gt)
        plt.title('Ground Truth Mask')
        plt.axis('off')
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
        iou = metric.iou(tensor_mask, tensor_gt.long())
        ds = metric.dsc(tensor_mask, tensor_gt.long())

        # Prints out the segmentation metric results to the screen
        print('Dice Score Coefficient (DSC): ', ds.item())
        print('Intersection Over Union (IoU): ', iou.item())

        # Display the model segmentation mask on the screen and save it as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size, figure_size))
        plt.imshow(mask)
        plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
        plt.axis('off')
        plt.savefig(output_path + 'mask_only.png')
        plt.show()

        # Display the ground truth mask on the screen and save it as an image
        figure_size = 10.24
        plt.figure(figsize=(figure_size, figure_size))
        plt.imshow(gt)
        plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
        plt.savefig(output_path + 'gt_only.png')
        plt.show()

# Executes the main method from the main.py Python script
if __name__ == '__main__':
    # Calls the main function for finetuning the SAM model
    main()