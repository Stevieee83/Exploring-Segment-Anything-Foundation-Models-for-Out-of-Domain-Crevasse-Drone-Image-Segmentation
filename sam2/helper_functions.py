import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import torchvision.transforms as transforms

# Defines the HelperFunctions object
class HelperFunctions():

    # Helper function to show the mask output
    def show_mask(self, mask, ax, random_color=False):

        # If the random_colour equals True display a random colour on the segmentation mask
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:

            # Displays the colour blue on the segmentation mask
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        # Stores the height and width of the segmentation mask
        h, w = mask.shape[-2:]

        # Combines the input mask with the segmentation colour and the image to display to the screen
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # Displays the segmentation mask to teh screen
        ax.imshow(mask_image)


    # Helper function to show the points of the prompt
    def show_points(self, coords, labels, ax, marker_size=375):

        # Stores the positive and negative co-ordinates
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]

        # Adds the positive (foreground prompt) marker to the output segmentation mask as a green star
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                    linewidth=1.25)

        # Adds the negative (background prompt) marker to the output segmentation mask as a green star
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                    linewidth=1.25)

    # Prepares the segemntation mask to be displayed over the top of the input image
    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)

    # Python helper function to convert a NumPy array to a Pytorch tensor
    def to_tensor(self, mask, gt):

        # Define a transform to convert the NumPy array segmentation masks to PyTorch tensors
        transform = transforms.ToTensor()

        # Apply the transform to the PyTorch tensors
        tensor_mask = transform(mask)
        tensor_gt = transform(gt)

        # Returns the mask and ground truth as a Pytorch tensor
        return tensor_mask, tensor_gt

    # Python method to convert values in a NumPy array
    def post_process_mask(self, old_value, new_value, mask):

        # Replace all occurrences of old_value with new_value
        mask[mask == old_value] = new_value

        return mask