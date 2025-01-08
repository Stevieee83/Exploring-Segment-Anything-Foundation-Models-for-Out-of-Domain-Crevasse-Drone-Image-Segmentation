import torch
import segmentation_models_pytorch as smp

# Python class to define the test metrics
class Metrics:

    # Python method to take the colur channels of the image
    def _take_channels(self, *xs, ignore_channels=None):
        if ignore_channels is None:
            return xs
        else:
            channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
            xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
            return xs

    # Pyhton method to set a threshold for evaluating the model
    def _threshold(self, x, threshold=None):
        if threshold is not None:
            return (x > threshold).type(x.dtype)
        else:
            return x

    # Pyhton method to calculate the Intersection Over Union(IoU)
    def iou(self, pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
        """Calculate Intersection over Union between ground truth and prediction
           Args:
              pr (torch.Tensor): predicted tensor
              gt (torch.Tensor):  ground truth tensor
              eps (float): epsilon to avoid zero division
              threshold: threshold for outputs binarization
            Returns:
              float: IoU (Jaccard) score
          """

        # Stores the threshold value in the pr Python variable
        pr = self._threshold(pr, threshold=threshold)
        pr, gt = self._take_channels(pr, gt, ignore_channels=ignore_channels)

        # Calculates the (IoU) score
        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + eps

        # Returns the calculated intersection over union score
        return (intersection + eps) / union

    # Python method to calculate the F1 score
    def dsc(self, pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):

        """Calculate Dice Score Coefficient between ground truth and prediction
          Args:
              pr (torch.Tensor): predicted tensor
              gt (torch.Tensor):  ground truth tensor
              beta (float): positive constant
              eps (float): epsilon to avoid zero division
              threshold: threshold for outputs binarization
          Returns:
              float: F score
        """

        # Stores the threshold value in the pr Python variable
        pr = self._threshold(pr, threshold=threshold)

        # Stores the channels of hte image in the pr and gt Python variables
        pr, gt = self._take_channels(pr, gt, ignore_channels=ignore_channels)

        # Calcualtes the true positives, false positives and the false negatives
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp

        # Calculates the F1 Score
        score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

        # Returns the F1 Score value
        return score
