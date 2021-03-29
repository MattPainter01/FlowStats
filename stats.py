import numpy as np
import matplotlib.pyplot as plt


class FlowStats(object):
    def __init__(self, pred_flow, true_flow):
        """ Class to compute end point error (EPE), Angular error and Fl score for flow fields.

        :param pred_flow: Flow field estimate
        :param true_flow: True flow field
        """
        super(FlowStats, self).__init__()
        self.pred = pred_flow
        self.true = true_flow

    def get_epe(self, flow, keepdims=False):
        """Method to get the end point errors of a given flow field.

        :param flow: Flow field to get EPE for
        :param keepdims: If True keep the (x,y) dimension
        :return: EPE for each pixel
        """
        return np.linalg.norm(flow, 2, 1, keepdims=keepdims)

    def get_epe_diff(self):
        """Method to get the EPE for the difference of the given predicted and true flow fields
        :return: EPE of pred - true for each pixel
        """
        return self.get_epe(self.pred - self.true)

    def get_angle(self, flow):
        """Method to get the angles of vectors in a flow field

        :param flow: Flow field to get angles of
        :return: Angles for each pixel
        """
        unit_flow = np.reshape(flow / self.get_epe(flow, True), (-1, 2))
        imaj = unit_flow[:, 1]*1.0j
        flow_comp = unit_flow[:,0]+imaj
        angles = np.angle(flow_comp)
        if len(flow.shape) > 2:
            angles = angles.reshape(flow.shape[0], *flow.shape[2:])
        return angles

    def get_angle_difference(self):
        """Method to get the angular difference between given predicted and true flow fields

        :return: Angular difference for each pixel
        """
        unit_pred = np.reshape(self.pred / self.get_epe(self.pred, True), (-1, 2))
        unit_true = np.reshape(self.true / self.get_epe(self.true, True), (-1, 2))
        dot_prod = np.einsum('ij,ij->i', unit_pred, unit_true)
        angles = np.arccos(np.clip(dot_prod, -1.0, 1.0))
        if len(self.pred.shape) > 2:
            angles = angles.reshape(self.pred.shape[0], self.pred.shape[1])
        return angles

    def get_fl_score(self, pix=3, perc=5, per_img=False):
        """Method to get the Fl score for predicted flow field. Defined as the ratio of pixels where flow estimate is
        wrong by both >= pix pixels and >= perc percent

        :param pix: Number of pixels estimate has to be wrong by to count
        :param perc: Percentage of target size that estimate has to be wrong by to count
        :param per_img: If True, Fl score is returned for each image in the 0th dimension
        :return: Ratio of bad pixels in estimate to good pixels in estimate
        """
        flat_true = self.true.reshape(-1, 2)
        size_true = np.einsum('ij,ij->i', flat_true, flat_true)
        if len(self.true.shape) > 2:
            size_true = size_true.reshape(self.true.shape[0], *self.true.shape[2:])

        diff = self.true - self.pred
        diff = np.sqrt(diff[:,0]**2+diff[:,1]**2)
        gepix = diff >= pix

        percentage = size_true / 100 * perc
        geperc = diff >= percentage

        if not per_img:
            bad = np.sum(gepix & geperc)
            num_pix = np.prod(diff.shape)
        else:
            bad = np.sum(gepix & geperc, tuple(range(1, len(gepix.shape))))
            num_pix = np.prod(diff.shape[1:])

        return float(bad) / num_pix

    def epe_angle_analysis(self, num_bins=10):
        """Method to get the mean and std of epe against angle of target. Useful to see if estimate is biased
        towards directions

        :param num_bins: Number of bins for the target angle
        :return (tuple): (Mean epe, std of epe, bin mid-points)
        """
        angles = self.get_angle(self.true)
        epe = self.get_epe_diff()

        min, max = angles.min(), angles.max()
        bins = np.linspace(min, max, num_bins)
        bin_width = float(bins[1]-bins[0]) / 2

        means = np.zeros_like(bins)
        stds = np.zeros_like(bins)
        for i, b in enumerate(bins):
            inds = np.where((angles >= b-bin_width) & (angles <= b+bin_width))
            mask_epe = epe[inds]
            means[i] = mask_epe.mean() if mask_epe.shape != (0,) else 0
            stds[i] = mask_epe.std() if mask_epe.shape != (0, ) else 0
        return means, stds, bins

    def epe_norm_analysis(self, num_bins=10):
        """Method to get the mean and std of epe against norm of target. Useful to see if estimate is biased
        towards large or small motions

        :param num_bins: Number of bins for the target angle
        :return (tuple): (Mean epe, std of epe, bin mid-points)
        """
        epe = self.get_epe_diff()
        norms = np.linalg.norm(self.true, 2, 1)

        min, max = norms.min(), norms.max()
        bins = np.linspace(min, max, num_bins)
        bin_width = (bins[1]-bins[0]) / 2

        means = np.zeros_like(bins)
        stds = np.zeros_like(bins)
        for i, b in enumerate(bins):
            inds = np.where((norms >= b-bin_width) & (norms <= b+bin_width))
            mask_epe = epe[inds]
            means[i] = mask_epe.mean() if mask_epe.shape != (0,) else 0
            stds[i] = mask_epe.std() if mask_epe.shape != (0, ) else 0
        return means, stds, bins

    def plot_analysis(self, means, stds, bins):
        """Method to plot to pyplot an error bar plot of output from analysis methods. Plt.plot() should be called after
        this to display the plot.

        :param means: Means for each bin
        :param stds: Standard deviations for each bin
        :param bins: Bin mid-points
        """
        plt.errorbar(bins, means, stds)

    def table_stats(self, pix=3, percent=5):
        """Method returns dictionary containing mean of epe, angular difference and Fl score.

        :param pix: Number of pixels estimate has to be wrong by to count
        :param percent: Percentage of target size that estimate has to be wrong by to count
        :return:
        """
        epe = np.mean(self.get_epe_diff())
        angles = np.mean(self.get_angle_difference())
        fl = self.get_fl_score(pix, percent, per_img=False)
        return {'epe': epe, 'angular_diff': angles, 'Fl': fl}
