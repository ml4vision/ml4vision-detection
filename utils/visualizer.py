import matplotlib
import matplotlib.pyplot as plt
from .image_utils import denormalize_img_tensor
from .colorizer import get_colors
from matplotlib.patches import Rectangle
from IPython import display as idisplay


class ObjectDetectionVisualizer:

    def __init__(self, im_norm=True):
        plt.ion()
        self.inline = 'inline' in matplotlib.get_backend()
        self.dh = None
        self.im_norm = im_norm
        self.colors = get_colors(float=True)[1:]
        self.window = plt.subplots(ncols=2)

    def display(self, image, pred, gt):
        fig, axs = self.window
        for ax in axs:
            ax.cla()
            ax.set_axis_off()

        if self.im_norm:
            image = image.squeeze_()
            image = denormalize_img_tensor(image)

        # display pred
        axs[0].imshow(image.numpy().transpose(1,2,0))
        for box in pred.numpy():
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=self.colors[int(box[4])], facecolor='none')
            axs[0].add_patch(rect)

        # display gt
        axs[1].imshow(image.numpy().transpose(1,2,0))
        for box in gt.numpy():
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=self.colors[int(box[4])], facecolor='none')
            axs[1].add_patch(rect)

        if self.inline: 
            if self.dh:
                self.dh.update(fig)
            else:
                self.dh = idisplay.display(fig, display_id=True)
        else:
            plt.draw()
            
        self.mypause(0.001)

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return




