from os.path import join

from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np

import config as c
import datasets

n_imgs = 4
n_plots = 2
figsize = (4,4)

class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.counter = 1

            header = 'Epoch'
            for l in loss_labels:
                header += '\t\t%s' % (l)

            self.config_str = ""
            self.config_str += "==="*30 + "\n"
            self.config_str += "Config options:\n\n"

            for v in dir(c):
                if v[0]=='_': continue
                s=eval('c.%s'%(v))
                self.config_str += "  {:25}\t{}\n".format(v,s)

            self.config_str += "==="*30 + "\n"

            print(self.config_str)
            print(header)

    def update_losses(self, losses, *args):
        print('\r', '    '*20, end='')
        line = '\r%.3i' % (self.counter)
        for l in losses:
            line += '\t\t%.4f' % (l)

        print(line)
        self.counter += 1

    def update_images(self, *img_list):
        w = img_list[0].shape[2]
        k = 0
        k_img = 0

        show_img = np.zeros((3, w*n_imgs, w*n_imgs), dtype=np.uint8)
        img_list_np = []
        for im in img_list:
            im_np = im
            img_list_np.append(np.clip((255. * im_np), 0, 255).astype(np.uint8))

        for i in range(n_imgs):
            for j in range(n_imgs):
                show_img[:, w*i:w*i+w, w*j:w*j+w] = img_list_np[k]

                k += 1
                if k >= len(img_list_np):
                    k = 0
                    k_img += 1

        plt.imsave(join(c.img_folder, '%.4d.jpg'%(self.counter)), show_img.transpose(1,2,0))
        return zoom(show_img, (1., c.preview_upscale, c.preview_upscale), order=0)

    def update_hist(self, *args):
        pass

    def update_running(self, *args):
        pass


visualizer = Visualizer(c.loss_names)

def show_loss(losses, logscale=False):
    visualizer.update_losses(losses)

def show_imgs(*imgs):
    visualizer.update_images(*imgs)

def show_hist(data):
    visualizer.update_hist(data.data)

def signal_start():
    visualizer.update_running(True)

def signal_stop():
    visualizer.update_running(False)

def close():
    visualizer.close()

