import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import csv

font = {'family': 'serif',
        'weight': 'normal',
        'size': 18}

__author__ = 'Siavash Malektaji'
__license__ = "GPL"
__email__ = "siavashmalektaji@gmail.com"

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def parse_POCT_output_file(file_name: str, label: str=None):

    def parse_row():
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                parsed_row = list(map(lambda x: float(x) , row))
                yield parsed_row

    x_positions = np.array(list(sorted(set([x[0] for x in parse_row()]))))
    z_positions = np.array(list(sorted(set([x[1] for x in parse_row()]))))

    print('number of A-Scans:{}'.format(len(x_positions)))

    reflectances = np.array([x[2] for x in parse_row()])

    reflectance_grid = reflectances.reshape((x_positions.shape[0], z_positions.shape[0]))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    if label:
            fig.suptitle(label, **font)

    im = axes[1].imshow(reflectance_grid.transpose(), extent=(x_positions.min(), x_positions.max(), z_positions.max(), z_positions.min()), cmap='jet', interpolation='none')
    axes[1].set_xlabel('Distance X [cm]', **font)
    axes[1].set_ylabel('Depth Z [cm]', **font)
    axes[1].set_xticks([x_positions.min(), x_positions.max()])

    axes[0].imshow(reflectance_grid[:, :150].transpose(), extent=(x_positions.min(), x_positions.max(), z_positions.max(), z_positions.min()), cmap='jet', interpolation='none')
    axes[0].set_xlabel('Distance X [cm]', **font)
    axes[0].set_ylabel('Depth Z [cm]', **font)
    axes[0].set_aspect('auto')

    colorbar = fig.colorbar(im)
    colorbar.set_label(label='Reflectance', **font)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--OCTMPS_output_file", type=str, required=True, help="OCTMPS output file")
    parser.add_argument("--Label", type=str, required=False, help="Label for the figures")
    args = parser.parse_args()

    parse_POCT_output_file(args.OCTMPS_output_file, args.Label)

