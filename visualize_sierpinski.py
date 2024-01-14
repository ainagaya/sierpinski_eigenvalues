import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, help='Value of k')
args = parser.parse_args()

k = args.k

# Read the matrix from a file
file_path = 'Sierpinski_{}.dat'.format(k)  # Replace with your file path
matrix = np.loadtxt(file_path)

# Remove the axes
plt.axis('off')

# Create a heatmap
plt.imshow(matrix, cmap='binary', interpolation='nearest')

# Show grid lines
#plt.grid(True, which='both', color='black', linewidth=1.5, linestyle='-', alpha=0.5)

# Show the plot
plt.savefig("Sierpinski_{}.png".format(k))
