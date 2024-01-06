import matplotlib.pyplot as plt
import numpy as np

# Read the matrix from a file
file_path = 'Sierpinski.dat'  # Replace with your file path
matrix = np.loadtxt(file_path)

# Create a heatmap
plt.imshow(matrix, cmap='binary', interpolation='nearest')

# Show grid lines
plt.grid(True, which='both', color='black', linewidth=1.5, linestyle='-', alpha=0.5)

# Show the plot
plt.show()
