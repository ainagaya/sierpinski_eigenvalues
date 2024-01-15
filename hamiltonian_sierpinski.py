import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import ndimage
from scipy.sparse import lil_matrix
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, help='Value of k')
args = parser.parse_args()

k = args.k

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Function to read potential matrix from a file
def read_potential_matrix(file_path):
    return np.loadtxt(file_path)

def generate_hamiltonian_matrix(potential_matrix):

    num_points_orig = potential_matrix.shape[0]

    factor = output_shape / num_points_orig + 1

    # Resize the potential matrix directly
    print("Resizing...")
    potential_matrix = resize(potential_matrix, (output_shape, output_shape), order=0)

    # Dirichlet
    potential_matrix = np.pad(potential_matrix, pad_width=1, constant_values=10**8)

    # Create a heatmap
    plt.imshow(potential_matrix, cmap='binary', interpolation='none')
    plt.tick_params(axis='both', which='major', labelsize=48)  # Increase the font size of the axis labels
    plt.tight_layout()
    # Show grid lines
    #plt.grid(True, which='both', color='black', linewidth=1.5, linestyle='-', alpha=0.5)

    # Show the plot
    plt.savefig("Sierpinski_Dirichlet_" + str(k) + "_" + str(output_shape) + ".eps")
    
    #plt.show()
    num_points = potential_matrix.shape[0]

    dx = 1.0 / (num_points - 1)

    hamiltonian_matrix = lil_matrix((num_points**2, num_points**2), dtype=complex)
    #hamiltonian_matrix = np.array((num_points**2, num_points**2), dtype=complex)

    for i in range(num_points):
        for j in range(num_points):

            #if potential_matrix[i, j] == 1:
             #   potential_matrix[i, j] = np.inf

            index = i * num_points + j

            # Diagonal elements
            hamiltonian_matrix[index, index] = 0.5 / (dx**2) * 4 + potential_matrix[i, j]

            # Off-diagonal elements (finite difference approximation of Laplacian)
            if i > 0:
                hamiltonian_matrix[index, index - num_points] = -0.5 / (dx**2) 
            if i < num_points - 1:
                hamiltonian_matrix[index, index + num_points] = -0.5 / (dx**2)
            if j > 0:
                hamiltonian_matrix[index, index - 1] = -0.5 / (dx**2) 
            if j < num_points - 1:
                hamiltonian_matrix[index, index + 1] = -0.5 / (dx**2) 

    #x = hamiltonian_matrix.toarray()

   # print(hamiltonian_matrix)
    return hamiltonian_matrix.tocsr()

# Read the matrix from a file
file_path = 'Sierpinski_' + str(k) + '.dat'  # Replace with your file path
print("File path: ", file_path)
output_shape = 3**4

# Read potential matrix from file
potential_matrix = read_potential_matrix(file_path)
#print(potential_matrix.astype(int))
num_points_orig = potential_matrix.shape[0]
k = math.log(num_points_orig, 3)

# Generate Hamiltonian matrix
hamiltonian_matrix = generate_hamiltonian_matrix(potential_matrix)

# Diagonalize the matrix
#eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix.toarray())
#eigenvalues, eigenvectors = sp.linalg.eigh(hamiltonian_matrix)

filename = "eigenvalues.dat"
#plt.imsave("Eigenvector_" + str(i+1) + "_" + str(k) + "_" + str(output_shape) + ".eps", np.abs(eigenvector)**2, cmap='GnBu')  # Provide a valid file path

# Plot the first few eigenvectors
num_eigenvectors_to_plot = 5

# Appending content to the file
with open(filename, 'a') as file:
    for i in range(num_eigenvectors_to_plot):
        file.write(str(k) + " " + str(output_shape) + " " + 
                   str(i) + " " + str(eigenvalues[i]) + "\n")

for i in range(num_eigenvectors_to_plot):
    eigenvector = eigenvectors[:, i].reshape((int(np.sqrt(len(eigenvectors[:, i]))), int(np.sqrt(len(eigenvectors[:, i])))))
    filename_eig = "Eigenvector_" + str(i+1) + "_" + str(k) + "_" + str(output_shape) + ".dat"
    with open(filename_eig, 'a') as file:
        np.savetxt(file, np.abs(eigenvector)**2)

    plt.imshow(np.abs(eigenvector)**2, extent=(0, 1, 0, 1), origin='lower', cmap='GnBu')
    plt.title("Eigenvector " + str(i+1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.imsave("Eigenvector_" + str(i+1) + "_" + str(k) + "_" + str(output_shape) + ".eps", np.abs(eigenvector)**2, cmap='GnBu')  # Provide a valid file path


from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools

# Set up a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_eigenvectors_to_plot):
     eigenvector = eigenvectors[:, i].reshape((int(np.sqrt(len(eigenvectors[:, i]))), int(np.sqrt(len(eigenvectors[:, i])))))

     # Generate 3D coordinates based on the matrix indices
     x = np.linspace(0, 1, eigenvector.shape[0])
     y = np.linspace(0, 1, eigenvector.shape[1])
     x, y = np.meshgrid(x, y)
     z = np.abs(eigenvector)**2
#     # Plot the surface
     ax.plot_surface(x, y, z, cmap='GnBu', edgecolors='none', linewidth=0.5)
     ax.set_title("Eigenvector {} (k={})".format(i+1, k))
     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.set_zlabel('Probability Density')
     ax.set_xlim(0, 1)
     ax.set_ylim(0, 1)
     ax.set_zlim(0, np.max(z))  # Adjust the z-axis limit as needed

#     # Save the plot as SVG
     plt.savefig("Eigenvector_3d_{}_{}_{}.eps".format(i+1, k, output_shape))

#     # Clear the plot for the next iteration
     ax.clear()

