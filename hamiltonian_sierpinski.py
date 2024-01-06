import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import ndimage

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Function to read potential matrix from a file
def read_potential_matrix(file_path):
    return np.loadtxt(file_path)

def generate_hamiltonian_matrix(potential_matrix):

    output_shape = 3**4

    num_points = potential_matrix.shape[0]

    factor = output_shape/num_points + 1

    print(potential_matrix)

    potential_matrix = ndimage.zoom(potential_matrix, factor, order=0, mode='constant', cval=0)

    #potential_matrix = resize(potential_matrix, output_shape, anti_aliasing=True)

    print(potential_matrix)

    #Dirichlet
    potential_matrix = np.pad(potential_matrix, pad_width=1, constant_values=10**8)


    # Create a heatmap
    plt.imshow(potential_matrix, cmap='binary')

    # Show grid lines
    plt.grid(True, which='both', color='black', linewidth=1.5, linestyle='-', alpha=0.5)

    # Show the plot
    plt.show()

    num_points = potential_matrix.shape[0]

    dx = 1.0 / (num_points - 1)

    hamiltonian_matrix = np.zeros((num_points**2, num_points**2), dtype=complex)

    for i in range(num_points):
        for j in range(num_points):

            #if potential_matrix[i, j] == 1:
             #   potential_matrix[i, j] = np.inf

            index = i * num_points + j

            # Diagonal elements
            hamiltonian_matrix[index, index] = -0.5 / (dx**2) * 2 + potential_matrix[i, j]

            # Off-diagonal elements (finite difference approximation of Laplacian)
            if i > 0:
                hamiltonian_matrix[index, index - num_points] = 0.5 / (dx**2) 
            if i < num_points - 1:
                hamiltonian_matrix[index, index + num_points] = 0.5 / (dx**2)
            if j > 0:
                hamiltonian_matrix[index, index - 1] = 0.5 / (dx**2) 
            if j < num_points - 1:
                hamiltonian_matrix[index, index + 1] = 0.5 / (dx**2) 

    return hamiltonian_matrix

# Parameters
file_path = 'Sierpinski.dat' 

# Read potential matrix from file
potential_matrix = read_potential_matrix(file_path)
print(potential_matrix.astype(int))

# Generate Hamiltonian matrix
hamiltonian_matrix = generate_hamiltonian_matrix(potential_matrix)

# Diagonalize the matrix
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)

# Print eigenvalues
print("Eigenvalues:")
print(eigenvalues[0])

# Plot the first few eigenvectors
num_eigenvectors_to_plot = 5
for i in range(num_eigenvectors_to_plot):
    eigenvector = eigenvectors[:, i].reshape((int(np.sqrt(len(eigenvectors[:, i]))), int(np.sqrt(len(eigenvectors[:, i])))))
    plt.imshow(np.abs(eigenvector)**2, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    plt.title(f"Eigenvector {i+1}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
