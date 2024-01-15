import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator_potential(x, y, m, omega):
    return 0.5 * m * omega**2 * (x**2 + y**2)

def generate_hamiltonian_matrix(x_range, y_range, num_points, m, omega):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]

    hamiltonian_matrix = np.zeros((num_points**2, num_points**2), dtype=complex)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            index = i * num_points + j

            # Diagonal elements
            hamiltonian_matrix[index, index] = 0.5 / (dx**2) * 4 - 0.5 / (dy**2) + harmonic_oscillator_potential(x, y, m, omega)

            # Off-diagonal elements (finite difference approximation of Laplacian)
            if i > 0:
                hamiltonian_matrix[index, index - num_points] = -0.5 / (dx**2)
            if i < num_points - 1:
                hamiltonian_matrix[index, index + num_points] = -0.5 / (dx**2)
            if j > 0:
                hamiltonian_matrix[index, index - 1] = -0.5 / (dy**2)
            if j < num_points - 1:
                hamiltonian_matrix[index, index + 1] = -0.5 / (dy**2)

    return hamiltonian_matrix

# Parameters
x_range = (0, 1)
y_range = (0, 1)
num_points = 20
m = 1.0
omega = 1.0

# Generate Hamiltonian matrix
hamiltonian_matrix = generate_hamiltonian_matrix(x_range, y_range, num_points, m, omega)

print(hamiltonian_matrix)

# Diagonalize the matrix
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)

# Print eigenvalues
print("Eigenvalues:")
print(eigenvalues)


print("Eigenvectors:")
print(eigenvectors)

# Plot the first few eigenvectors
num_eigenvectors_to_plot = 5
for i in range(num_eigenvectors_to_plot):
    eigenvector = eigenvectors[:, i].reshape((num_points, num_points))
    x_values_plot = np.linspace(x_range[0], x_range[1], num_points)
    y_values_plot = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x_values_plot, y_values_plot)
    
    plt.contourf(X, Y, np.abs(eigenvector)**2, levels=20, cmap='viridis')
    plt.title(f"Eigenvector {i+1}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.imsave("EigenvectorHar_" + str(i+1) + ".eps", np.abs(eigenvector)**2, cmap='GnBu')  # Provide a valid file path
