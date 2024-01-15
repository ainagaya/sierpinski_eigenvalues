import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Read the data from the file
data = []
with open('eigenvals81.dat', 'r') as file:
    for line in file:
        line = line.strip().split()
        k = float(line[0])
        size = int(line[1])
        i = int(line[2])
        energy = float(line[3])
        data.append((k, size, i, energy))

# Separate the data into separate lists
k_values = [entry[0] for entry in data]
print(k_values)
size_values = [entry[1] for entry in data]
i_values = [entry[2] for entry in data]
print(i_values)
energy_values = [entry[3] for entry in data]


# Plot the data
for i in range(5):
    print(k_values[i::5])
    plt.plot(k_values[i::5], energy_values[i::5], 'x', label='i = ' + str(i),  markersize=15)
    
plt.tick_params(axis='both', which='major', labelsize=24)  # Increase the font size of the axis labels
plt.xlabel('k', fontsize=24)
plt.ylabel('E/(ℏ²/ma²)', fontsize=24)
plt.legend()
plt.title('E vs. k, 5 first eigenvalues', fontsize=24)
plt.yscale('log')
plt.tight_layout()
plt.savefig(f"eiglog.eps")

def exponential_func(x, a, b):
    return np.multiply(a, np.exp(np.multiply(b, x)))


def lineal(x, a, b):
    return a*x + b

# Plot the data

plt.clf()  # Reset the plot

for i in range(5):
    if i == 0:
        popt, pcov = curve_fit(exponential_func, k_values[i::5], energy_values[i::5], p0=(1, 1))
        #popt, pcov = curve_fit(lineal, k_values[i::5], np.log(energy_values[i::5]), p0=(1, 1))
        print(popt)
        # Generate x values
        x_values = np.linspace(min(k_values), max(k_values), 100)
        # Plot the function
        plt.plot(x_values, exponential_func(x_values, *popt), label='Adjust of the ground state', color='black')
        #plt.plot(x_values, lineal(x_values, *popt), label='Adjust of the ground state', color='black')
        plt.plot(k_values[i::5], energy_values[i::5], 'x', label='i = ' + str(i), markersize=15)
    else:
        plt.plot(k_values[i::5], energy_values[i::5], 'x', label='i = ' + str(i), markersize=15)


plt.tick_params(axis='both', which='major', labelsize=24)  # Increase the font size of the axis labels
plt.xlabel('k', fontsize=24)
plt.ylabel('E/(ℏ²/ma²)', fontsize=24)
plt.legend()
plt.title('E vs. k, 5 first eigenvalues', fontsize=24)
plt.tight_layout()
plt.savefig(f"eigadjust.eps")
