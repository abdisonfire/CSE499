import lib
import matplotlib.pyplot as plt
import math
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
import numpy as np

def fn(x):
    return (1/(2*math.pi))*math.exp(-((x)**2) / 2)


def supremum_infimum(func, interval):
    a, b = interval  # Unpack start and end points

    # Sample points within the interval (adjust num_points as needed)
    num_points = 2**15
    samples = np.linspace(a, b, num_points)

    # Evaluate function at sample points
    function_values = [func(x) for x in samples]

    # Estimate supremum and infimum using NumPy
    sup = np.max(function_values)
    inf = np.min(function_values)
    # print(np.sum(function_values))
    return sup, inf

# Example usage (replace 'your_function' with your actual function)
def your_function(x):
    return x**2

def get_k0(eps, fn, interval):
    sup, inf = supremum_infimum(fn, interval)
    return max(math.ceil(math.log2(max(abs(sup),abs(inf))/eps)), 2)

def fidelity(psi1, psi2):
    overlap = np.abs(np.dot((np.array(psi1).conj()), np.array(psi2)))**2
    return overlap

def test_fn(x):
    return math.exp(math.sin(x))

def double_derivative_test_fn(x):
    return -test_fn(x)*(math.sin(x)-math.cos(x)**2)

# print(get_k0(0.01, double_derivative_test_fn, (-1000,1000)))

# qc1 = lib.get_naive_quantum_circuit(10, test_fn, -6, 6)
# qc2 = lib.get_approximate_quantum_circuit(10, test_fn, -6, 6, 6)

# print(1-fidelity(Statevector(qc1), Statevector(qc2)))

# qc1 = lib.get_naive_quantum_circuit(7, test_fn, -6, 6)
# qc2 = lib.get_approximate_quantum_circuit(18, test_fn, -25, 25, 2)

fidelities1 = []
fidelities2 = []
max_qubits = 14
k_0 = 9
for num_qubits in range(2, max_qubits):
    qc1 = lib.get_naive_quantum_circuit(num_qubits, fn, -3.8, 3.8)
    qc2 = lib.get_approximate_quantum_circuit(num_qubits, fn, -3.8, 3.8, 5)
    fidelities1.append(fidelity(Statevector(qc1), Statevector(qc2)))
for num_qubits in range(2, max_qubits):
    qc1 = lib.get_naive_quantum_circuit(num_qubits, fn, -3.8, 3.8)
    qc2 = lib.get_approximate_quantum_circuit(num_qubits, fn, -3.8, 3.8, k_0)
    fidelities2.append(fidelity(Statevector(qc1), Statevector(qc2)))

plt.plot([i for i in range(2, max_qubits)], fidelities1, label = 'Fidelity k_0/2')
plt.plot([i for i in range(2, max_qubits)], fidelities2, label = 'Fidelity k_0')
plt.xlabel('num_qubits')
plt.ylabel('Fidelity')
plt.show()

# print(1-fidelity(Statevector(qc1), Statevector(qc2)))

# statistics = Statevector(qc2).probabilities()
# plt.plot(statistics)
# plt.show()
# statistics = Statevector(qc1).probabilities()
# plt.plot(statistics)
# plt.show()
