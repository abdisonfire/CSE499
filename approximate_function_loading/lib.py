from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RYGate
import math
import numpy as np
from scipy.integrate import quad

def get_integral(fn, x_min, x_max):
    integral, error = quad(fn, x_min, x_max)
    return integral

def theta_lk(fn, x_min, x_max, k, l):
        del_k = (x_max - x_min) / (2 ** (k - 1))
        return 2 * math.acos(
            math.sqrt(
                get_integral(fn, x_min + (l * del_k), x_min + (l + 0.5) * del_k)
                / get_integral(fn, x_min + (l * del_k), x_min + (l + 1) * del_k)
            )
        )
def generate_binary(n):
    res = []
    for i in range(2**n):
        temp = []
        bin = format(i, "0" + str(n) + "b")
        for pos, char in enumerate(bin):
            if char == "0":
                temp.append(pos)
        res.append(temp)
    return res


def get_naive_quantum_circuit(num_qubit, fn, x_min, x_max):
    qc = QuantumCircuit(num_qubit)
    for k in range(num_qubit):
        k += 1
        # if k>2:
        #     qc.x([bit for bit in range(k-2)])
        for i, item in enumerate(generate_binary(k - 1)):
            # print(len(list(item)))
            # print(item)
            cry = RYGate(theta_lk(fn, x_min, x_max, k, i))
            cbit = [bit for bit in range(k - 1)]
            if k > 1:
                cry = cry.control(k - 1)
            if len(item) > 0 and k > 1:
                qc.x(item)
            cbit.append(k - 1)
            # print(cbit)
            qc.append(cry, cbit)
            cbit = [bit for bit in range(k - 1)]
            if len(item) > 0 and (k != num_qubit or i != (2 ** (k - 1) - 1)) and k > 1:
                # print(k, item)
                qc.x(item)
    for i in range(num_qubit // 2):
        qc.swap(i, num_qubit - i - 1)
    return qc


def get_approximate_quantum_circuit(
    num_qubit, fn, x_min, x_max, k0, theta_tilde=math.pi / 2
):

    qc = QuantumCircuit(num_qubit)
    for k in range(min(k0, num_qubit)):
        k += 1
        # if k>2:
        #     qc.x([bit for bit in range(k-2)])
        for i, item in enumerate(generate_binary(k - 1)):
            # print(len(list(item)))
            # print(item)
            cry = RYGate(theta_lk(fn, x_min, x_max, k, i))
            cbit = [bit for bit in range(k - 1)]
            if k > 1:
                cry = cry.control(k - 1)
            if len(item) > 0 and k > 1:
                qc.x(item)
            cbit.append(k - 1)
            # print(cbit)
            qc.append(cry, cbit)
            cbit = [bit for bit in range(k - 1)]
            if len(item) > 0 and (k != num_qubit or i != (2 ** (k - 1) - 1)) and k > 1:
                # print(k, item)
                qc.x(item)
    for k in range(k0 + 1, num_qubit + 1):
        # print('1')
        # thetas = [theta_lk(fn, x_min, x_max, k, l) for l in range(2**(k-1))]
        # print(thetas)
        # print(np.mean(np.array(thetas)))
        ry = RYGate(theta_tilde)
        # print('k=',k, np.pi/2-np.mean(np.array(thetas)))
        qc.append(ry, [k - 1])
        # print((((x_max-x_min)/2**(k-1))/8) * 2, [abs(x - math.pi/2) for x in thetas])
    for i in range(num_qubit // 2):
        qc.swap(i, num_qubit - i - 1)
    return qc


# qc = get_approximate_quantum_circuit_for_f(10, theta_lk_for_test_fn,    0, 100, 8)
