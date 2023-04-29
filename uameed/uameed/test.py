from qiskit_optimization.applications import Tsp


# Generating a graph of 3 nodes
n = 3
num_qubits = n**2
tsp = Tsp()

print(tsp, type(tsp.graph))
