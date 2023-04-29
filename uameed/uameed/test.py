# Sample run of TSP

from tsp import (
    run_tsp_on_simulator,
    sample_graph_with_weights,
    run_tsp_on_hardware,
    add_missing_edges,
)


graph = sample_graph_with_weights()

graph = add_missing_edges(graph)
# z, result = run_tsp_on_simulator(graph)

# Sample run of TSP on hardware
# z, result = run_tsp_on_hardware(graph)


from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler
from qiskit_ibm_provider import IBMProvider

service = QiskitRuntimeService(channel="ibm_quantum")


options = {
    "max_execution_time": 300,
    "instance": "ibm-q/open/main",
}


with Session(service, backend="ibmq_qasm_simulator") as session:
    sampler = Sampler()

    # Sample run of TSP on hardware
    z, result = run_tsp_on_hardware(graph, sampler)
    print(result)
    print(z, "result")
    print("done")
