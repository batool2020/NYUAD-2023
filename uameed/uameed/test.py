# Sample run of TSP

from tsp import run_tsp_on_simulator, sample_graph_with_weights, run_tsp_on_hardware


graph = sample_graph_with_weights()
# optimizer = run_tsp_on_simulator(graph)

# Sample run of TSP on hardware
optimizer = run_tsp_on_hardware(graph)
