# Sample run of TSP

from tsp import run_tsp_on_simulator, sample_graph_with_weights


graph = sample_graph_with_weights()
optimizer = run_tsp_on_simulator(graph)
