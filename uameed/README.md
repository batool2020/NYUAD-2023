# NYUAD 2023 - Team 14 Wameed aka Flash 

## Motivation 
The purpose of this project is to provide a solution to the growing climate change problem by reducing the amount of emissions produced by transportation and delivery vehicles. In order to account for the difficulty presented by the size and denseness of cities, we propose to model this problem for determining the optimal path by utilizing the Traveling Salesman Problem (TSP) and quantum computing.
We're particular proud for using quantum computers in two sections of our solution including the Xanadu device as well as the IBM Vqe Sampler on a qasm_simulator. Although our work will not be able to be run on a NISQ device for the time being, we're excited to dream big on how in the future quantum computers may be used.


## Demo
Please see the ![demo](https://github.com/TheGupta2012/NYUAD-2203/tree/main/demos) for an example of the classical implementation and quanutm implementation.

## Our solution 
- We try to solve resource allocation problems which can be mapped to a graph. This induced graph has nodes which map to entities and the edges which map to the cost in going from one node to another
- Leverage **quantum advantage** to break down impractical graph based logistics problems into realistic sizes
- Our approach involves the following steps - 
  1. We reduce a graph containing clusters into different dense sub-graphs via [Gaussian Boson Sampling](https://www.nature.com/articles/s41586-022-04725-x)
  2. Post this, we solve Travelling Salesman Problem via VQE on the smaller dense subgraphs and find a route which connects all the nodes in a cycle
  3. These cycles are computed for each smaller subgraph and then for the city wide network 
  4. Finally, we use the Dikstra's algorithm to find the shortest routes for moving between intercity edges 
  5. This global cycle is the route we propose as the solution
  
  
## Installation 
Please see the requirements.txt

## SDG


## Our Stack
1. Python
2. qBraid
3. Xanadu Gaussian Boson Sampler
4. IBM Backend




