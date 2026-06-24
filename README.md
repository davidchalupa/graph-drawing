
# Graph Drawing Project

This repository contains an interactive tool for graph drawing exploration and analysis.
The primary entry point for the project is `graph_drawing_explorer.py`

## Getting Started

- `graph_drawing_explorer.py`: The main entry point for the project, providing an interactive interface for graph
  drawing and analysis.

To install the dependencies, run:

```
pip install -r requirements.txt
```

After that the project can be run as follows:

```
python graph_drawing_explorer.py
```

## Supported Functionalities

- **Finding Clique**: Identify maximum (or a very large) clique within a graph.
- **Finding Dominating Set**: Identify the smallest possible subset of vertices such that every vertex in the graph 
  is either in the subset or adjacent to a vertex in the subset.
- **Finding K-Medoids**: Identify a graph clustering using  set of k representative vertices.
- **Graph Layout Algorithms**: Apply various algorithms to position nodes in a graph for better visualization.

