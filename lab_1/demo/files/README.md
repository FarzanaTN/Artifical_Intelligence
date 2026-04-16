# Mirpur Pathfinding — OSMnx + 12 Search Algorithms

## Setup

```bash
pip install osmnx matplotlib networkx geopandas shapely
```

## Run

```bash
python main.py                    # default traveler profile
python main.py --profile female   # safety/lighting/gender weights boosted
python main.py --profile elderly  # age_ease + road_quality boosted
python main.py --profile rush_hour# time + traffic weights boosted
python main.py --profile budget   # cost weight maximized
python main.py --no-plot          # print results only, skip images
```

## What happens on first run
1. Downloads real Mirpur road network from OpenStreetMap via OSMnx
2. Caches to `mirpur_graph.graphml` (subsequent runs load from cache)
3. Finds nearest nodes to Stadium (source) and Pallabi Bus Stand (dest)
4. Runs all 12 algorithms
5. Saves 16 images to `./output/`

## Files

| File | Purpose |
|---|---|
| `graph.py` | OSMnx download, `Node` + `Edge` dataclasses, `CityGraph` |
| `cost_heuristic.py` | `cost(edge, weights)` + `heuristic(graph, n, goal, w)` |
| `algorithms.py` | All 12 algorithms, `SearchResult` dataclass |
| `visualize.py` | Saves all plots using real OSM basemap |
| `main.py` | Runner |

## Algorithms

### Uninformed
| Algo | Strategy | Optimal |
|---|---|---|
| BFS | FIFO, level-by-level | Hop-count |
| DFS | LIFO, deepest first | No |
| DLS | DFS with depth cap | No |
| IDDFS | DLS with increasing limits | Hop-count |
| UCS | Min g(n) priority queue | Yes (cost) |
| BiDS | Two BFS frontiers | Hop-count |

### Informed
| Algo | Strategy | Optimal |
|---|---|---|
| Greedy | Min h(n) only | No |
| A* | Min g(n)+h(n) | Yes (cost) |
| Weighted A* | Min g(n)+ε·h(n) | Within ε× |
| BiA* | Two A* frontiers | Approx. |
| IDA* | DFS with f-threshold | Yes (cost) |
| Beam | BFS, top-k by h(n) | No |

## Cost function

```
cost(edge, weights) =
    base_cost × (w_cost/5)          # travel time
  + base_cost × (w_time/5)          # time proxy
  + traffic   × (w_traffic/10)      # congestion penalty
  + (10-safety)      × (w_safety/10)
  + (10-lighting)    × (w_lighting/10)
  + (10-gender_safe) × (w_gender/10)
  + (10-age_ease)    × (w_age/10)
  + (10-road_quality)× (w_road_quality/10)
```

## Heuristic function

```
h(n) = euclidean_distance(n, goal) × 0.4 × (w_cost+w_time)/10
```
Uses only GPS geometry → admissible → A* remains optimal.
