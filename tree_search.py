# tree_search.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# 仅用于类型提示：避免循环依赖时可用 Any
# from graph_utils import Graph, Goal
# from agent import Agent

INF = float("inf")
def compute_energy_from_model(
    *,
    dist: float,
    node_tag: str,
    node_score: Optional[float] = None, # not being used
    edge_type: str,
    energy_model: Any,
    cost_cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute edge traversal energy using a linear model.

    Linear model:
        energy = a * dist * region_factor * edge_type_factor

    Notes:
      - Only 'linear' is supported in this demo.
      - No modality switching cost here.
      - score is ignored (per demo assumption).
    """
    if dist <= 0.0:
        return 0.0

    model_tag = energy_model.energy_model_tag
    if model_tag != "linear":
        raise NotImplementedError(
            f"Energy model '{model_tag}' is not supported in this demo."
        )

    # ---- base linear coefficient ----
    a = float(energy_model.energy_factors.get("a", 1.0))

    # ---- region multiplier ----
    region_factor = float(
        energy_model.region_factors.get(node_tag, 1.0)
    )

    # ---- edge-type multiplier (optional) ----
    edge_factor = float(
        energy_model.edge_type_factors.get(edge_type, 1.0)
    )

    # ---- final energy ----
    energy = a * dist * region_factor * edge_factor
    return float(energy)



@dataclass
class TreeResult:
    # identity
    root_goal: Any                 # Goal
    root_node_id: int
    agent_name: str

    # budgets
    budget_energy: float

    # node-level best energy (root -> node)
    energy_cost: Dict[int, float] = field(default_factory=dict)

    # shortest-path tree pointers
    parent_node: Dict[int, int] = field(default_factory=dict)         # v -> u
    parent_edge_id: Dict[int, int] = field(default_factory=dict)      # v -> eid
    chosen_model: Dict[int, str] = field(default_factory=dict)        # v -> model used for (parent[v] -> v)
    step_energy: Dict[int, float] = field(default_factory=dict)       # v -> energy of (parent[v] -> v)

    # settled nodes (finalized by Dijkstra)
    settled_nodes: Set[int] = field(default_factory=set)

    # optional: edges for visualization
    tree_edge_ids: Set[int] = field(default_factory=set)              # parent edges
    reachable_edge_ids: Optional[Set[int]] = None                     # induced subgraph edges (optional)

    # debug/stats
    n_pq_pop: int = 0
    n_relax: int = 0
    n_relax_success: int = 0

        
def edge_energy_min_model(
    g: Any,
    *,
    u: int,
    v: int,
    w: float,
    eid: int,
    agent: Any,
    rules: Dict[str, List[str]],
    cost_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Optional[str]]:
    """
    Compute incremental energy for traversing edge u->v, choosing the best feasible model.

    Must return:
      (edge_energy, chosen_model)

    If infeasible:
      (INF, None)

    Notes:
      - DO NOT update any global state here.
      - Keep this function pure and easy to extend.
    """
    # --- environment features ---
    nv = g.nodes[v]
    node_tag = nv.tag
    node_score = getattr(nv, "score", 1.0)

    e = g.edges[eid]
    edge_type = getattr(e, "etype", "intra")

    # --- hard block (agent-level) ---
    if node_tag in agent.no_traversable_tags:
        return INF, None

    # --- region/model rule ---
    allowed = rules.get(node_tag, [])
    feasible = [m for m in allowed if m in agent.models]
    if not feasible:
        return INF, None

    # --- choose best model by energy ---
    # NOTE: energy formula registry should live in tree_search.py (as you要求)
    best_cost = INF
    best_model = None
    for m in feasible:
        em = agent.get_energy_model(m)  # EnergyModel is params-only
        cost = compute_energy_from_model(
            dist=w,
            node_tag=node_tag,
            node_score=node_score,
            edge_type=edge_type,
            energy_model=em,
            cost_cfg=cost_cfg,
        )
        if cost < best_cost:
            best_cost = cost
            best_model = m

    return best_cost, best_model


import heapq




def dijkstra_budget_expand(
    g: Any,
    *,
    root_goal: Any,
    agent: Any,
    rules: Dict[str, List[str]],
    budget_energy: Optional[float] = None,
    cost_cfg: Optional[Dict[str, Any]] = None,
) -> TreeResult:
    """
    Budget-limited Dijkstra (Uniform Cost Search) on node-level states.

    - Settled nodes = nodes popped from PQ with minimal finalized energy.
    - Stops when next PQ energy exceeds budget.

    Returns TreeResult with:
      - energy_cost[v] = min energy to reach v
      - parent_node/parent_edge_id/chosen_model/step_energy for path recovery
      - settled_nodes: finalized reachable set under budget
    """
    if budget_energy is None:
        budget_energy = float(agent.energy_budget)

    root = int(root_goal.node_id)

    res = TreeResult(
        root_goal=root_goal,
        root_node_id=root,
        agent_name=str(agent.name),
        budget_energy=float(budget_energy),
    )

    # init
    res.energy_cost[root] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, root)]  # (energy_cost, node)

    while pq:
        cur_cost, u = heapq.heappop(pq)
        res.n_pq_pop += 1

        # outdated entry?
        if cur_cost != res.energy_cost.get(u, INF):
            continue

        # budget cutoff: since pq is nondecreasing, we can stop
        if cur_cost > res.budget_energy:
            break

        # settled check
        if u in res.settled_nodes:
            continue
        res.settled_nodes.add(u)

        # expand neighbors
        for (v, w, eid) in g.adj[u]:
            res.n_relax += 1
            # if v already settled, we can skip relax (optional optimization)
            # Dijkstra correctness allows skipping settled target
            if v in res.settled_nodes:
                continue

            edge_e, best_model = edge_energy_min_model(
                g,
                u=u, v=v, w=w, eid=eid,
                agent=agent,
                rules=rules,
                cost_cfg=cost_cfg,
            )
            if edge_e == INF or best_model is None:
                continue

            new_cost = cur_cost + float(edge_e)

            if new_cost > res.budget_energy:
                continue

            old = res.energy_cost.get(v, INF)
            if new_cost < old:
                res.n_relax_success += 1
                res.energy_cost[v] = new_cost

                res.parent_node[v] = u
                res.parent_edge_id[v] = int(eid)
                res.chosen_model[v] = str(best_model)
                res.step_energy[v] = float(edge_e)

                heapq.heappush(pq, (new_cost, v))

    # tree edges are parent edges (shortest-path tree)
    res.tree_edge_ids = set(res.parent_edge_id.values())

    # reachable_edge_ids (optional) - can be computed lazily later
    # If you want eagerly:
    # res.reachable_edge_ids = {
    #     e.id for e in g.edges
    #     if e.u in res.settled_nodes and e.v in res.settled_nodes
    # }

    return res



@dataclass
class PathResult:
    agent_name: str
    root_node_id: int
    target_node_id: int

    nodes: List[int] = field(default_factory=list)              # node ids from root -> target
    edge_ids: List[int] = field(default_factory=list)           # edge ids aligned with nodes[i]->nodes[i+1]
    models: List[str] = field(default_factory=list)             # chosen model per edge (same length as edge_ids)
    step_energy: List[float] = field(default_factory=list)      # per-edge energy (same length as edge_ids)

    total_energy: float = 0.0
    coords: Optional[List[Tuple[float, float]]] = None           # optional (filled if graph provided)

        
        
INF = float("inf")

def recover_path(
    g: Any,
    tree: Any,              # TreeResult
    target_node_id: int,
) -> PathResult:
    """
    Recover root->target path from TreeResult (shortest-path tree).
    Assumes target is reachable (in tree.energy_cost & has parent unless it's root).
    """
    root = int(tree.root_node_id)
    t = int(target_node_id)

    if t not in tree.energy_cost or tree.energy_cost[t] == INF:
        raise ValueError(f"Target node {t} is not reachable in this tree.")

    if t != root and t not in tree.parent_node:
        raise ValueError(f"Target node {t} has no parent in tree (maybe not settled?).")

    # reverse build
    nodes_rev = [t]
    edge_ids_rev: List[int] = []
    models_rev: List[str] = []
    step_energy_rev: List[float] = []

    cur = t
    while cur != root:
        if cur not in tree.parent_node:
            raise ValueError(
                f"Broken parent chain at node {cur}. Cannot recover full path to root {root}."
            )
        u = int(tree.parent_node[cur])
        eid = int(tree.parent_edge_id[cur])
        m = str(tree.chosen_model[cur])
        se = float(tree.step_energy.get(cur, 0.0))

        edge_ids_rev.append(eid)
        models_rev.append(m)
        step_energy_rev.append(se)

        cur = u
        nodes_rev.append(cur)

    # reverse to root->target
    nodes = list(reversed(nodes_rev))
    edge_ids = list(reversed(edge_ids_rev))
    models = list(reversed(models_rev))
    step_energy = list(reversed(step_energy_rev))

    path = PathResult(
        agent_name=str(tree.agent_name),
        root_node_id=root,
        target_node_id=t,
        nodes=nodes,
        edge_ids=edge_ids,
        models=models,
        step_energy=step_energy,
        total_energy=float(tree.energy_cost[t]),
    )

    # optional coords
    coords = []
    for nid in nodes:
        n = g.nodes[nid]
        coords.append((float(n.x), float(n.y)))
    path.coords = coords

    return path
