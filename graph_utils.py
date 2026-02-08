#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# graph_utils.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Any
import numpy as np

from typing import List, Tuple, Optional, Set, Dict, Iterable, Union

                

# ----------------------------
# Constants / Tags / Edge Types
# ----------------------------

TAG_DOOR = "door"  # 建议 door 独立成一个 domain/tag
EDGE_INTRA = "intra"    # 同域边：indoor-indoor / outdoor-outdoor / door-door
EDGE_PORTAL = "portal"  # door 跨域边：door-indoor / door-outdoor


# ----------------------------
# Node / Edge dataclasses
# ----------------------------

@dataclass(frozen=True)
class Node:
    """Topology node in continuous 2D space."""
    id: int
    x: float
    y: float

    tag: str                 # "indoor" / "outdoor" / "door" / "no-traversable" / ...
    score: float             # traversability score at this point (or default)

    region_id: Optional[str] = None  # Optional: Which specific Region instance
    source: str = "random"           # "random" / "forced_door" / ...


@dataclass(frozen=True)
class Edge:
    """Graph edge with semantic type."""
    id: int
    u: int
    v: int
    w: float                 # edge weight (e.g., Euclidean distance)

    etype: str = EDGE_INTRA  # "intra" or "portal"
    reason: Optional[str] = None     # Debug: Why does it exist / or why was it rejected
    meta: Dict[str, Any] = field(default_factory=dict)  # extension：length、cost components


# ----------------------------
# Graph container
# ----------------------------

class Graph:
    """
    Graph stores:
      - nodes: list[Node] indexed by node.id
      - edges: list[Edge] indexed by edge.id
      - adj: adjacency list where adj[u] = list[(v, w, edge_id)]
      - indices: node_ids_by_tag, edge_ids_by_type
    """

    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.adj: List[List[Tuple[int, float, int]]] = []

        # indices/caches
        self.node_ids_by_tag: Dict[str, List[int]] = {}
        self.edge_ids_by_type: Dict[str, List[int]] = {}

    # ---- node ops ----
    def add_node(
        self,
        x: float,
        y: float,
        tag: str,
        score: float,
        region_id: Optional[str] = None,
        source: str = "random",
    ) -> int:
        node_id = len(self.nodes)
        n = Node(id=node_id, x=float(x), y=float(y), tag=str(tag), score=float(score),
                 region_id=region_id, source=str(source))
        self.nodes.append(n)
        self.adj.append([])

        self.node_ids_by_tag.setdefault(n.tag, []).append(node_id)
        return node_id

    # ---- edge ops ----
    def add_edge(
        self,
        u: int,
        v: int,
        w: float,
        etype: str = EDGE_INTRA,
        reason: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        undirected: bool = True,
    ) -> int:
        """
        Add an edge. By default graph is undirected: adds both (u->v) and (v->u) in adj list,
        but only one Edge object is stored.
        """
        if u == v:
            raise ValueError("Self-edge is not allowed.")
        if not (0 <= u < len(self.nodes)) or not (0 <= v < len(self.nodes)):
            raise IndexError("Node index out of range.")

        edge_id = len(self.edges)
        e = Edge(
            id=edge_id,
            u=int(u),
            v=int(v),
            w=float(w),
            etype=str(etype),
            reason=reason,
            meta={} if meta is None else dict(meta),
        )
        self.edges.append(e)

        self.adj[u].append((v, float(w), edge_id))
        if undirected:
            self.adj[v].append((u, float(w), edge_id))

        self.edge_ids_by_type.setdefault(e.etype, []).append(edge_id)
        return edge_id

    # ---- query helpers ----
    def neighbors(self, u: int) -> List[Tuple[int, float, int]]:
        return self.adj[u]

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def get_edge(self, edge_id: int) -> Edge:
        return self.edges[edge_id]

    def node_ids(self, tag: Optional[str] = None) -> List[int]:
        if tag is None:
            return list(range(len(self.nodes)))
        return self.node_ids_by_tag.get(tag, [])

    def edge_ids(self, etype: Optional[str] = None) -> List[int]:
        if etype is None:
            return list(range(len(self.edges)))
        return self.edge_ids_by_type.get(etype, [])

    def __len__(self) -> int:
        return len(self.nodes)
    
    
    def add_anchor_node(self, *, x: float, y: float, tag: str, score: float, source: str) -> int:
        # 本质就是 add_node 的一个语义封装
        return self.add_node(x=x, y=y, tag=tag, score=score, source=source)

# ----------------------------
# NodeSampler
# ----------------------------

class NodeSampler:
    """
    Sample nodes in continuous 2D space under environment constraints.

    Required interface from RegionManager (from map_utils.py):
      - query_tag(x, y, default="unknown") -> str
      - query_score(x, y, default=1.0) -> float

    You can force sample door nodes from a separate door RegionManager (dm),
    and label them with tag 'door' (or whatever dm returns).
    """

    def __init__(
        self,
        width: float,
        height: float,
        region_manager,             # main RegionManager (indoor/outdoor/obstacles)
        door_manager=None,          # optional: door RegionManager
        *,
        default_score: float = 1.0,
        default_tag: str = "unknown",
        no_traversable_tag: str = "no-traversable",
        rng_seed: int = 0,
    ) -> None:
        self.W = float(width)
        self.H = float(height)

        self.rm = region_manager
        self.dm = door_manager

        self.default_score = float(default_score)
        self.default_tag = str(default_tag)
        self.no_trav_tag = str(no_traversable_tag)

        self.rng = np.random.default_rng(rng_seed)

    # ---- basic environment queries ----
    def _in_bounds(self, x: float, y: float) -> bool:
        return (0.0 <= x < self.W) and (0.0 <= y < self.H)

    def _query_tag_score(self, x: float, y: float) -> Tuple[str, float]:
        tag = self.rm.query_tag(x, y, default=self.default_tag)
        score = self.rm.query_score(x, y, default=self.default_score)
        return tag, score

    def _is_free(self, tag: str, score: float) -> bool:
        # 你 map_utils 里 obstacle 可能用 tag 或 score=inf 表达，两者都兜住
        if tag == self.no_trav_tag:
            return False
        if not np.isfinite(score):
            return False
        return True
    
    def _is_door(self, x, y, door_tag) -> bool:
        dtag = self.dm.query_tag(x, y, default="unknown")
        if dtag == door_tag:
            return True
        return False

    
    def _is_far_enough(self, x, y, existing_xy, min_dist):
        if min_dist is None or min_dist <= 0:
            return True
        md2 = min_dist * min_dist
        for (xx, yy) in existing_xy:
            dx = x - xx
            dy = y - yy
            if dx*dx + dy*dy < md2:
                return False
        return True


    # ---- sampling primitives ----
    def sample_random_nodes(
        self,
        n,
        *,
        allowed_tags=None,
        min_dist=None,
        max_tries=100000,
        source="random",
        door_tag="door",
    ):
        allowed_set = set(allowed_tags) if allowed_tags is not None else None

        out = []
        existing_xy = []
        tries = 0

        while len(out) < n and tries < max_tries:
            tries += 1
            x = float(self.rng.uniform(0.0, self.W))
            y = float(self.rng.uniform(0.0, self.H))

            if not self._in_bounds(x, y):
                continue

            tag, score = self._query_tag_score(x, y)
#             if not self._is_free(tag, score):
#                 continue
            if allowed_set is not None and tag not in allowed_set:
                continue
            if self._is_door(x, y, door_tag):
                continue

            # NEW: distance constraint
            if not self._is_far_enough(x, y, existing_xy, min_dist):
                continue

            out.append((x, y, tag, score))
            existing_xy.append((x, y))

        if len(out) < n:
            raise RuntimeError(f"Failed to sample {n} nodes. Got {len(out)}.")

        return out

    def sample_door_nodes(self,
                          n, *,
                          min_dist=None,
                          max_tries=200000,
                          door_tag="door",
                          source: str = "forced_door",
        ):
        if self.dm is None:
            raise ValueError("door_manager is None")

        out = []
        existing_xy = []
        tries = 0

        while len(out) < n and tries < max_tries:
            tries += 1
            x = float(self.rng.uniform(0.0, self.W))
            y = float(self.rng.uniform(0.0, self.H))

            if not self._in_bounds(x, y):
                continue

            dtag = self.dm.query_tag(x, y, default="unknown")
            if dtag != door_tag:
                continue

            tag_main, score_main = self._query_tag_score(x, y)
            if not self._is_free(tag_main, score_main):
                continue

            if not self._is_far_enough(x, y, existing_xy, min_dist):
                continue

            out.append((x, y, door_tag, score_main))
            existing_xy.append((x, y))

        if len(out) < n:
            raise RuntimeError("Failed to sample door nodes.")

        return out

    # ----------------------- boundary -----------------------
    def add_portal_nodes_from_region_boundary(
        self,
        g,
        region_name: str,
        *,
        n: int,
        res: float,
        min_dist: Optional[float] = None,
        stride: int = 1,
        door_tag: str = "door",
        source: Optional[str] = None,
        jitter: float = 0.0,
        max_tries: int = 1_000_000,
        require_inside_region: bool = False,
        require_boundary_mask: bool = True) -> List[int]:
        """
        从 RegionManager 预计算的 boundary mask 上采样 portal nodes，并直接加入 Graph。

        参数
        ----
        g : Graph
            目标图（已包含 random nodes / 其他 portal nodes 等）
        region_name : str
            对应某个 Region 的 name（rm.boundary_masks[region_name] 必须存在）
        n : int
            希望新增的 portal node 数量
        res : float
            栅格分辨率（用于把 mask cell index 转换到世界坐标）
        min_dist : float | None
            与图中已有节点的最小距离约束（包括之前加的 portal nodes）
        stride : int
            对 boundary mask 的下采样步长（cell 单位）；越大越稀疏
        door_tag : str
            portal node 的 tag（你希望与 door 相同）
        source : str | None
            node.source 字段（用于 debug）；默认 "boundary:<region_name>"
        jitter : float
            在 cell center 周围加入随机抖动（单位：世界坐标），范围约为 [-jitter, +jitter]
            demo 可以先设 0.0
        max_tries : int
            防止死循环的上限（候选点太少 / min_dist 太大时）
        require_inside_region : bool
            True 时：采样点必须满足 region.contains(x,y)（更严格）
            demo 可以 False
        require_boundary_mask : bool
            True 时：如果没找到 boundary mask 则报错；否则返回空列表

        返回
        ----
        added_node_ids : List[int]
            新增节点的 node_id 列表
        """
        if n <= 0:
            return []

        if stride <= 0:
            raise ValueError("stride must be >= 1")

        rm = self.rm
        if not hasattr(rm, "boundary_masks") or region_name not in rm.boundary_masks:
            if require_boundary_mask:
                raise ValueError(f"boundary mask not found for region '{region_name}'. "
                                 f"Call rm.compute_boundary_masks(...) first.")
            return []

        region = rm.get_region_by_name(region_name)
        if region is None:
            raise ValueError(f"region '{region_name}' not found in RegionManager.")

        mask = rm.boundary_masks[region_name]
        H, W = mask.shape

        # --- build candidate cell list (boundary==True), with stride downsampling ---
        # 方式：先取 True 的索引，再用 stride 筛一遍（比整图遍历更快）
        ys, xs = np.where(mask)
        if ys.size == 0:
            raise RuntimeError(f"Region '{region_name}' has empty boundary mask.")

        if stride > 1:
            keep = (xs % stride == 0) & (ys % stride == 0)
            xs = xs[keep]
            ys = ys[keep]

        if xs.size == 0:
            raise RuntimeError(f"After applying stride={stride}, no boundary cells remain "
                               f"for region '{region_name}'.")

        # shuffle candidates for random sampling
        idx = np.arange(xs.size)
        self.rng.shuffle(idx)
        xs = xs[idx]
        ys = ys[idx]

        # --- prepare existing nodes for min_dist check ---
        existing_xy: List[Tuple[float, float]] = [
            (nd.x, nd.y) for nd in g.nodes
            if getattr(nd, "tag", None) == door_tag
        ]

        md2 = None if (min_dist is None or min_dist <= 0) else float(min_dist) * float(min_dist)

        def far_enough(x: float, y: float) -> bool:
            if md2 is None:
                return True
            for (xx, yy) in existing_xy:
                dx = x - xx
                dy = y - yy
                if dx * dx + dy * dy < md2:
                    return False
            return True

        H, W = mask.shape
        world_w = W * res
        world_h = H * res
        def in_bounds(x: float, y: float) -> bool:
            return (0.0 <= x <= float(self.W)) and (0.0 <= y <= float(self.H))

        # --- sample ---
        added_node_ids: List[int] = []
        src = source if source is not None else f"boundary:{region_name}"

        tries = 0
        # 一次扫描候选列表；若不够，允许重新洗牌再扫（由 max_tries 控制）
        cursor = 0
        while len(added_node_ids) < n and tries < max_tries:
            tries += 1
            if cursor >= xs.size:
                # 没采够但候选用完了：重新打乱再试（通常是 min_dist 太大）
                cursor = 0
                idx = np.arange(xs.size)
                self.rng.shuffle(idx)
                xs = xs[idx]
                ys = ys[idx]

            ix = int(xs[cursor])
            iy = int(ys[cursor])
            cursor += 1

            # cell center in world
            x = (ix + 0.5) * res
            y = (iy + 0.5) * res

            # optional jitter
            if jitter > 0.0:
                x = float(x + self.rng.uniform(-jitter, jitter))
                y = float(y + self.rng.uniform(-jitter, jitter))

            # bounds check (against sampler world)
            if not in_bounds(x, y):
                continue

            if require_inside_region and (not region.contains(x, y)):
                continue

            # min_dist check against ALL existing nodes
            if not far_enough(x, y):
                continue

            # score/tag from region manager (or you can force tag to door_tag)
            score = rm.query_score(x, y, default=1.0)

            # add node
            nid = g.add_node(x=x, y=y, tag=door_tag, score=score, source=src)

            added_node_ids.append(nid)
            existing_xy.append((x, y))  # update for subsequent spacing checks

        if len(added_node_ids) < n:
            raise RuntimeError(
                f"Failed to sample enough portal nodes on boundary of '{region_name}'. "
                f"Requested n={n}, got {len(added_node_ids)}. "
                f"Try decreasing min_dist / stride, increasing boundary density, or reducing n."
            )

        return added_node_ids
    
    

    # ---- convenience: create a Graph with sampled nodes ----
    def make_graph_with_nodes(
        self,
        n_random: int,
        n_door: int = 0,
        min_dist=None,
        *,
        random_allowed_tags: Optional[Iterable[str]] = None,
    ) -> Graph:

        g = Graph()

        # random nodes
        random_samples = self.sample_random_nodes(
            n_random,
            min_dist=min_dist,
            allowed_tags=random_allowed_tags,
            source="random",
        )
        for x, y, tag, score in random_samples:
            g.add_node(x=x, y=y, tag=tag, score=score, source="random")

        # forced door nodes
        if n_door > 0:
            door_samples = self.sample_door_nodes(
                n_door,
                min_dist=min_dist,
                source="forced_door",
            )
            for x, y, tag, score in door_samples:
                g.add_node(x=x, y=y, tag=tag, score=score, source="forced_door")

        return g

    
    
    
    
#     ------------------------------------------------------------------------- ###



def _pair_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)

def _coords_of_nodes(g, node_ids: List[int]) -> np.ndarray:
    pts = np.zeros((len(node_ids), 2), dtype=float)
    for i, nid in enumerate(node_ids):
        n = g.nodes[nid]
        pts[i, 0] = n.x
        pts[i, 1] = n.y
    return pts


class GraphBuilderKNN:
    """
    Generic builder for multiple subgraphs defined by tags.

    - Intra edges: for each domain tag in `domain_tags`, build kNN within that tag only
    - Portal edges: for each node with tag `portal_tag`, connect to kNN among ALL non-portal nodes
      (no portal_targets; portal is a generic cross-domain node)
    - Optional r_max to filter long edges
    """

    def __init__(
        self,
        *,
        domain_tags: List[str],
        portal_tag: str = "door",
    ) -> None:
        if not domain_tags:
            raise ValueError("domain_tags must be a non-empty list.")
        self.domain_tags = list(domain_tags)
        self.portal_tag = portal_tag

    def build_edges(
        self,
        g,
        *,
        # intra config (global defaults)
        k_intra: int = 8,
        r_max: Optional[float] = None,

        # per-domain override (optional)
        k_intra_by_tag: Optional[Dict[str, int]] = None,
        r_max_by_tag: Optional[Dict[str, float]] = None,

        # portal config
        k_portal: int = 3,
        r_max_portal: Optional[float] = None,

        undirected: bool = True,
        allow_portal_intra: bool = False,  # allow portal<->portal intra edges
        allow_portal_to_portal: bool = False,  # portal edges connect to other portal nodes?
    ):
        """
        Mutates graph g by adding edges.
        """
        if k_intra <= 0:
            raise ValueError("k_intra must be > 0")
        if k_portal < 0:
            raise ValueError("k_portal must be >= 0")
        if r_max is not None and r_max <= 0:
            raise ValueError("r_max must be > 0 if provided")
        if r_max_portal is not None and r_max_portal <= 0:
            raise ValueError("r_max_portal must be > 0 if provided")

        added: Set[Tuple[int, int]] = set()

        # 1) Intra edges for each domain tag (separate subgraphs)
        for tag in self.domain_tags:
            ids = g.node_ids(tag)
            if len(ids) < 2:
                continue

            kk = k_intra_by_tag.get(tag, k_intra) if k_intra_by_tag else k_intra
            rr = r_max_by_tag.get(tag, r_max) if r_max_by_tag else r_max

            kk = min(int(kk), len(ids) - 1)
            if kk <= 0:
                continue

            self._add_knn_intra_edges(
                g, ids, k=kk, r_max=rr, added=added, undirected=undirected, etype="intra"
            )

        # Optional: portal-tag self intra edges (usually not needed)
        portal_ids = g.node_ids(self.portal_tag)
        if allow_portal_intra and len(portal_ids) >= 2:
            kk = min(k_intra, len(portal_ids) - 1)
            self._add_knn_intra_edges(
                g, portal_ids, k=kk, r_max=r_max, added=added, undirected=undirected, etype="intra"
            )

        # 2) Portal edges: portal -> ALL (no portal_targets)
        if k_portal > 0 and len(portal_ids) > 0:
            self._add_portal_edges_any(
                g,
                portal_ids=portal_ids,
                k_portal=k_portal,
                r_max=r_max_portal if r_max_portal is not None else r_max,
                added=added,
                undirected=undirected,
                allow_portal_to_portal=allow_portal_to_portal,
            )

        return g

    def _add_knn_intra_edges(
        self,
        g,
        node_ids: List[int],
        *,
        k: int,
        r_max: Optional[float],
        added: Set[Tuple[int, int]],
        undirected: bool,
        etype: str = "intra",
    ):
        m = len(node_ids)
        if m < 2:
            return

        pts = _coords_of_nodes(g, node_ids)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)

        r2_max = None if r_max is None else float(r_max) * float(r_max)

        for i in range(m):
            nbr_idx = np.argsort(d2[i])[:k]
            u = node_ids[i]
            for j in nbr_idx:
                if not np.isfinite(d2[i, j]):
                    continue
                if r2_max is not None and d2[i, j] > r2_max:
                    continue

                v = node_ids[j]
                key = _pair_key(u, v)
                if key in added:
                    continue

                w = float(np.sqrt(d2[i, j]))
                g.add_edge(u, v, w, etype=etype, undirected=undirected)
                added.add(key)

    def _add_portal_edges_any(
        self,
        g,
        *,
        portal_ids: List[int],
        k_portal: int,
        r_max: Optional[float],
        added: Set[Tuple[int, int]],
        undirected: bool,
        allow_portal_to_portal: bool = False,
    ):
        """
        For each portal node p:
          connect p to k_portal nearest nodes among ALL nodes
          (excluding itself), optionally excluding other portal nodes.

        This removes portal_targets; portal nodes are generic cross-domain connectors.
        """
        r2_max = None if r_max is None else float(r_max) * float(r_max)

        # neighbor pool
        if allow_portal_to_portal:
            nbr_ids = list(range(len(g.nodes)))
        else:
            nbr_ids = [
                i for i, nd in enumerate(g.nodes)
                if getattr(nd, "tag", None) != self.portal_tag
            ]

        if len(nbr_ids) == 0:
            return

        nbr_pts = _coords_of_nodes(g, nbr_ids)

        for p in portal_ids:
            pn = g.nodes[p]
            ppt = np.array([pn.x, pn.y], dtype=float)

            diff = nbr_pts - ppt[None, :]
            d2 = np.sum(diff * diff, axis=1)

            # exclude self if present in pool
            if allow_portal_to_portal:
                for idx, nid in enumerate(nbr_ids):
                    if nid == p:
                        d2[idx] = np.inf
                        break

            order = np.argsort(d2)[:min(k_portal, len(nbr_ids))]

            for idx in order:
                if not np.isfinite(d2[idx]):
                    continue
                if r2_max is not None and d2[idx] > r2_max:
                    continue

                u, v = p, nbr_ids[int(idx)]
                if u == v:
                    continue

                key = _pair_key(u, v)
                if key in added:
                    continue

                w = float(np.sqrt(d2[idx]))
                g.add_edge(u, v, w, etype="portal", undirected=undirected)
                added.add(key)
                
                
                
                

@dataclass
class Goal:
    type_tag: str            # "start" or "end" (or "goal")
    pose: Tuple[float, float]
    node_id: int             # graph node id for this goal anchor
    name: Optional[str] = None

                
                

def add_goal_anchor(
    g,
    *,
    rm,                       # RegionManager
    goal_type_tag: str,        # "start" / "end"
    pose: Tuple[float, float], # (x,y)
    name: str = "",
    source_prefix: str = "goal",
) -> Goal:
    """
    Create a Goal by FORCE inserting a node at pose.
    Node tag/score are queried from RegionManager based on pose location.
    """
    x, y = float(pose[0]), float(pose[1])

    # 1) query semantic tag/score from region manager
    node_tag = rm.query_tag(x, y, default="unknown")
    node_score = rm.query_score(x, y, default=1.0)

    # 2) force insert node at exact pose
    nid = g.add_node(
        x=x,
        y=y,
        tag=node_tag,      # IMPORTANT: semantic tag
        score=node_score,
        source=f"{source_prefix}:{goal_type_tag}" + (f":{name}" if name else "")
    )

    # 3) return Goal object (holds start/end semantics)
    return Goal(
        type_tag=goal_type_tag,
        pose=(x, y),
        node_id=nid,
        name=name or None
    )

# from typing import List, Optional, Tuple, Set
# import numpy as np


def connect_goals_to_graph(
    g,
    goals: List,
    *,
    r: float,
    k: int,
    undirected: bool = True,
    etype: str = "anchor",
    skip_portal: bool = True,
    portal_tag: str = "door",
) -> None:
    """
    Connect each goal anchor node to nearby nodes in the SAME semantic tag.

    Parameters
    ----------
    g : Graph
    goals : list[Goal]
        Each Goal must have .node_id and the corresponding node exists in g.nodes
    r : float
        Search radius in world units
    k : int
        Max number of edges to add per goal
    undirected : bool
        Whether to add undirected edges (depends on your g.add_edge implementation)
    etype : str
        Edge type label, e.g., "anchor"
    skip_portal : bool
        If True, do not connect anchor->portal nodes (door nodes), even if tag matches.
        Recommended, because door tag usually differs anyway.
    portal_tag : str
        Portal tag name (default "door")
    """
    if r <= 0:
        raise ValueError("r must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")

    r2 = float(r) * float(r)

    # Build (optional) quick lookup for existing edges to avoid duplicates
    added: Set[Tuple[int, int]] = set()
    if hasattr(g, "edges"):
        for e in g.edges:
            u, v = int(e.u), int(e.v)
            added.add((u, v) if u < v else (v, u))

    for goal in goals:
        gid = int(goal.node_id)
        gn = g.nodes[gid]
        goal_tag = getattr(gn, "tag", None)

        if goal_tag is None:
            continue

        # Collect candidate nodes with same tag (exclude self)
        cand_ids = []
        cand_xy = []

        for nid, nd in enumerate(g.nodes):
            if nid == gid:
                continue
            if getattr(nd, "tag", None) != goal_tag:
                continue
            if skip_portal and getattr(nd, "tag", None) == portal_tag:
                continue

            dx = nd.x - gn.x
            dy = nd.y - gn.y
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                cand_ids.append(nid)
                cand_xy.append((d2, nid))

        if not cand_xy:
            # no neighbor in radius -> skip
            continue

        # sort by distance and take top-k
        cand_xy.sort(key=lambda t: t[0])
        chosen = cand_xy[: min(k, len(cand_xy))]

        for d2, nid in chosen:
            u, v = gid, int(nid)
            key = (u, v) if u < v else (v, u)
            if key in added:
                continue

            w = float(np.sqrt(d2))
            g.add_edge(u, v, w, etype=etype, undirected=undirected)
            added.add(key)

                
            