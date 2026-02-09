#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# viz_utils.py (or put into graph_utils.py)
from __future__ import annotations

from typing import Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Iterable, Tuple, Union

def _collect_node_arrays(g):
    xs = np.array([n.x for n in g.nodes], dtype=float)
    ys = np.array([n.y for n in g.nodes], dtype=float)
    tags = [n.tag for n in g.nodes]
    return xs, ys, tags


def _default_tag_style() -> Dict[str, Dict]:
    """
    你没要求固定配色，我这里给一个最基本的默认样式。
    需要你自己也可以传入 tag_style 覆盖。
    """
    return {
        "indoor": {"marker": "o", "s": 18, "alpha": 0.9},
        "outdoor": {"marker": "o", "s": 18, "alpha": 0.9},
        "door": {"marker": "s", "s": 30, "alpha": 1.0},
        "no-traversable": {"marker": "x", "s": 30, "alpha": 1.0},
        "unknown": {"marker": ".", "s": 14, "alpha": 0.8},
    }


def _plot_nodes(ax, g, tag_style: Optional[Dict[str, Dict]] = None, show_legend: bool = True):
    xs, ys, tags = _collect_node_arrays(g)
    style = _default_tag_style()
    if tag_style is not None:
        # shallow merge
        for k, v in tag_style.items():
            style.setdefault(k, {}).update(v)

    # 按 tag 分组画（便于 legend）
    uniq_tags = sorted(set(tags))
    for t in uniq_tags:
        idx = [i for i, tt in enumerate(tags) if tt == t]
        st = style.get(t, style["unknown"])
        ax.scatter(xs[idx], ys[idx], label=t, **st)

    if show_legend:
        ax.legend(loc="upper right", frameon=True, fontsize=9)


def _random_rgb(rng: np.random.Generator):
    return tuple(rng.random(3).tolist())  # (r,g,b)

def _plot_edges(
    ax,
    g,
    *,
    alpha: float = 0.25,
    linewidth: float = 1.0,
    portal_alpha: float = 0.6,
    portal_linewidth: float = 1.5,
    show_portal: bool = True,

    # NEW: color controls
    domain_colors: Optional[Dict[str, Any]] = None,   # tag -> matplotlib color
    portal_color: Any = "black",
    random_seed: Optional[int] = None,               # for reproducible random colors
):
    """
    画边：
    - intra edge：同一子域(tag)内部的边使用同一颜色（每个tag随机一种，或由domain_colors指定）
    - portal edge：统一使用 portal_color

    注意：要求你的 GraphBuilder 保证 intra edges 不跨域。
    """
    # 1) build/complete domain_colors mapping
    if domain_colors is None:
        domain_colors = {}

    rng = np.random.default_rng(random_seed)

    # 2) draw edges
    for e in g.edges:
        u = g.nodes[e.u]
        v = g.nodes[e.v]

        etype = getattr(e, "etype", None)

        # portal edges
        if show_portal and etype == "portal":
            ax.plot(
                [u.x, v.x], [u.y, v.y],
                color=portal_color,
                alpha=portal_alpha,
                linewidth=portal_linewidth,
                zorder=2,
            )
            continue

        # intra edges: color by domain tag
        # 正常情况下 u.tag == v.tag；如果不等，取一个组合key做兜底
        if u.tag == v.tag:
            key = u.tag
        else:
            key = f"{u.tag}|{v.tag}"  # fallback for unexpected cross-tag intra edges

        if key not in domain_colors:
            domain_colors[key] = _random_rgb(rng)

        ax.plot(
            [u.x, v.x], [u.y, v.y],
            color=domain_colors[key],
            alpha=alpha,
            linewidth=linewidth,
            zorder=1,
        )

    return domain_colors  # 方便你在外面拿到颜色映射做legend/debug



def plot_graph_on_blank(
    g,
    *,
    width: float,
    height: float,
    title: str = "Graph (blank map)",
    figsize: Tuple[float, float] = (6.5, 6.5),
    show_edges: bool = True,
    show_nodes: bool = True,
    show_legend: bool = True,
    tag_style: Optional[Dict[str, Dict]] = None,
    edge_alpha: float = 0.25,
    edge_linewidth: float = 1.0,
    portal_alpha: float = 0.6,
    portal_linewidth: float = 1.5,

    # NEW: edge color options
    domain_colors: Optional[Dict[str, Any]] = None,
    portal_color: Any = "black",
    edge_color_seed: Optional[int] = 0,

    # NEW: goals overlay
    goals: Optional[List[Any]] = None,   # Any to avoid import cycle; ideally List[Goal]
    goal_marker_size: float = 140,
    goal_show_label: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    if show_edges:
        domain_colors = _plot_edges(
            ax, g,
            alpha=edge_alpha, linewidth=edge_linewidth,
            portal_alpha=portal_alpha, portal_linewidth=portal_linewidth,
            domain_colors=domain_colors,
            portal_color=portal_color,
            random_seed=edge_color_seed,
        )

    if show_nodes:
        _plot_nodes(ax, g, tag_style=tag_style, show_legend=show_legend)

    # NEW: plot goals on top
    if goals:
        for goal in goals:
            plot_goal(
                ax,
                goal,
                size=goal_marker_size,
                show_label=goal_show_label,
            )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    return domain_colors  # 可选：返回用于 legend/debug


def plot_graph_on_costmap(
    costmap: np.ndarray,
    g,
    *,
    title: str = "Graph on costmap",
    figsize: Tuple[float, float] = (6.5, 6.5),
    origin: str = "lower",
    show_colorbar: bool = True,
    show_edges: bool = True,
    show_nodes: bool = True,
    show_legend: bool = True,
    tag_style: Optional[Dict[str, Dict]] = None,
    edge_alpha: float = 0.25,
    edge_linewidth: float = 1.0,
    portal_alpha: float = 0.6,
    portal_linewidth: float = 1.5,
    inf_as_max: bool = True,

    # NEW: edge color options
    domain_colors: Optional[Dict[str, Any]] = None,
    portal_color: Any = "black",
    edge_color_seed: Optional[int] = 0,

    # NEW: goals overlay
    goals: Optional[List[Any]] = None,   # Any to avoid import cycle; ideally List[Goal]
    goal_marker_size: float = 140,
    goal_show_label: bool = True,
):
    T = np.array(costmap, dtype=float)

    if inf_as_max:
        finite = T[np.isfinite(T)]
        maxv = finite.max() if finite.size > 0 else 1.0
        T = T.copy()
        T[~np.isfinite(T)] = maxv * 1.2

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(T, origin=origin)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    H, W = T.shape
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")

    if show_edges:
        domain_colors = _plot_edges(
            ax, g,
            alpha=edge_alpha, linewidth=edge_linewidth,
            portal_alpha=portal_alpha, portal_linewidth=portal_linewidth,
            domain_colors=domain_colors,
            portal_color=portal_color,
            random_seed=edge_color_seed,
        )

    if show_nodes:
        _plot_nodes(ax, g, tag_style=tag_style, show_legend=show_legend)

    # NEW: plot goals on top
    if goals:
        for goal in goals:
            plot_goal(
                ax,
                goal,
                size=goal_marker_size,
                show_label=goal_show_label,
            )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    return domain_colors
# def plot_graph_on_blank(
#     g,
#     *,
#     width: float,
#     height: float,
#     title: str = "Graph (blank map)",
#     figsize: Tuple[float, float] = (6.5, 6.5),
#     show_edges: bool = True,
#     show_nodes: bool = True,
#     show_legend: bool = True,
#     tag_style: Optional[Dict[str, Dict]] = None,
#     edge_alpha: float = 0.25,
#     edge_linewidth: float = 1.0,
#     portal_alpha: float = 0.6,
#     portal_linewidth: float = 1.5,

#     # NEW: edge color options
#     domain_colors: Optional[Dict[str, Any]] = None,
#     portal_color: Any = "black",
#     edge_color_seed: Optional[int] = 0,
# ):
#     fig, ax = plt.subplots(figsize=figsize)

#     ax.set_xlim(0, width)
#     ax.set_ylim(0, height)
#     ax.set_aspect("equal", adjustable="box")
#     ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

#     if show_edges:
#         domain_colors = _plot_edges(
#             ax, g,
#             alpha=edge_alpha, linewidth=edge_linewidth,
#             portal_alpha=portal_alpha, portal_linewidth=portal_linewidth,
#             domain_colors=domain_colors,
#             portal_color=portal_color,
#             random_seed=edge_color_seed,
#         )

#     if show_nodes:
#         _plot_nodes(ax, g, tag_style=tag_style, show_legend=show_legend)

#     ax.set_title(title)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     plt.tight_layout()
#     plt.show()

#     return domain_colors  # 可选：返回用于 legend/debug


# def plot_graph_on_costmap(
#     costmap: np.ndarray,
#     g,
#     *,
#     title: str = "Graph on costmap",
#     figsize: Tuple[float, float] = (6.5, 6.5),
#     origin: str = "lower",
#     show_colorbar: bool = True,
#     show_edges: bool = True,
#     show_nodes: bool = True,
#     show_legend: bool = True,
#     tag_style: Optional[Dict[str, Dict]] = None,
#     edge_alpha: float = 0.25,
#     edge_linewidth: float = 1.0,
#     portal_alpha: float = 0.6,
#     portal_linewidth: float = 1.5,
#     inf_as_max: bool = True,

#     # NEW: edge color options
#     domain_colors: Optional[Dict[str, Any]] = None,
#     portal_color: Any = "black",
#     edge_color_seed: Optional[int] = 0,
# ):
#     T = np.array(costmap, dtype=float)

#     if inf_as_max:
#         finite = T[np.isfinite(T)]
#         maxv = finite.max() if finite.size > 0 else 1.0
#         T = T.copy()
#         T[~np.isfinite(T)] = maxv * 1.2

#     fig, ax = plt.subplots(figsize=figsize)

#     im = ax.imshow(T, origin=origin)
#     if show_colorbar:
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     H, W = T.shape
#     ax.set_xlim(0, W)
#     ax.set_ylim(0, H)
#     ax.set_aspect("equal", adjustable="box")

#     if show_edges:
#         domain_colors = _plot_edges(
#             ax, g,
#             alpha=edge_alpha, linewidth=edge_linewidth,
#             portal_alpha=portal_alpha, portal_linewidth=portal_linewidth,
#             domain_colors=domain_colors,
#             portal_color=portal_color,
#             random_seed=edge_color_seed,
#         )

#     if show_nodes:
#         _plot_nodes(ax, g, tag_style=tag_style, show_legend=show_legend)

#     ax.set_title(title)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     plt.tight_layout()
#     plt.show()

#     return domain_colors


    
    
#     --------------------------------------------------------------------------  #

def plot_goal(
    ax,
    goal,
    *,
    size: float = 120,
    edgecolor: str = "k",
    linewidth: float = 1.5,
    zorder: int = 10,
    show_label: bool = True,
    fontsize: Optional[int] = 10
):
    """
    Plot a start/end goal on given axes.

    Parameters
    ----------
    ax : matplotlib Axes
    goal : Goal
        Goal object with fields:
          - goal.type_tag ("start" / "end")
          - goal.pose (x,y)
          - goal.name (optional)
    size : float
        Marker size.
    """

    x, y = goal.pose

    if goal.type_tag == "start":
        color = "red"
        marker = "o"
    elif goal.type_tag in ("end", "goal"):
        color = "blue"
        marker = "o"
    else:
        color = "black"
        marker = "o"

    ax.scatter(
        [x], [y],
        s=size,
        c=color,
        marker=marker,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=zorder,
    )

    if show_label:
        label = goal.name if goal.name else goal.type_tag
        ax.text(
            x, y,
            f" {label}",
            color=color,
            fontsize=fontsize,
            weight="bold",
            zorder=zorder + 1,
        )


#     --------------------------------------------------------------------------  #


# viz_utils.py



# def plot_graph_edges(
#     ax: plt.Axes,
#     g,
#     *,
#     etypes: Optional[Union[str, Iterable[str]]] = None,
#     show_portal_emphasis: bool = True,
#     alpha: float = 0.25,
#     linewidth: float = 1.0,
#     portal_alpha: float = 0.6,
#     portal_linewidth: float = 1.5,
#     zorder: int = 3,
# ):
#     """
#     Draw ONLY edges of graph on the given matplotlib Axes.

#     Parameters
#     ----------
#     ax : matplotlib Axes
#         The axes to draw on (so you can overlay on blank or costmap).
#     g : Graph
#         Your graph object (must have g.edges and g.nodes).
#     etypes : None | str | Iterable[str]
#         If provided, only draw edges whose e.etype is in etypes.
#         Examples:
#           - etypes="portal"
#           - etypes=["intra", "portal"]
#     show_portal_emphasis : bool
#         If True, draw portal edges using (portal_alpha, portal_linewidth).
#     alpha, linewidth :
#         Style for non-portal (or non-emphasized) edges.
#     portal_alpha, portal_linewidth :
#         Style for portal edges if show_portal_emphasis is True.
#     zorder : int
#         Matplotlib z-order for drawing.
#     """
#     if etypes is None:
#         etype_set = None
#     else:
#         if isinstance(etypes, str):
#             etype_set = {etypes}
#         else:
#             etype_set = set(etypes)

#     for e in g.edges:
#         et = getattr(e, "etype", None)
#         if etype_set is not None and et not in etype_set:
#             continue

#         u = g.nodes[e.u]
#         v = g.nodes[e.v]

#         if show_portal_emphasis and et == "portal":
#             ax.plot([u.x, v.x], [u.y, v.y],
#                     alpha=portal_alpha, linewidth=portal_linewidth, zorder=zorder)
#         else:
#             ax.plot([u.x, v.x], [u.y, v.y],
#                     alpha=alpha, linewidth=linewidth, zorder=zorder)


#     --------------------------------------------------------------------------  #

def _plot_edges_by_ids(
    ax,
    g,
    edge_ids,
    *,
    alpha=0.9,
    linewidth=2.0,
    color="black",
    zorder=5,
):
    """Draw edges specified by edge_ids (using Graph.edges[eid])."""
    for eid in edge_ids:
        e = g.edges[int(eid)]
        u = g.nodes[e.u]
        v = g.nodes[e.v]
        ax.plot([u.x, v.x], [u.y, v.y],
                alpha=alpha, linewidth=linewidth,
                color=color, zorder=zorder)
        
        
def plot_tree_on_blank(
    g,
    tree,
    *,
    width: float,
    height: float,
    title: str = "Tree on blank map",
    figsize=(6.5, 6.5),

    # base graph style
    show_base_graph: bool = True,
    base_edge_alpha: float = 0.08,
    base_edge_lw: float = 0.8,

    # tree style
    tree_color: str = "black",
    tree_alpha: float = 0.95,
    tree_lw: float = 2.2,

    # nodes style
    show_settled_nodes: bool = True,
    settled_node_size: float = 12.0,
    settled_node_alpha: float = 0.8,

    show_root: bool = True,
    root_size: float = 60.0,
    root_marker: str = "x",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 1) base graph (optional)
    if show_base_graph:
        _plot_edges(
            ax, g,
            alpha=base_edge_alpha,
            linewidth=base_edge_lw,
            show_portal=True,  # 你原函数里 portal 会更醒目
        )

    # 2) tree edges
    tree_edge_ids = getattr(tree, "tree_edge_ids", None)
    if tree_edge_ids is None or len(tree_edge_ids) == 0:
        # fallback: build from parent_edge_id
        tree_edge_ids = set(getattr(tree, "parent_edge_id", {}).values())

    _plot_edges_by_ids(
        ax, g, tree_edge_ids,
        alpha=tree_alpha, linewidth=tree_lw, color=tree_color, zorder=6
    )

    # 3) settled nodes (optional)
    if show_settled_nodes:
        xs, ys = [], []
        for nid in getattr(tree, "settled_nodes", set()):
            n = g.nodes[nid]
            xs.append(n.x); ys.append(n.y)
        if xs:
            ax.scatter(xs, ys, s=settled_node_size, alpha=settled_node_alpha, zorder=7)

    # 4) root marker
    if show_root:
        r = int(tree.root_node_id)
        rn = g.nodes[r]
        ax.scatter([rn.x], [rn.y], s=root_size, marker=root_marker, zorder=8)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    
def plot_tree_on_costmap(
    costmap,
    g,
    tree,
    *,
    title: str = "Tree on costmap",
    figsize=(6.5, 6.5),
    origin: str = "lower",
    show_colorbar: bool = True,

    show_base_graph: bool = True,
    base_edge_alpha: float = 0.08,
    base_edge_lw: float = 0.8,

    tree_color: str = "black",
    tree_alpha: float = 0.95,
    tree_lw: float = 2.2,

    show_settled_nodes: bool = True,
    settled_node_size: float = 12.0,
    settled_node_alpha: float = 0.8,

    show_root: bool = True,
    root_size: float = 60.0,
    root_marker: str = "x",

    inf_as_max: bool = True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    T = np.array(costmap, dtype=float)
    if inf_as_max:
        finite = T[np.isfinite(T)]
        maxv = finite.max() if finite.size else 1.0
        T = T.copy()
        T[~np.isfinite(T)] = maxv * 1.2

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(T, origin=origin)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    H, W = T.shape
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")

    if show_base_graph:
        _plot_edges(
            ax, g,
            alpha=base_edge_alpha,
            linewidth=base_edge_lw,
            show_portal=True,
        )

    tree_edge_ids = getattr(tree, "tree_edge_ids", None)
    if tree_edge_ids is None or len(tree_edge_ids) == 0:
        tree_edge_ids = set(getattr(tree, "parent_edge_id", {}).values())

    _plot_edges_by_ids(
        ax, g, tree_edge_ids,
        alpha=tree_alpha, linewidth=tree_lw, color=tree_color, zorder=6
    )

    if show_settled_nodes:
        xs, ys = [], []
        for nid in getattr(tree, "settled_nodes", set()):
            n = g.nodes[nid]
            xs.append(n.x); ys.append(n.y)
        if xs:
            ax.scatter(xs, ys, s=settled_node_size, alpha=settled_node_alpha, zorder=7)

    if show_root:
        r = int(tree.root_node_id)
        rn = g.nodes[r]
        ax.scatter([rn.x], [rn.y], s=root_size, marker=root_marker, zorder=8)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    
    
    
# def _plot_path_edges(
#     ax,
#     g,
#     edge_ids,
#     *,
#     color="red",
#     alpha=0.95,
#     linewidth=3.0,
#     zorder=10,
# ):
#     _plot_edges_by_ids(
#         ax, g, edge_ids,
#         alpha=alpha,
#         linewidth=linewidth,
#         color=color,
#         zorder=zorder,
#     )

def _plot_path_edges(
    ax,
    g,
    edge_ids: Sequence[int],
    *,
    color: Any = "red",
    alpha: float = 0.95,
    linewidth: float = 3.0,
    zorder: int = 12,
    label: Optional[str] = None,
):
    # label only once
    first = True
    for eid in edge_ids:
        e = g.edges[int(eid)]
        u = g.nodes[e.u]
        v = g.nodes[e.v]
        if first and label is not None:
            ax.plot([u.x, v.x], [u.y, v.y], alpha=alpha, linewidth=linewidth, color=color, zorder=zorder, label=label)
            first = False
        else:
            ax.plot([u.x, v.x], [u.y, v.y], alpha=alpha, linewidth=linewidth, color=color, zorder=zorder)


               
def plot_path_on_blank(
    g,
    path,
    *,
    width: float,
    height: float,
    title: str = "Recovered path (blank map)",
    figsize=(6.5, 6.5),

    show_base_graph: bool = True,
    base_edge_alpha: float = 0.06,
    base_edge_lw: float = 0.8,
    show_portal: bool = False,

    show_tree: bool = False,
    tree=None,
    tree_color: str = "black",
    tree_alpha: float = 0.25,
    tree_lw: float = 1.5,

    path_color: str = "red",
    path_alpha: float = 0.95,
    path_lw: float = 3.2,

    show_start_end: bool = True,
    start_color: str = "red",
    end_color: str = "blue",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    if show_base_graph:
        _plot_edges(ax, g, alpha=base_edge_alpha, linewidth=base_edge_lw, show_portal=show_portal)

    if show_tree and tree is not None:
        tree_edge_ids = getattr(tree, "tree_edge_ids", None)
        if not tree_edge_ids:
            tree_edge_ids = set(getattr(tree, "parent_edge_id", {}).values())
        _plot_edges_by_ids(ax, g, tree_edge_ids, alpha=tree_alpha, linewidth=tree_lw, color=tree_color, zorder=6)

    # path
    _plot_path_edges(ax, g, path.edge_ids, color=path_color, alpha=path_alpha, linewidth=path_lw, zorder=10)

    if show_start_end and path.nodes:
        s = g.nodes[path.nodes[0]]
        t = g.nodes[path.nodes[-1]]
        ax.scatter([s.x], [s.y], s=70, color=start_color, zorder=12, marker="o")
        ax.scatter([t.x], [t.y], s=70, color=end_color, zorder=12, marker="o")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    
def plot_path_on_costmap(
    costmap,
    g,
    path,
    *,
    title: str = "Recovered path (costmap)",
    figsize=(6.5, 6.5),
    origin: str = "lower",
    show_colorbar: bool = True,

    show_base_graph: bool = True,
    base_edge_alpha: float = 0.06,
    base_edge_lw: float = 0.8,
    show_portal: bool = False,
    
    show_tree: bool = False,
    tree=None,
    tree_color: str = "black",
    tree_alpha: float = 0.25,
    tree_lw: float = 1.5,

    path_color: str = "red",
    path_alpha: float = 0.95,
    path_lw: float = 3.2,

    show_start_end: bool = True,
    start_color: str = "red",
    end_color: str = "blue",

    inf_as_max: bool = True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    T = np.array(costmap, dtype=float)
    if inf_as_max:
        finite = T[np.isfinite(T)]
        maxv = finite.max() if finite.size else 1.0
        T = T.copy()
        T[~np.isfinite(T)] = maxv * 1.2

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(T, origin=origin)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    H, W = T.shape
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")

    if show_base_graph:
        _plot_edges(ax, g, alpha=base_edge_alpha, linewidth=base_edge_lw, show_portal=show_portal)

    if show_tree and tree is not None:
        tree_edge_ids = getattr(tree, "tree_edge_ids", None)
        if not tree_edge_ids:
            tree_edge_ids = set(getattr(tree, "parent_edge_id", {}).values())
        _plot_edges_by_ids(ax, g, tree_edge_ids, alpha=tree_alpha, linewidth=tree_lw, color=tree_color, zorder=6)

    _plot_path_edges(ax, g, path.edge_ids, color=path_color, alpha=path_alpha, linewidth=path_lw, zorder=10)

    if show_start_end and path.nodes:
        s = g.nodes[path.nodes[0]]
        t = g.nodes[path.nodes[-1]]
        ax.scatter([s.x], [s.y], s=70, color=start_color, zorder=12, marker="o")
        ax.scatter([t.x], [t.y], s=70, color=end_color, zorder=12, marker="o")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    

    
    
    
    
# ---------- helper: draw selected edges by edge ids ----------
# def _plot_edges_by_ids(
#     ax,
#     g,
#     edge_ids: Iterable[int],
#     *,
#     alpha: float = 0.9,
#     linewidth: float = 2.0,
#     color: Any = "black",
#     zorder: int = 5,
# ):
#     for eid in edge_ids:
#         e = g.edges[int(eid)]
#         u = g.nodes[e.u]
#         v = g.nodes[e.v]
#         ax.plot([u.x, v.x], [u.y, v.y], alpha=alpha, linewidth=linewidth, color=color, zorder=zorder)


# # ---------- helper: draw a path (edge id list) ----------
# def _plot_path_edges(
#     ax,
#     g,
#     edge_ids: Sequence[int],
#     *,
#     color: Any = "red",
#     alpha: float = 0.95,
#     linewidth: float = 3.0,
#     zorder: int = 12,
#     label: Optional[str] = None,
# ):
#     # label only once
#     first = True
#     for eid in edge_ids:
#         e = g.edges[int(eid)]
#         u = g.nodes[e.u]
#         v = g.nodes[e.v]
#         if first and label is not None:
#             ax.plot([u.x, v.x], [u.y, v.y], alpha=alpha, linewidth=linewidth, color=color, zorder=zorder, label=label)
#             first = False
#         else:
#             ax.plot([u.x, v.x], [u.y, v.y], alpha=alpha, linewidth=linewidth, color=color, zorder=zorder)


# ---------- helper: draw goals ----------
def _plot_goals(
    ax,
    goals: Sequence[Any],
    *,
    start_color: Any = "red",
    end_color: Any = "blue",
    size: float = 80,
    alpha: float = 0.95,
    zorder: int = 20,
    fontsize: Optional[int] = 10

):
    # Goal expected fields: .tag in {"sta_plot_goalsrt","end"} and either .pose (x,y) or .node_id
    for g0 in goals:
        tag = getattr(g0, "tag", "")
        name = getattr(g0, "name", None)

        if hasattr(g0, "pose") and g0.pose is not None:
            x, y = float(g0.pose[0]), float(g0.pose[1])
        else:
            # fallback: node_id -> graph node coord
            nid = int(g0.node_id)
            n = ax._viz_graph.nodes[nid]  # will be set before calling
            x, y = float(n.x), float(n.y)

        if tag == "start":
            c = start_color
        elif tag == "end":
            c = end_color
        else:
            c = "magenta"

        ax.scatter([x], [y], s=size, alpha=alpha, color=c, zorder=zorder)
        if name:
            ax.text(x, y, f" {name}", fontsize=fontsize, alpha=0.9, zorder=zorder)


# ---------- main: combined viz ----------
def plot_multi_tree_overlap_paths(
    g,
    *,
    width: float,
    height: float,

    # optional background
    costmap: Optional[np.ndarray] = None,
    origin: str = "lower",
    show_colorbar: bool = True,
    inf_as_max: bool = True,

    # inputs
    trees: Sequence[Any],                 # list of TreeResult
    tree_labels: Optional[Sequence[str]] = None,
    tree_colors: Optional[Sequence[Any]] = None,

    overlap_nodes: Optional[Set[int]] = None,
    overlap_ranked: Optional[Sequence[Tuple[int, float]]] = None,  # [(nid, total_energy),...], optional
    best_overlap_node: Optional[int] = None,                       # optional override

    paths: Optional[Sequence[Any]] = None,     # list of PathResult
    path_labels: Optional[Sequence[str]] = None,
    path_colors: Optional[Sequence[Any]] = None,

    goals: Optional[Sequence[Any]] = None,     # list of Goal objects
    meeting_marker: str = "*",

    # style: base graph edges
    show_base_graph: bool = True,
    base_edge_alpha: float = 0.06,
    base_edge_lw: float = 0.8,

    # style: tree edges
    show_trees: bool = True,
    tree_alpha: float = 0.25,
    tree_lw: float = 1.6,

    # style: overlap nodes (energy-colored)
    show_overlap_nodes: bool = True,
    overlap_point_size: float = 22.0,
    overlap_alpha: float = 0.85,
    overlap_cmap: str = "viridis",       # energy -> color
    overlap_use_log: bool = False,       # optional: log scale energy
    overlap_colorbar: bool = True,

    # style: paths
    show_paths: bool = True,
    path_alpha: float = 0.95,
    path_lw: float = 3.2,

    # style: goals / best meeting
    show_goals: bool = True,
    start_goal_color: Any = "red",
    end_goal_color: Any = "blue",
    goal_size: float = 85.0,

    show_best_overlap: bool = True,
    best_overlap_color: Any = "gold",
    best_overlap_size: float = 140.0,

    # misc
    title: str = "Multi-tree + overlap + paths",
    figsize: Tuple[float, float] = (7.2, 7.2),
    grid: bool = True,
    legend: bool = True,
    
    font_cfg: Optional[dict] = None,

):
    """
    Combined visualization:
      1) draw base graph (optional)
      2) draw multiple trees (different colors)
      3) draw overlap nodes colored by total energy (or by provided overlap_ranked)
      4) draw multiple paths (different colors)
      5) draw goals + best overlap node

    Requirements:
      - TreeResult has: .tree_edge_ids (or .parent_edge_id), .energy_cost, .settled_nodes, .root_node_id
      - PathResult has: .edge_ids, .nodes
      - Goal has: .tag in {"start","end"}, and either .pose or .node_id
    """
    # defaults
    if tree_colors is None:
        tree_colors = ["tab:green", "tab:orange", "tab:purple"]
    if path_colors is None:
        path_colors = ["tab:red", "tab:blue", "tab:brown"]

    if tree_labels is None:
        tree_labels = [getattr(t, "agent_name", f"tree{i}") for i, t in enumerate(trees)]
    if paths is not None and path_labels is None:
        path_labels = [getattr(p, "agent_name", f"path{i}") for i, p in enumerate(paths)]

    # prepare figure/axes
    fig, ax = plt.subplots(figsize=figsize)

    # attach graph reference for goal plotting fallback
    ax._viz_graph = g  # internal

    # background: costmap or blank
    if costmap is not None:
        T = np.array(costmap, dtype=float)
        if inf_as_max:
            finite = T[np.isfinite(T)]
            maxv = finite.max() if finite.size else 1.0
            T = T.copy()
            T[~np.isfinite(T)] = maxv * 1.2
        im = ax.imshow(T, origin=origin)
        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        H, W = T.shape
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            
    if font_cfg is None:
        font_cfg = {}

    title_fs  = font_cfg.get("title", 14)
    label_fs  = font_cfg.get("label", 12)
    tick_fs   = font_cfg.get("tick", 11)
    legend_fs = font_cfg.get("legend", 11)
    text_fs   = font_cfg.get("text", 10)

    ax.set_aspect("equal", adjustable="box")

    # 1) base graph
    if show_base_graph:
        # reuse your existing _plot_edges if present; otherwise draw all edges lightly
        if "_plot_edges" in globals():
            _plot_edges(ax, g, alpha=base_edge_alpha, linewidth=base_edge_lw, show_portal=True)
        else:
            _plot_edges_by_ids(ax, g, [e.id for e in g.edges], alpha=base_edge_alpha, linewidth=base_edge_lw, color="gray", zorder=1)

    # 2) trees
    if show_trees:
        for i, t in enumerate(trees):
            color = tree_colors[i % len(tree_colors)]
            label = tree_labels[i] if tree_labels else None
            edge_ids = getattr(t, "tree_edge_ids", None)
            if not edge_ids:
                edge_ids = set(getattr(t, "parent_edge_id", {}).values())
            # plot with label only once
            _plot_edges_by_ids(ax, g, edge_ids, alpha=tree_alpha, linewidth=tree_lw, color=color, zorder=4)
            if legend and label:
                # add a dummy handle for legend
                ax.plot([], [], color=color, alpha=tree_alpha, linewidth=tree_lw, label=f"tree:{label}")

    # 3) overlap nodes colored by energy
    energies = None
    if show_overlap_nodes and overlap_nodes:
        # determine energies for overlap nodes
        if overlap_ranked is not None:
            # overlap_ranked is a list of (nid, total_energy)
            energy_map = {int(nid): float(e) for nid, e in overlap_ranked}
            xs, ys, es = [], [], []
            for nid in overlap_nodes:
                if nid in energy_map:
                    n = g.nodes[nid]
                    xs.append(n.x); ys.append(n.y); es.append(energy_map[nid])
            energies = np.array(es, dtype=float) if es else None
        else:
            # default: sum tree.energy_cost
            xs, ys, es = [], [], []
            for nid in overlap_nodes:
                tot = 0.0
                ok = True
                for t in trees:
                    ec = t.energy_cost.get(nid, None)
                    if ec is None:
                        ok = False
                        break
                    tot += float(ec)
                if ok:
                    n = g.nodes[nid]
                    xs.append(n.x); ys.append(n.y); es.append(tot)
            energies = np.array(es, dtype=float) if es else None

        if energies is not None and energies.size > 0:
            cvals = np.log1p(energies) if overlap_use_log else energies
            sc = ax.scatter(
                xs, ys,
                s=overlap_point_size,
                c=cvals,
                cmap=overlap_cmap,
                alpha=overlap_alpha,
                zorder=9,
                label="overlap nodes" if legend else None,
            )
            if overlap_colorbar:
                cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("total energy (log1p)" if overlap_use_log else "total energy")

    # 4) best overlap node
    if show_best_overlap:
        if best_overlap_node is None and overlap_ranked is not None and len(overlap_ranked) > 0:
            best_overlap_node = int(overlap_ranked[0][0])
        if best_overlap_node is not None:
            bn = g.nodes[int(best_overlap_node)]
            ax.scatter(
                [bn.x], [bn.y],
                s=best_overlap_size,
                color=best_overlap_color,
                marker=meeting_marker,
                zorder=15,
                label="best overlap" if legend else None,
                edgecolors="k",
                linewidths=0.5,
            )

    # 5) paths
    if show_paths and paths is not None:
        for i, p in enumerate(paths):
            color = path_colors[i % len(path_colors)]
            label = None
            if legend and path_labels is not None:
                label = f"path:{path_labels[i]}"
            _plot_path_edges(
                ax, g, p.edge_ids,
                color=color, alpha=path_alpha, linewidth=path_lw, zorder=12,
                label=label,
            )

            # mark start/end of each path
            if p.nodes:
                s = g.nodes[p.nodes[0]]
                tnode = g.nodes[p.nodes[-1]]
                ax.scatter([s.x], [s.y], s=60, color=color, alpha=0.95, zorder=13, marker="o")
                ax.scatter([tnode.x], [tnode.y], s=60, color=color, alpha=0.95, zorder=13, marker="o")

    # 6) goals
    if show_goals and goals is not None:
        _plot_goals(
            ax,
            goals,
            start_color=start_goal_color,
            end_color=end_goal_color,
            size=goal_size,
            alpha=0.95,
            zorder=18,
            fontsize = text_fs,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel("x", fontsize=label_fs)
    ax.set_ylabel("y", fontsize=label_fs)

    ax.tick_params(axis="both", labelsize=tick_fs)
    if legend:
        ax.legend(loc="best", fontsize=legend_fs)


    plt.tight_layout()
    plt.show()
