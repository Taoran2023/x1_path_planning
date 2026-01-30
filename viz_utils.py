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

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    return domain_colors


    
    
#     --------------------------------------------------------------------------  #





# viz_utils.py



def plot_graph_edges(
    ax: plt.Axes,
    g,
    *,
    etypes: Optional[Union[str, Iterable[str]]] = None,
    show_portal_emphasis: bool = True,
    alpha: float = 0.25,
    linewidth: float = 1.0,
    portal_alpha: float = 0.6,
    portal_linewidth: float = 1.5,
    zorder: int = 3,
):
    """
    Draw ONLY edges of graph on the given matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to draw on (so you can overlay on blank or costmap).
    g : Graph
        Your graph object (must have g.edges and g.nodes).
    etypes : None | str | Iterable[str]
        If provided, only draw edges whose e.etype is in etypes.
        Examples:
          - etypes="portal"
          - etypes=["intra", "portal"]
    show_portal_emphasis : bool
        If True, draw portal edges using (portal_alpha, portal_linewidth).
    alpha, linewidth :
        Style for non-portal (or non-emphasized) edges.
    portal_alpha, portal_linewidth :
        Style for portal edges if show_portal_emphasis is True.
    zorder : int
        Matplotlib z-order for drawing.
    """
    if etypes is None:
        etype_set = None
    else:
        if isinstance(etypes, str):
            etype_set = {etypes}
        else:
            etype_set = set(etypes)

    for e in g.edges:
        et = getattr(e, "etype", None)
        if etype_set is not None and et not in etype_set:
            continue

        u = g.nodes[e.u]
        v = g.nodes[e.v]

        if show_portal_emphasis and et == "portal":
            ax.plot([u.x, v.x], [u.y, v.y],
                    alpha=portal_alpha, linewidth=portal_linewidth, zorder=zorder)
        else:
            ax.plot([u.x, v.x], [u.y, v.y],
                    alpha=alpha, linewidth=linewidth, zorder=zorder)
