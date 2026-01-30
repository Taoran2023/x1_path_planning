#!/usr/bin/env python
# coding: utf-8

# In[1]:


from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

# ================== Region Tag Enum ==================
class RegionTag:
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NO_TRAVERSABLE = "no-traversable"
    DOOR = "door"
    CUSTOM = "custom"

# ================== Base Region ==================
@dataclass
class Region(ABC):
    name: str
    tag: str
    score: float   # traversability cost multiplier
    print("region added")

    @abstractmethod
    def contains(self, x: float, y: float) -> bool:
        """Check if (x,y) is inside region"""
        pass
    
@dataclass
class RectRegion(Region):
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x, y):
        return (self.x0 <= x <= self.x1) and (self.y0 <= y <= self.y1)

    
@dataclass
class CircleRegion(Region):
    cx: float
    cy: float
    r: float

    def contains(self, x, y):
            return (x - self.cx)**2 + (y - self.cy)**2 <= self.r**2

class RegionManager:
    def __init__(self):
        self.regions = []

    def add_region(self, region: Region):
        self.regions.append(region)

    def query_score(self, x, y, default=1.0):
        """Return traversability score at (x,y)"""
        for r in self.regions:
            if r.contains(x, y):
                return r.score
        return default

    def query_tag(self, x, y, default="unknown"):
        for r in self.regions:
            if r.contains(x, y):
                return r.tag
        return default

    def get_region_by_name(self, name: str):
        """Return Region object by name, or None if not found."""
        for r in self.regions:
            if r.name == name:
                return r
        return None
    def compute_boundary_masks(self, width: float, height: float, res: float, connectivity: int = 4):
        """
        Precompute boundary masks for every region by rasterizing its occupancy on a grid and
        extracting boundary cells using 4-neighborhood (default) or 8-neighborhood.

        Parameters
        ----------
        width, height : float
            World size. Grid spans [0,width) x [0,height)
        res : float
            Cell resolution. Grid size: W = ceil(width/res), H = ceil(height/res)
        connectivity : int
            4 or 8. For this demo, you asked for 4-neighborhood.

        Outputs
        -------
        self.boundary_masks[name] : bool array (H,W)
            True where cell center is inside region AND at least one neighbor is outside.
        self.occupancy_masks[name] : bool array (H,W) (optional cache)
        """
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if res <= 0:
            raise ValueError("res must be > 0")

        W = int(np.ceil(width / res))
        H = int(np.ceil(height / res))

        # grid cell centers
        xs = (np.arange(W) + 0.5) * res
        ys = (np.arange(H) + 0.5) * res

        self.boundary_masks = {}
        self.occupancy_masks = {}

        for r in self.regions:
            # occupancy mask: occ[y,x]
            occ = np.zeros((H, W), dtype=bool)

            # Vectorization is hard because Region.contains is per-point;
            # For demo size this nested loop is fine.
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    occ[iy, ix] = bool(r.contains(float(x), float(y)))

            # boundary extraction
            # boundary = occ & (has any neighbor outside)
            boundary = np.zeros_like(occ)

            # 4-neighborhood: up/down/left/right
            # outside = neighbor cell is False (or out-of-bounds treated as outside)
            # We compute "interior neighbors" by shifting occ; out-of-bounds => False.
            up = np.zeros_like(occ);    up[1:, :]  = occ[:-1, :]
            down = np.zeros_like(occ);  down[:-1,:]= occ[1:,  :]
            left = np.zeros_like(occ);  left[:, 1:] = occ[:, :-1]
            right= np.zeros_like(occ);  right[:, :-1]= occ[:, 1:]

            # A cell is boundary if it is inside and any 4-neighbor is outside
            boundary = occ & (~up | ~down | ~left | ~right)

            if connectivity == 8:
                # add diagonals for 8-neighborhood
                ul = np.zeros_like(occ); ul[1:, 1:] = occ[:-1, :-1]
                ur = np.zeros_like(occ); ur[1:, :-1]= occ[:-1, 1:]
                dl = np.zeros_like(occ); dl[:-1,1:] = occ[1:, :-1]
                dr = np.zeros_like(occ); dr[:-1,:-1]= occ[1:, 1:]
                boundary = boundary & (~ul | ~ur | ~dl | ~dr | True)  # keep previous condition
                # The above line is a bit clunky; simplest:
                boundary = occ & (~up | ~down | ~left | ~right | ~ul | ~ur | ~dl | ~dr)

            self.occupancy_masks[r.name] = occ
            self.boundary_masks[r.name] = boundary

        # return grid size for convenience
        return H, W







