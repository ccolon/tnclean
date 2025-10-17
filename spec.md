# tnclean — Transport Network Cleaner

A small Python package to **clean & simplify transport network lines**.  
**Input:** Shapefile or GeoJSON with `LineString`/`MultiLineString`.  
**Output:** **GeoJSON (RFC 7946)** with **LineString-only**, de-duplicated, split at intersections, and **topology-preservingly simplified** geometries.

---

## Why

Real-world network layers (roads/rail/sea links) often contain MultiLineStrings, near-duplicates, partial overlaps, and unsnapped intersections. These break downstream routing, loading, and ABM pipelines. **tnclean** applies a fixed, topology-first pipeline to produce a consistent, minimal, **LineString-only** network you can trust.

---

## Features (v0)

- **Explode** MultiLineStrings → LineStrings
- **Snap** coordinates to a metric grid to stabilize topology
- **Split** edges at true intersections (edge–edge, node–edge)
- **De-overlap**:
  - Detect **partial overlaps** and **split** A/B where they coincide
  - **Deduplicate** identical edges between the **same endpoints**, deterministically **keeping the shortest**
  - Option to **keep parallels** even if endpoints match
- **Simplify (topology-preserving)**: global TopoJSON-style arcs (node-anchored), then prune tiny non-anchor slivers
- **Attributes**: on split, **inherit all fields**; on merge/dedup/dissolve, **concatenate unique categorical values** (comma-separated)
- **LineString-only** **GeoJSON** export + **report JSON** (counts, lengths, duplicates removed, timings)

---

## Install (dev)

```bash
# clone your repo; then in editable mode:
conda install -c conda-forge geopandas shapely>=2 pyproj rtree
uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"
# dev deps you'll want: geopandas, shapely>=2, pyproj, rtree, typer, pytest, hypothesis, ruff, mypy
```

> **Python 3.11+ required**. Package manager: Conda/Mamba + uv/pip + hatchling.

---

## Quick start

### CLI

```bash
tnclean input.geojson output.geojson   --metric-crs auto   --snap-precision 1.0   --overlap-tolerance 0.5   --simplify-tolerance 2.0   --prune-epsilon 0.1   --dedup-policy keep-shortest   --pairing-batch-size 100000   --pairing-bearing-delta-deg 5.0   --pairing-length-ratio-min 0.5   --pairing-length-ratio-max 2.0   --pairing-overlap-alpha 0.9
```

### Python

```python
from tnclean import clean_network, CleanConfig

cfg = CleanConfig(
    metric_crs="auto",            # all tolerances in meters
    snap_precision=1.0,
    overlap_tolerance=0.5,
    simplify_tolerance=2.0,
    prune_epsilon=0.1,
    dedup_policy="keep-shortest",
    keep_parallel_edges=False,    # set True to keep both when endpoints match
)

gdf, report = clean_network("input.geojson", "output.geojson", cfg)
print(report)   # dict with counts/lengths/timings/provenance
```

---

## Fixed pipeline (v0)

**explode → snap → split → de-overlap → simplify → export**

1. **Explode**: MultiLineString → LineString, attributes preserved.  
2. **Snap**: quantize coords to a **metric grid** (`snap_precision`) to stabilize intersections/duplicates.  
3. **Split**: create nodes at **true intersections**; split edges so each feature is between two nodes.  
4. **De-overlap** (partial-overlap aware):
   - Pair candidates using **STRtree** on buffered envelopes; cheap pre-filters (bearing Δ, length ratio, projected overlap).
   - For each overlapping pair A/B: compute the **lineal overlap**, add its endpoints as **cut points**, and **split** A, B.
   - After normalization, treat two sub-edges as duplicates iff they share the same **unordered node pair** `{u,v}` and are within tolerance; **keep the shortest** (or keep both if configured).
5. **Simplify** (**topology-preserving**):
   - Build **shared arcs** (TopoJSON-style) between **anchored graph nodes** and simplify arcs (VW/RDP) with **nodes fixed**.
   - **Prune** tiny **non-anchor** slivers (≤ `prune_epsilon`) and reassemble edges.
6. **Export**: LineString-only **GeoJSON** + **report** (provenance & stats).

---

## Configuration (key knobs)

- **Units**: all tolerances are **meters**; `metric_crs="auto"` reprojects to a suitable projected CRS.  
- `snap_precision` *(m)*: grid size for coordinate quantization.  
- `overlap_tolerance` *(m)*: proximity tolerance for overlap detection & equality checks.  
- `simplify_tolerance` *(m)*: geometric tolerance for arc simplification (node-anchored).  
- `prune_epsilon` *(m)*: drop tiny non-anchor segments after simplify.  
- `dedup_policy`: how to resolve duplicates with same `{u,v}` (`keep-shortest` | `keep-first` | `keep-longest`).  
- `keep_parallel_edges`: if **True**, skip dedup when endpoints match (retain parallels).  
- `preserve_fields`: list of field names to preserve; None = preserve all categorical fields.  
- Pairing filters (performance/precision):
  - `pairing_bearing_delta_deg` (default 5°)
  - `pairing_length_ratio_[min,max]` (default 0.5–2.0)
  - `pairing_overlap_alpha` (default 0.9, overlap/common length ratio)
  - `pairing_batch_size` for streaming STRtree queries (adaptive online batching)

---

## Specification details

### Auto CRS Selection (`metric_crs="auto"`)

- **Dataset diagonal ≤ 800km**: Select UTM zone from centroid (EPSG 326xx/327xx)
- **|latitude| ≥ 84°**: Use Universal Polar Stereographic (EPSG 5041/5042)  
- **Otherwise**: Custom Lambert Azimuthal Equal-Area centered at dataset centroid (`+proj=laea +lat_0=... +lon_0=... +units=m`)
- **Output CRS**: Always preserve original input CRS

### Data-Driven Tolerance Validation

**Adaptive bounds with auto-clamp (warnings) or `--strict` (fail):**
- Compute: `V05/V50` = 5th/50th percentile vertex spacing, `L50` = median edge length, `D` = dataset diagonal
- `snap_precision` ∈ [0.25×V05, 0.75×V50]
- `overlap_tolerance` ∈ (0, 0.5×snap_precision], default = 0.5×snap_precision  
- `simplify_tolerance` ∈ [snap_precision, min(2×L50, 0.05×D)]
- `prune_epsilon` ∈ (0, min(0.5×simplify_tolerance, overlap_tolerance, 0.1×V50))

### Performance & Memory Management

- **STRtree**: Use Shapely's implementation
- **Adaptive batching**: Online measurement and adjustment of `pairing_batch_size`
- **Memory management**: Hybrid streaming with spill-to-disk for intermediate results

### Error Handling

- **Invalid geometries**: **Fail** (no skip/warn)
- **Empty results**: **Fail** (should not happen)
- **Tolerance violations**: Auto-clamp with warnings, or fail if `--strict`

### Node identity (QCK)

- **Quantized-Coordinate Key (QCK)** after snap:  
  `ix = floor(x / p)`, `iy = floor(y / p)`, where `p = snap_precision`.  
  `NodeID = hash(ix, iy, CRS, p)` (or `f"{ix}:{iy}:{epsg}:{p}"`).  
- **Edge key**: unordered pair `{min(NodeID_u, NodeID_v), max(...)};` used for dedup.

### Candidate pairing at scale

- Global **STRtree** over edge envelopes.  
- For each edge, query **bbox buffered** by `overlap_tolerance`.  
- Pre-filters:
  - **Bearing** difference ≤ `pairing_bearing_delta_deg`
  - **Length ratio** within `[min, max]`
  - **Projected overlap** ratio ≥ `pairing_overlap_alpha`  
- Stream in **batches** (`pairing_batch_size`), de-dup pairs via `{min(id), max(id)}`.

### Partial-overlap normalization

- For each candidate pair, compute **lineal intersection**; collect its endpoints; **split** both edges at those points; proceed to dedup.

### Deduplication

- If multiple edges share `{u,v}` and are within `overlap_tolerance` (Hausdorff), resolve per `dedup_policy` (default **keep-shortest**).  
- **keep_parallel_edges=True** will **skip** this removal even when `{u,v}` matches.

### Attributes

- **On split**: each part **inherits all attributes** unchanged.  
- **On merge/dedup/dissolve**: for each field in `preserve_fields` (or all if None), concatenate unique values:
  - **Ordering**: stable (input order).
  - **Separator**: comma for int fields, comma + quotes for string fields (if string contains comma).
  - Fields not in `preserve_fields` are dropped.

### Topology-preserving simplify

- **Quantization** aligned to `snap_precision`.  
- Build **shared arcs** between **anchored nodes**; simplify arcs (VW/RDP) **without moving anchors**.  
- **Prune** non-anchor arcs ≤ `prune_epsilon`, then reassemble edges.  
- Guarantees: **connectivity/adjacency/containment** preserved; node degrees unchanged.

### Output & report

- **GeoJSON** (RFC 7946), **LineString-only**.  
- **Report JSON** includes: input/output counts, total length before/after, #splits, #duplicates removed, simplify/prune deltas, timings, CRS & all tolerances, algorithm versions.

---

## API surface (v0)

```python
from dataclasses import dataclass
from typing import Literal, Optional, Iterable, Dict, Any, Tuple
import geopandas as gpd

PrecisionMode = Literal["grid","decimal"]
SimplifyMode  = Literal["topology","rdp","vw"]
DedupPolicy   = Literal["keep-shortest","keep-first","keep-longest"]

@dataclass
class CleanConfig:
    metric_crs: Literal["auto"] | str = "auto"
    crs: Optional[str] = None
    snap_precision: float = 1.0
    snap_mode: PrecisionMode = "grid"
    overlap_tolerance: float = 0.5
    simplify_tolerance: float = 2.0
    prune_epsilon: float = 0.1
    dedup_policy: DedupPolicy = "keep-shortest"
    keep_parallel_edges: bool = False
    pairing_batch_size: int = 100_000
    pairing_bearing_delta_deg: float = 5.0
    pairing_length_ratio_min: float = 0.5
    pairing_length_ratio_max: float = 2.0
    pairing_overlap_alpha: float = 0.9
    simplify_mode: SimplifyMode = "topology"
    preserve_fields: Optional[Iterable[str]] = None
    dissolve_fields: Optional[Iterable[str]] = None
    metadata: Dict[str, Any] = None
    n_workers: int = 0
    strict_validation: bool = False
    progress_bar: bool = True
    verbose: int = 0

def clean_network(input_path: str, output_path: str, config: CleanConfig = CleanConfig()
                  ) -> Tuple[gpd.GeoDataFrame, dict]:
    """Run the fixed pipeline and return (gdf, report)."""
```

**CLI entry point:** `tnclean` with flags mirroring `CleanConfig`.  
**Config file support:** `--config config.yaml` or `config.json`  
**Progress & logging:** `--progress/--no-progress`, `--verbose/-v` (multiple levels)  
**Validation modes:** `--strict` (fail on tolerance violations)

---

## Design decisions (finalized)

1. **Units & CRS:** All tolerances are **meters**; `metric_crs="auto"` selects UTM (diagonal ≤800km), UPS (|lat|≥84°), or custom LAEA. Output preserves original CRS.  
2. **Pipeline order:** **explode → snap → split → de-overlap → simplify → export**.  
3. **Overlap semantics:** After snap→split, duplicates are defined on the **same node pair** `{u,v}`; partial overlaps are **split first**, then dedup.  
4. **Simplification backend:** **Shapely STRtree** + **Global TopoJSON-style** arcs; quantization aligned to `snap_precision`; **nodes anchored**.  
5. **Node identity:** **QCK** (Quantized-Coordinate Key) from snapped coords; `NodeID = hash(ix,iy,CRS,p)`.  
6. **Duplicate resolution:** deterministic **`keep-shortest`** by default; boolean **`keep_parallel_edges`** to retain both.  
7. **Pairing at scale:** **Shapely STRtree** + buffered bbox queries; **adaptive online batching**; hybrid streaming with spill-to-disk.  
8. **Error handling:** **Fail on invalid geometries**, empty results, strict validation with data-driven adaptive bounds.  
9. **Attributes:** User specifies `preserve_fields` (None = all); stable input order; comma separator with quote handling.  
10. **CLI/Testing:** Progress bars, verbose logging, YAML/JSON config support. Hypothesis property-based testing with synthetic data.

---

## Repo layout (suggested)

```
src/tnclean/
  __init__.py
  types.py
  pipeline.py
  ops.py          # I/O, CRS, explode, snap, split, de-overlap, simplify hooks
  graph.py        # QCK utils, edge keys
  topo.py         # TopoJSON-style arc builder & simplifier
  cli.py
tests/
  test_e2e_pipeline.py
  test_qck.py
  test_split_overlap.py
  test_simplify_topology.py
examples/
  messy_lines.geojson
```

---

## Roadmap

- [ ] v0: end-to-end pipeline, CLI, report, synthetic test fixtures with Hypothesis  
- [ ] v0.x: performance tuning, adaptive streaming, richer diagnostics  
- [ ] v1: optional tiling for very large datasets; plugin hooks; advanced attribute policies

---

## License

TBD (suggest: BSD-3-Clause or MIT).

---

## Contributing

PRs welcome! Please run `ruff format && ruff check && mypy && pytest -q` before submitting.
