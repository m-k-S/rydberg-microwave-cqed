
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
from build123d import BuildPart, Box, Cylinder, Align, Mode, Location, Locations, Axis, Compound, import_step
from math import atan2, degrees, sqrt

AxisName = Literal["x", "y", "z"]
FaceName = Literal["+x", "-x", "+y", "-y", "+z", "-z"]

@dataclass
class SMAMountSpec:
    step_path: str = "142-1701-201.step"
    align_axis: AxisName = "x"       # which cross-hole the SMA pin shares
    mate_face: FaceName = "-x"       # face the SMA flange mates to
    mate_offset: float = 0.0
    screw_clearance_diam: float = 2.18  # 2-56 close clearance (~0.086")
    screw_spacing: float | None = None  # None => auto-measure from STEP; fallback used if measurement fails
    screw_axis: AxisName | None = None  # auto: perpendicular to align_axis
    pin_center_hint: Tuple[float, float, float] | None = None  # after axis alignment; (x,y,z) in model units
    pin_axis_in_step: AxisName = "x"           # which axis the STEP's pin is along before any rotation
    extra_clock_deg: float = 0.0               # extra in-axis rotation after alignment (about align_axis)
    screw_hole_depth: float = 3.0            # depth of SMA flange screw holes (mm), blind

@dataclass
class BottomMountSpec:
    enabled: bool = True
    clearance_diam: float = 2.18       # 2-56 close clearance (~0.086")
    edge_margin_x: float = 4.0         # inset from ±X edges (mm)
    edge_margin_y: float = 4.0         # inset from ±Y edges (mm)
    face: FaceName = "-z"              # “bottom” face; flip to +z if you prefer
    hole_depth: float = 4.0                  # depth of bottom mounting holes (mm), blind

def _face_center_and_normal(L, W, H, face: FaceName):
    L2, W2, H2 = L/2, W/2, H/2
    d = {
        "+x": ((+L2, 0, 0), (+1, 0, 0)),
        "-x": ((-L2, 0, 0), (-1, 0, 0)),
        "+y": ((0, +W2, 0), (0, +1, 0)),
        "-y": ((0, -W2, 0), (0, -1, 0)),
        "+z": ((0, 0, +H2), (0, 0, +1)),
        "-z": ((0, 0, -H2), (0, 0, -1)),
    }
    return d[face]

def _axis_to_euler(axis: AxisName) -> Tuple[float, float, float]:
    return {"z": (0, 0, 0), "x": (0, 90, 0), "y": (90, 0, 0)}[axis]

# Rotate a Part so that 'from_axis' becomes 'to_axis' (principal axes only)
def _rotate_part_axis(part, from_axis: AxisName, to_axis: AxisName):
    """
    Rotate a Part so that 'from_axis' becomes 'to_axis'.
    Uses 90° rotations about principal axes.
    """
    if from_axis == to_axis:
        return part
    # Map of (from, to) -> list of (Axis, degrees) rotations
    seq = {
        ("z","x"): [(Axis.Y,  90)],
        ("z","y"): [(Axis.X,  90)],
        ("x","z"): [(Axis.Y, -90)],
        ("y","z"): [(Axis.X, -90)],
        ("x","y"): [(Axis.Z,  90)],
        ("y","x"): [(Axis.Z, -90)],
    }
    if (from_axis, to_axis) not in seq:
        # Handle two-step via Z if not in table (shouldn't happen with principal axes)
        inter = "z" if from_axis != "z" and to_axis != "z" else ("x" if from_axis == "z" else "y")
        part = _rotate_part_axis(part, from_axis, inter)
        return _rotate_part_axis(part, inter, to_axis)
    for ax, ang in seq[(from_axis, to_axis)]:
        part = part.rotate(ax, ang)
    return part

def _auto_screw_axis(axis: AxisName) -> AxisName:
    return {"x": "y", "y": "x", "z": "x"}[axis]


# Helper to attempt to measure SMA flange spacing from STEP geometry
def _auto_measure_sma_spacing(sma_part, align_axis: AxisName, assumed_hole_diam_range=(1.6, 2.6)) -> float | None:
    """
    Try to measure the flange 2-hole spacing from the STEP geometry.
    Heuristic: find circular edges that lie in planes perpendicular to the SMA pin axis
    and whose radii fall within a plausible 2-56 clearance range (~0.8–1.3 mm radius).
    Returns center-to-center spacing along the axis perpendicular to align_axis
    (i.e., 'y' for 'x', 'x' for 'y', 'x' for 'z').
    """
    try:
        # Import here to avoid hard dependency if build123d re-exports these
        from build123d import Edge, Axis, Location
    except Exception:
        pass  # proceed; we may still get the attributes from the objects

    # Rotate a copy so the pin axis is aligned as requested (already handled outside typically).
    # Here we assume caller passes in a part already rotated so its pin axis == align_axis.
    # We'll just search for small circular edges and compute the best symmetric pair.
    try:
        circles = []
        for e in sma_part.edges():
            # Some backends expose helpers:
            is_circ = getattr(e, "is_circle", None)
            if callable(is_circ):
                is_circ = e.is_circle()
            elif isinstance(is_circ, bool):
                pass
            else:
                # Try to access curve properties; if not available, skip
                continue

            if not is_circ:
                continue

            # Radius
            r = getattr(e, "radius", None)
            if r is None:
                # Some versions provide e.to_circle().radius
                to_circ = getattr(e, "to_circle", None)
                if callable(to_circ):
                    try:
                        r = to_circ().radius
                    except Exception:
                        continue
            if r is None:
                continue

            # Filter plausible 2-56 clearance radii (in mm)
            if not (assumed_hole_diam_range[0] / 2.0 <= r <= assumed_hole_diam_range[1] / 2.0):
                continue

            # Center
            center = None
            get_center = getattr(e, "center", None)
            if callable(get_center):
                try:
                    c = get_center()
                    # Vector-like with x, y, z
                    center = (float(c.X), float(c.Y), float(c.Z)) if hasattr(c, "X") else (float(c.x), float(c.y), float(c.z))
                except Exception:
                    center = None
            if center is None:
                # Try approximate center from bounding box midpoint of edge
                try:
                    bb = e.bounding_box()
                    cx = (bb.max.X + bb.min.X) / 2.0
                    cy = (bb.max.Y + bb.min.Y) / 2.0
                    cz = (bb.max.Z + bb.min.Z) / 2.0
                    center = (float(cx), float(cy), float(cz))
                except Exception:
                    continue

            circles.append((r, center))

        if len(circles) < 2:
            return None

        # Decide which axis the two holes are distributed along:
        screw_axis = _auto_screw_axis(align_axis)
        idx = {"x": 0, "y": 1, "z": 2}[screw_axis]

        # Sort by coordinate along the screw axis and take the outermost two
        circles.sort(key=lambda rc: rc[1][idx])
        first = circles[0][1][idx]
        last = circles[-1][1][idx]
        spacing = abs(last - first)

        # Sanity check: typical SMA flange spacing ~ 8–12 mm
        if 5.0 <= spacing <= 20.0:
            return spacing
        return None
    except Exception:
        return None


# Helper to find flange hole centers (as tuples) after SMA is rotated so pin axis == align_axis
def _measure_flange_hole_centers(sma_part, align_axis: AxisName, assumed_hole_diam_range=(1.6, 2.6)):
    """
    Return two flange mounting hole centers (as tuples) after the SMA has already
    been rotated so its pin axis == align_axis. Returns None if not found.
    """
    try:
        circles = []
        for e in sma_part.edges():
            is_circ = getattr(e, "is_circle", None)
            if callable(is_circ):
                is_circ = e.is_circle()
            elif isinstance(is_circ, bool):
                pass
            else:
                continue
            if not is_circ:
                continue
            r = getattr(e, "radius", None)
            if r is None:
                to_circ = getattr(e, "to_circle", None)
                if callable(to_circ):
                    try:
                        r = to_circ().radius
                    except Exception:
                        continue
            if r is None:
                continue
            if not (assumed_hole_diam_range[0] / 2.0 <= r <= assumed_hole_diam_range[1] / 2.0):
                continue
            center = None
            get_center = getattr(e, "center", None)
            if callable(get_center):
                try:
                    c = get_center()
                    center = (float(getattr(c, "X", getattr(c, "x"))),
                              float(getattr(c, "Y", getattr(c, "y"))),
                              float(getattr(c, "Z", getattr(c, "z"))))
                except Exception:
                    center = None
            if center is None:
                try:
                    bb = e.bounding_box()
                    cx = (bb.max.X + bb.min.X) / 2.0
                    cy = (bb.max.Y + bb.min.Y) / 2.0
                    cz = (bb.max.Z + bb.min.Z) / 2.0
                    center = (float(cx), float(cy), float(cz))
                except Exception:
                    continue
            circles.append((r, center))
        if len(circles) < 2:
            return None
        # Choose screw-axis perpendicular to align axis and sort along it
        screw_axis = _auto_screw_axis(align_axis)
        idx = {"x": 0, "y": 1, "z": 2}[screw_axis]
        circles.sort(key=lambda rc: rc[1][idx])
        return circles[0][1], circles[-1][1]
    except Exception:
        return None

def _measure_pin_center(sma_part, align_axis: AxisName, assumed_pin_diam_range=(0.6, 2.0)):
    """
    Find the center of the SMA *pin* by averaging centers of circular edges whose
    radii fall in a plausible SMA pin diameter range (default ~0.6–2.0 mm).
    Assumes the SMA has already been rotated so its pin axis == align_axis.
    Returns (x, y, z) or None if not found.
    """
    try:
        centers = []
        radii = []
        r_min = assumed_pin_diam_range[0] / 2.0
        r_max = assumed_pin_diam_range[1] / 2.0
        for e in sma_part.edges():
            is_circ = getattr(e, "is_circle", None)
            if callable(is_circ):
                ok = e.is_circle()
            elif isinstance(is_circ, bool):
                ok = is_circ
            else:
                continue
            if not ok:
                continue
            # try to get radius
            r = getattr(e, "radius", None)
            if r is None:
                to_circ = getattr(e, "to_circle", None)
                if callable(to_circ):
                    try:
                        r = to_circ().radius
                    except Exception:
                        continue
            if r is None or not (r_min <= r <= r_max):
                continue
            # get center
            c = None
            get_center = getattr(e, "center", None)
            if callable(get_center):
                try:
                    v = get_center()
                    c = (float(getattr(v, "X", getattr(v, "x"))),
                         float(getattr(v, "Y", getattr(v, "y"))),
                         float(getattr(v, "Z", getattr(v, "z"))))
                except Exception:
                    c = None
            if c is None:
                try:
                    bb = e.bounding_box()
                    c = ((bb.max.X + bb.min.X) * 0.5,
                         (bb.max.Y + bb.min.Y) * 0.5,
                         (bb.max.Z + bb.min.Z) * 0.5)
                    c = (float(c[0]), float(c[1]), float(c[2]))
                except Exception:
                    continue
            centers.append(c)
            radii.append(r)
        if not centers:
            return None
        # Prefer the smallest few circles (most likely the pin)
        # Pair (radius, center); sort by radius
        candidates = sorted([(radii[i], centers[i]) for i in range(len(centers))], key=lambda rc: rc[0] if rc[0] is not None else 1e9)
        sel = [rc[1] for rc in candidates[:3]] if len(candidates) >= 3 else [rc[1] for rc in candidates]
        if not sel:
            return None
        sx = sum(c[0] for c in sel)
        sy = sum(c[1] for c in sel)
        sz = sum(c[2] for c in sel)
        n = float(len(sel))
        return (sx / n, sy / n, sz / n)
    except Exception:
        return None


def block_with_cross_holes_and_sma_and_bottom_mount(
    length: float,
    width: float,
    height: float,
    hole_diameter: float,
    sma: SMAMountSpec = SMAMountSpec(),
    bottom: BottomMountSpec = BottomMountSpec(),
):
    r_main = hole_diameter / 2.0
    screw_axis = sma.screw_axis or _auto_screw_axis(sma.align_axis)

    # Import and pre-orient SMA early so we can auto-measure flange spacing if requested
    sma_part = import_step(sma.step_path)

    # Rotate the SMA so its *STEP pin axis* matches the chosen align_axis (pre-rotation for measurement)
    sma_part = _rotate_part_axis(sma_part, sma.pin_axis_in_step, sma.align_axis)
    # Optional extra in-axis clocking if needed
    if abs(sma.extra_clock_deg) > 1e-9:
        sma_part = sma_part.rotate({"x": Axis.X, "y": Axis.Y, "z": Axis.Z}[sma.align_axis], sma.extra_clock_deg)

    # Try to locate the SMA pin center (after axis alignment)
    pin_center = _measure_pin_center(sma_part, sma.align_axis)
    if sma.pin_center_hint is not None:
        pin_center = sma.pin_center_hint

    # Auto-measure or use provided spacing
    spacing = sma.screw_spacing
    if spacing is None:
        spacing = _auto_measure_sma_spacing(sma_part, sma.align_axis)
    if spacing is None:
        spacing = 9.5  # fallback typical SMA flange spacing

    with BuildPart() as bp:
        # Main block
        Box(length, width, height, align=(Align.CENTER, Align.CENTER, Align.CENTER))

        # Cross holes
        Cylinder(radius=r_main, height=length + 1.0, rotation=_axis_to_euler("x"), mode=Mode.SUBTRACT)
        Cylinder(radius=r_main, height=width + 1.0,  rotation=_axis_to_euler("y"), mode=Mode.SUBTRACT)

        # SMA flange holes (blind holes from mating face)
        # Base position for blind SMA screw holes: start at the mating face center and move inward by depth/2
        face_center_sma, face_normal_sma = _face_center_and_normal(length, width, height, sma.mate_face)
        axis_idx = {"x": 0, "y": 1, "z": 2}[sma.align_axis]
        base_center = [*face_center_sma]
        base_center[axis_idx] = base_center[axis_idx] - face_normal_sma[axis_idx] * (sma.screw_hole_depth / 2.0)

        r_sma = sma.screw_clearance_diam / 2.0
        # spacing computed above
        if screw_axis == "y":
            offsets = [(0, +spacing/2, 0), (0, -spacing/2, 0)]
        elif screw_axis == "x":
            offsets = [(+spacing/2, 0, 0), (-spacing/2, 0, 0)]
        else:  # "z"
            offsets = [(0, 0, +spacing/2), (0, 0, -spacing/2)]

        for p in offsets:
            cx, cy, cz = base_center
            if screw_axis == "y":
                cy += p[1]
            elif screw_axis == "x":
                cx += p[0]
            else:  # "z"
                cz += p[2]
            with Locations((cx, cy, cz)):
                Cylinder(
                    radius=r_sma,
                    height=sma.screw_hole_depth,
                    rotation=_axis_to_euler(sma.align_axis),
                    mode=Mode.SUBTRACT,
                )

        # Bottom four 2-56 mounting holes (blind along Z from the chosen bottom face)
        if bottom.enabled:
            r_btm = bottom.clearance_diam / 2.0
            face_center_btm, face_normal_btm = _face_center_and_normal(length, width, height, bottom.face)
            # Center along Z is the bottom face shifted inward by depth/2
            z_center = face_center_btm[2] - face_normal_btm[2] * (bottom.hole_depth / 2.0)
            x_off = length / 2 - bottom.edge_margin_x
            y_off = width  / 2 - bottom.edge_margin_y
            for sx in (-1, +1):
                for sy in (-1, +1):
                    with Locations((sx * x_off, sy * y_off, z_center)):
                        Cylinder(
                            radius=r_btm,
                            height=bottom.hole_depth,
                            mode=Mode.SUBTRACT,  # default axis = Z; no rotation needed
                        )

    block_part = bp.part

    # Auto-clock the SMA so the flange hole pair lines up with our chosen screw axis
    hole_centers = _measure_flange_hole_centers(sma_part, sma.align_axis)
    if hole_centers:
        (c1x, c1y, c1z), (c2x, c2y, c2z) = hole_centers
        # Vector between holes projected into the face plane
        if sma.align_axis == "x":
            vy, vz = (c2y - c1y), (c2z - c1z)
            # Desired along +Y (our default screw-axis for X-aligned SMA)
            ang = atan2(vz, vy)  # angle from +Y toward +Z
            if abs(ang) > 1e-6:
                sma_part = sma_part.rotate(Axis.X, -degrees(ang))
        elif sma.align_axis == "y":
            vx, vz = (c2x - c1x), (c2z - c1z)
            ang = atan2(vz, vx)  # angle from +X toward +Z
            if abs(ang) > 1e-6:
                sma_part = sma_part.rotate(Axis.Y, -degrees(ang))
        else:  # "z"
            vx, vy = (c2x - c1x), (c2y - c1y)
            ang = atan2(vy, vx)  # angle from +X toward +Y
            if abs(ang) > 1e-6:
                sma_part = sma_part.rotate(Axis.Z, -degrees(ang))
        # Recompute centers after clocking
        hole_centers = _measure_flange_hole_centers(sma_part, sma.align_axis)

    # Translate SMA so its flange holes' midpoint sits at the block hole centerline,
    # and put the flange plane on the chosen mating face with mate_offset
    if hole_centers:
        (c1x, c1y, c1z), (c2x, c2y, c2z) = hole_centers
        midx, midy, midz = (0.5*(c1x+c2x), 0.5*(c1y+c2y), 0.5*(c1z+c2z))
        # Target mid-in-plane is origin on the two axes perpendicular to align_axis
        if sma.align_axis == "x":
            # Prefer pin center alignment in YZ; fallback to holes’ midpoint if pin not found
            if pin_center:
                _, p_y, p_z = pin_center
                dy, dz = -p_y, -p_z
            else:
                dy, dz = -midy, -midz
            # Flange plane coordinate along X ~ average of hole centers' X
            plane = midx
            face_center, face_normal = _face_center_and_normal(length, width, height, sma.mate_face)
            target_plane = face_center[0]
            dx = (target_plane - plane) + (face_normal[0] * sma.mate_offset)
            sma_part = sma_part.moved(Location((dx, dy, dz)))
        elif sma.align_axis == "y":
            if pin_center:
                p_x, _, p_z = pin_center
                dx, dz = -p_x, -p_z
            else:
                dx, dz = -midx, -midz
            plane = midy
            face_center, face_normal = _face_center_and_normal(length, width, height, sma.mate_face)
            target_plane = face_center[1]
            dy = (target_plane - plane) + (face_normal[1] * sma.mate_offset)
            sma_part = sma_part.moved(Location((dx, dy, dz)))
        else:  # "z"
            if pin_center:
                p_x, p_y, _ = pin_center
                dx, dy = -p_x, -p_y
            else:
                dx, dy = -midx, -midy
            plane = midz
            face_center, face_normal = _face_center_and_normal(length, width, height, sma.mate_face)
            target_plane = face_center[2]
            dz = (target_plane - plane) + (face_normal[2] * sma.mate_offset)
            sma_part = sma_part.moved(Location((dx, dy, dz)))

    # If we couldn't measure/align via flange holes, place on the face and align pin center to bore axis if possible
    else:
        face_center, face_normal = _face_center_and_normal(length, width, height, sma.mate_face)
        tx, ty, tz = face_center
        nx, ny, nz = face_normal
        if pin_center:
            p_x, p_y, p_z = pin_center
            if sma.align_axis == "x":
                dx = (tx - p_x) + nx * sma.mate_offset
                dy = -p_y
                dz = -p_z
            elif sma.align_axis == "y":
                dx = -p_x
                dy = (ty - p_y) + ny * sma.mate_offset
                dz = -p_z
            else:  # "z"
                dx = -p_x
                dy = -p_y
                dz = (tz - p_z) + nz * sma.mate_offset
            sma_part = sma_part.moved(Location((dx, dy, dz)))
        else:
            sma_part = sma_part.moved(Location((tx + nx * sma.mate_offset,
                                                ty + ny * sma.mate_offset,
                                                tz + nz * sma.mate_offset)))

    return block_part, sma_part

# --- Example ---
if __name__ == "__main__":
    block, sma_body = block_with_cross_holes_and_sma_and_bottom_mount(
        length=40.0, width=25.0, height=12.0, hole_diameter=2.5,
        sma=SMAMountSpec(
            step_path="142-1701-201.step",
            align_axis="y", mate_face="+y",
            mate_offset=-4.445,
            screw_hole_depth=5.0,
            screw_clearance_diam=2.18, screw_spacing=12.2174,
            extra_clock_deg=90.0  # rotate about the pin axis by 90°
        ),
        bottom=BottomMountSpec(
            enabled=True,
            clearance_diam=2.18,
            edge_margin_x=4.0,
            edge_margin_y=4.0,
            face="-z",
            hole_depth=5.0,
        )
    )

    from build123d import export_step, Compound

    box_assembly = Compound(label="assembly", children=[block, sma_body])
    export_step(box_assembly, "cavity_with_sma.step")