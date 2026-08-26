"""Object-bound declarative layout library for the documentation SVG figures.

Design rules (enforced, not aspirational):

* Everything visible is an `Obj` with a semantic name and a `bounds`
  rectangle computed by the layout engine.  Nothing is drawn at a raw
  pixel coordinate.
* Positions are derived from object relations only: grid cells are placed
  by (row, col, cell size) from the grid's origin; every other object is
  placed by rules such as `right_of`, `below`, `align_left`,
  `center_x_on`.  The single exception is `at(x, y)`, which may only be
  used for the root anchor(s) of a scene (e.g. a grid origin).
* Connectors are declared as `connect(a, port_a, b, port_b)`; endpoint
  coordinates are computed from the objects' bounds.  Optional waypoints
  are callables evaluated after layout, so they too are computed from
  object bounds (e.g. "a lane 16px below the payload row").
* `enclose(children, padding)` produces a container whose bounds are the
  union of its children's bounds plus padding — it can never fail to
  contain its children.
* All text is measured with a monospace model (ASCII = 1 unit,
  CJK/full-width = 2 units, times a char-width factor); box sizes are
  derived from the measurement, never hand-written.

The validator runs automatically on every render and raises on:

  (a) a connector segment crossing the interior of any object that is not
      one of the connector's two endpoints;
  (b) an enclose container not strictly containing every child;
  (c) two objects in the same overlap group overlapping with positive
      area (grid cells, boxes, labels).
"""

import html
import math
import unicodedata

# --------------------------------------------------------------------------
# Palette: fixed color semantics (declared in images/fig00-legend.svg)
# --------------------------------------------------------------------------
PALETTE = {
    "blue":  {"stroke": "#2563eb", "fill": "#eff6ff"},   # forward / baseline data
    "red":   {"stroke": "#dc2626", "fill": "#fef2f2"},   # credit flow / changed object
    "gray":  {"stroke": "#6b7280", "fill": "#f3f4f6"},   # frozen content
    "green": {"stroke": "#16a34a", "fill": "#f0fdf4"},   # match / hit
    "ink":   {"stroke": "#111827", "fill": "#ffffff"},
}
LINE = "#374151"          # neutral thin lines
TEXT = "#111827"
STROKE_W = 1.5
FONT_STACK = "Consolas,Menlo,monospace"
CHAR_W_FACTOR = 0.62      # monospace advance / font-size
LINE_H_FACTOR = 1.42


def _units(ch):
    return 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1


def text_width(text, font_size):
    """Measured width: ASCII chars count 1 unit, CJK/full-width count 2."""
    return sum(_units(c) for c in text) * font_size * CHAR_W_FACTOR


def line_height(font_size):
    return font_size * LINE_H_FACTOR


class LayoutError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------
class Bounds:
    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    def union(self, other):
        return Bounds(min(self.x0, other.x0), min(self.y0, other.y0),
                      max(self.x1, other.x1), max(self.y1, other.y1))

    def padded(self, p):
        return Bounds(self.x0 - p, self.y0 - p, self.x1 + p, self.y1 + p)

    def overlaps(self, other, eps=0.5):
        return (min(self.x1, other.x1) - max(self.x0, other.x0) > eps and
                min(self.y1, other.y1) - max(self.y0, other.y0) > eps)

    def __repr__(self):
        return f"Bounds({self.x0:.1f},{self.y0:.1f},{self.x1:.1f},{self.y1:.1f})"


PORTS = {
    "left":   lambda b: (b.x0, b.cy),
    "right":  lambda b: (b.x1, b.cy),
    "top":    lambda b: (b.cx, b.y0),
    "bottom": lambda b: (b.cx, b.y1),
    "center": lambda b: (b.cx, b.cy),
    "top-left":     lambda b: (b.x0, b.y0),
    "top-right":    lambda b: (b.x1, b.y0),
    "bottom-left":  lambda b: (b.x0, b.y1),
    "bottom-right": lambda b: (b.x1, b.y1),
}
VERTICAL_PORTS = {"top", "bottom"}


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------
class Obj:
    """A named layout object.  Position comes from relation rules."""

    def __init__(self, scene, name, w, h, group=None):
        self.scene = scene
        self.name = name
        self.w, self.h = w, h
        self.group = group
        self._x_rule = None
        self._y_rule = None
        self._x = None
        self._y = None
        self._resolving = False

    # -- placement rules ------------------------------------------------
    # every relation accepts a single object or a list (union of bounds)
    @staticmethod
    def _B(other):
        if isinstance(other, (list, tuple)):
            return _union([o.bounds for o in other])
        return other.bounds

    def at(self, x, y):
        """Root anchor — the only place absolute coordinates are allowed."""
        self._x_rule = lambda: x
        self._y_rule = lambda: y
        return self

    def right_of(self, other, gap):
        self._x_rule = lambda: self._B(other).x1 + gap
        return self

    def left_of(self, other, gap):
        self._x_rule = lambda: self._B(other).x0 - gap - self.w
        return self

    def below(self, other, gap):
        self._y_rule = lambda: self._B(other).y1 + gap
        return self

    def above(self, other, gap):
        self._y_rule = lambda: self._B(other).y0 - gap - self.h
        return self

    def align_left(self, other):
        self._x_rule = lambda: self._B(other).x0
        return self

    def align_right(self, other):
        self._x_rule = lambda: self._B(other).x1 - self.w
        return self

    def align_top(self, other):
        self._y_rule = lambda: self._B(other).y0
        return self

    def align_bottom(self, other):
        self._y_rule = lambda: self._B(other).y1 - self.h
        return self

    def center_x_on(self, other):
        self._x_rule = lambda: self._B(other).cx - self.w / 2
        return self

    def center_y_on(self, other):
        self._y_rule = lambda: self._B(other).cy - self.h / 2
        return self

    # -- resolution -----------------------------------------------------
    def _resolve(self):
        if self._x is not None:
            return
        if self._resolving:
            raise LayoutError(f"cyclic placement rule at {self.name!r}")
        if self._x_rule is None or self._y_rule is None:
            raise LayoutError(f"object {self.name!r} has no placement rule")
        self._resolving = True
        self._x = float(self._x_rule())
        self._y = float(self._y_rule())
        self._resolving = False

    @property
    def bounds(self):
        self._resolve()
        return Bounds(self._x, self._y, self._x + self.w, self._y + self.h)

    def port(self, name):
        return PORTS[name](self.bounds)

    # -- rendering ------------------------------------------------------
    def svg(self):
        raise NotImplementedError


def _union(bounds_list):
    u = bounds_list[0]
    for b in bounds_list[1:]:
        u = u.union(b)
    return u


def _fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


class Label(Obj):
    """A free-standing text label; size is measured, never hand-set."""

    def __init__(self, scene, name, text, font_size=13, color=TEXT,
                 bold=False, anchor="middle"):
        super().__init__(scene, name, text_width(text, font_size),
                         line_height(font_size), group="label")
        self.text = text
        self.font_size = font_size
        self.color = color
        self.bold = bold
        self.anchor = anchor  # svg text-anchor for the label's own box

    def svg(self, dx, dy):
        b = self.bounds
        x = {"start": b.x0, "middle": b.cx, "end": b.x1}[self.anchor]
        y = b.y0 + self.font_size  # baseline
        weight = ' font-weight="bold"' if self.bold else ""
        return (f'<text x="{_fmt(x + dx)}" y="{_fmt(y + dy)}" '
                f'font-family="{FONT_STACK}" font-size="{self.font_size}" '
                f'fill="{self.color}" text-anchor="{self.anchor}"{weight}>'
                f'{html.escape(self.text)}</text>')


class Box(Obj):
    """Rounded box whose size is derived from measured text lines."""

    def __init__(self, scene, name, lines, style="blue", font_size=13,
                 pad_x=10, pad_y=7, rx=4, dashed=False, text_color=None,
                 bold_first=False):
        if isinstance(lines, str):
            lines = [lines]
        w = max(text_width(t, font_size) for t in lines) + 2 * pad_x
        h = len(lines) * line_height(font_size) + 2 * pad_y - \
            (line_height(font_size) - font_size)
        super().__init__(scene, name, w, h, group="box")
        self.lines = lines
        self.style = style
        self.font_size = font_size
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.rx = rx
        self.dashed = dashed
        self.text_color = text_color
        self.bold_first = bold_first

    def svg(self, dx, dy):
        b = self.bounds
        pal = PALETTE[self.style]
        dash = ' stroke-dasharray="5,3"' if self.dashed else ""
        out = [f'<rect x="{_fmt(b.x0 + dx)}" y="{_fmt(b.y0 + dy)}" '
               f'width="{_fmt(b.w)}" height="{_fmt(b.h)}" rx="{self.rx}" '
               f'fill="{pal["fill"]}" stroke="{pal["stroke"]}" '
               f'stroke-width="{STROKE_W}"{dash}/>']
        color = self.text_color or TEXT
        lh = line_height(self.font_size)
        for i, t in enumerate(self.lines):
            y = b.y0 + self.pad_y + i * lh + self.font_size * 0.82
            weight = ' font-weight="bold"' if (self.bold_first and i == 0) else ""
            out.append(f'<text x="{_fmt(b.cx + dx)}" y="{_fmt(y + dy)}" '
                       f'font-family="{FONT_STACK}" font-size="{self.font_size}" '
                       f'fill="{color}" text-anchor="middle"{weight}>'
                       f'{html.escape(t)}</text>')
        return "\n".join(out)


class Cell(Obj):
    """One cell of a Grid.  Position is (row, col, cell size) from the
    grid origin — never an absolute coordinate."""

    def __init__(self, grid, row, col, text="", style="cell"):
        self.grid = grid
        self.row, self.col = row, col
        super().__init__(grid.scene, f"{grid.name}.cell[{row},{col}]",
                         grid.cw, grid.ch, group=f"grid:{grid.name}")
        self.text = text
        self.style = style

    def _resolve(self):
        if self._x is not None:
            return
        g = self.grid.bounds
        self._x = g.x0 + self.col * self.grid.cw
        self._y = g.y0 + self.row * self.grid.ch

    def svg(self, dx, dy):
        b = self.bounds
        if self.style == "cell":
            fill, stroke = "#ffffff", LINE
            tcol = TEXT
        elif self.style == "plain":
            fill, stroke = "none", "none"
            tcol = TEXT
        else:
            pal = PALETTE[self.style]
            fill, stroke = pal["fill"], pal["stroke"]
            tcol = TEXT
        out = []
        if fill != "none":
            out.append(f'<rect x="{_fmt(b.x0 + dx)}" y="{_fmt(b.y0 + dy)}" '
                       f'width="{_fmt(b.w)}" height="{_fmt(b.h)}" '
                       f'fill="{fill}" stroke="{stroke}" stroke-width="{STROKE_W}"/>')
        if self.text:
            y = b.y0 + self.grid.ch / 2 + self.grid.font_size * 0.35
            out.append(f'<text x="{_fmt(b.cx + dx)}" y="{_fmt(y + dy)}" '
                       f'font-family="{FONT_STACK}" font-size="{self.grid.font_size}" '
                       f'fill="{tcol}" text-anchor="middle">{html.escape(self.text)}</text>')
        return "\n".join(out)


class Grid(Obj):
    def __init__(self, scene, name, rows, cols, cw=34, ch=30, font_size=13):
        super().__init__(scene, name, cols * cw, rows * ch, group=None)
        self.rows, self.cols = rows, cols
        self.cw, self.ch = cw, ch
        self.font_size = font_size
        self._cells = {}
        scene._register(self)

    def cell(self, r, c, text=None, style=None):
        key = (r, c)
        if key not in self._cells:
            self._cells[key] = Cell(self, r, c)
            self.scene._register(self._cells[key])
        cell = self._cells[key]
        if text is not None:
            cell.text = text
        if style is not None:
            cell.style = style
        return cell

    def row_cells(self, r, c0=None, c1=None):
        c0 = 0 if c0 is None else c0
        c1 = self.cols - 1 if c1 is None else c1
        return [self.cell(r, c) for c in range(c0, c1 + 1)]

    def svg(self, dx, dy):
        return ""  # cells render themselves


class Container(Obj):
    """enclose(children, padding): bounds = union(children) + padding."""

    def __init__(self, scene, name, children, padding, style="blue",
                 dashed=False, rx=6):
        self.children = list(children)
        self.padding = padding
        super().__init__(scene, name, 0, 0, group=None)
        self.style = style
        self.dashed = dashed
        self.rx = rx

    def _resolve(self):
        if self._x is not None:
            return
        u = _union([c.bounds for c in self.children]).padded(self.padding)
        self._x, self._y = u.x0, u.y0
        self.w, self.h = u.w, u.h

    def svg(self, dx, dy):
        b = self.bounds
        pal = PALETTE[self.style]
        dash = ' stroke-dasharray="6,3"' if self.dashed else ""
        return (f'<rect x="{_fmt(b.x0 + dx)}" y="{_fmt(b.y0 + dy)}" '
                f'width="{_fmt(b.w)}" height="{_fmt(b.h)}" rx="{self.rx}" '
                f'fill="none" stroke="{pal["stroke"]}" stroke-width="{STROKE_W}"'
                f'{dash}/>')


class Connector:
    """connect(a, port_a, b, port_b) — endpoints computed from bounds.

    `waypoints` is an optional callable evaluated after layout returning
    intermediate polyline points; it must derive its coordinates from
    object bounds.  `label` places a measured label next to the middle
    segment (`label_side`: 'above'/'below' for horizontal middles,
    'left'/'right' for vertical middles).
    """

    def __init__(self, scene, a, port_a, b, port_b, color=LINE, arrow=True,
                 dashed=False, waypoints=None, label=None, label_side="above",
                 label_font=12):
        self.scene = scene
        self.a, self.pa = a, port_a
        self.b, self.pb = b, port_b
        self.color = color
        self.arrow = arrow
        self.dashed = dashed
        self.waypoints = waypoints
        self.name = f"conn:{a.name}->{b.name}"
        self.label_obj = None
        if label is not None:
            self.label_obj = Label(scene, self.name + ".label", label,
                                   font_size=label_font, color=color)
            scene._register(self.label_obj)
            self._place_label(label_side)

    def _place_label(self, side):
        lab = self.label_obj
        seg = self._middle_segment
        (x0, y0), (x1, y1) = seg
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        gap = 5
        if abs(y0 - y1) < 0.01:  # horizontal middle segment
            if side in ("above", "below"):
                lab._x_rule = lambda: mx - lab.w / 2
                if side == "above":
                    lab._y_rule = lambda: my - gap - lab.h
                else:
                    lab._y_rule = lambda: my + gap
            else:
                raise LayoutError(f"bad label_side {side!r} for horizontal segment")
        else:  # vertical middle segment
            lab._y_rule = lambda: my - lab.h / 2
            if side == "left":
                lab._x_rule = lambda: mx - gap - lab.w
            else:
                lab._x_rule = lambda: mx + gap

    @property
    def points(self):
        pts = [self.a.port(self.pa)]
        if self.waypoints is not None:
            pts.extend(self.waypoints())
        pts.append(self.b.port(self.pb))
        # drop zero-length duplicates
        out = [pts[0]]
        for p in pts[1:]:
            if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
                out.append(p)
        return out

    @property
    def segments(self):
        pts = self.points
        return list(zip(pts, pts[1:]))

    @property
    def _middle_segment(self):
        return self.segments[len(self.segments) // 2]

    def svg(self, dx, dy):
        pts = [(x + dx, y + dy) for x, y in self.points]
        d = "M " + " L ".join(f"{_fmt(x)} {_fmt(y)}" for x, y in pts)
        dash = ' stroke-dasharray="5,3"' if self.dashed else ""
        marker = ""
        if self.arrow:
            marker = f' marker-end="url(#arr-{self.color.strip("#")})"'
        return (f'<path d="{d}" fill="none" stroke="{self.color}" '
                f'stroke-width="{STROKE_W}"{dash}{marker}/>')


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
class Scene:
    def __init__(self, name, title="", desc=""):
        self.name = name
        self.title = title
        self.desc = desc or title
        self.objects = []
        self.connectors = []
        self.margin = 14

    def _register(self, obj):
        self.objects.append(obj)

    # -- factories ------------------------------------------------------
    def grid(self, name, rows, cols, cw=34, ch=30, font_size=13):
        return Grid(self, name, rows, cols, cw, ch, font_size)

    def label(self, name, text, **kw):
        lab = Label(self, name, text, **kw)
        self._register(lab)
        return lab

    def box(self, name, lines, **kw):
        b = Box(self, name, lines, **kw)
        self._register(b)
        return b

    def enclose(self, name, children, padding=8, **kw):
        c = Container(self, name, children, padding, **kw)
        self._register(c)
        return c

    def connect(self, a, port_a, b, port_b, **kw):
        conn = Connector(self, a, port_a, b, port_b, **kw)
        self.connectors.append(conn)
        return conn

    # -- validation -----------------------------------------------------
    def validate(self):
        errors = []

        # force resolution of everything
        for o in self.objects:
            o.bounds
        for c in self.connectors:
            c.segments

        # (a) connector segments must not cross non-endpoint object interiors
        for conn in self.connectors:
            endpoints = {id(conn.a), id(conn.b)}
            for seg in conn.segments:
                for obj in self.objects:
                    if id(obj) in endpoints or isinstance(obj, Grid):
                        continue
                    if _seg_hits_rect(seg[0], seg[1], obj.bounds):
                        errors.append(
                            f"(a) connector {conn.name} crosses object "
                            f"{obj.name!r} interior")

        # (b) enclose containers strictly contain their children
        for obj in self.objects:
            if isinstance(obj, Container):
                cb = obj.bounds
                for child in obj.children:
                    b = child.bounds
                    if not (cb.x0 <= b.x0 - obj.padding + 0.01 and
                            cb.y0 <= b.y0 - obj.padding + 0.01 and
                            cb.x1 >= b.x1 + obj.padding - 0.01 and
                            cb.y1 >= b.y1 + obj.padding - 0.01):
                        errors.append(
                            f"(b) container {obj.name!r} does not strictly "
                            f"contain {child.name!r} with padding {obj.padding}")

        # (c) same-group objects must not overlap
        grouped = [o for o in self.objects if o.group]
        for i in range(len(grouped)):
            for j in range(i + 1, len(grouped)):
                a, b = grouped[i], grouped[j]
                if a.bounds.overlaps(b.bounds):
                    errors.append(
                        f"(c) objects {a.name!r} and {b.name!r} overlap")

        if errors:
            raise LayoutError("scene validation failed:\n  " +
                              "\n  ".join(errors))

    # -- rendering ------------------------------------------------------
    def render(self):
        self.validate()
        u = None
        for o in self.objects:
            u = o.bounds if u is None else u.union(o.bounds)
        for c in self.connectors:
            for p in c.points:
                u = u.union(Bounds(p[0], p[1], p[0], p[1]))
        dx, dy = self.margin - u.x0, self.margin - u.y0
        w, h = u.w + 2 * self.margin, u.h + 2 * self.margin

        containers = [o for o in self.objects if isinstance(o, Container)]
        cells = [o for o in self.objects if isinstance(o, Cell)]
        boxes = [o for o in self.objects if isinstance(o, Box)]
        labels = [o for o in self.objects if isinstance(o, Label)]

        marker_defs = []
        for key in set([c.color for c in self.connectors if c.arrow]) | {LINE}:
            mid = key.strip("#")
            marker_defs.append(
                f'<marker id="arr-{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{key}"/></marker>')

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(w)}" '
            f'height="{_fmt(h)}" viewBox="0 0 {_fmt(w)} {_fmt(h)}" '
            f'role="img" aria-label="{html.escape(self.title)}">',
            f'<title>{html.escape(self.title)}</title>',
            f'<desc>{html.escape(self.desc)}</desc>',
            f'<defs>{"".join(marker_defs)}</defs>',
            f'<rect x="0" y="0" width="{_fmt(w)}" height="{_fmt(h)}" fill="#ffffff"/>',
        ]
        for o in containers:
            parts.append(o.svg(dx, dy))
        for c in self.connectors:
            parts.append(c.svg(dx, dy))
        for o in cells + boxes + labels:
            parts.append(o.svg(dx, dy))
        parts.append("</svg>")
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _seg_hits_rect(p0, p1, rect, eps=0.6):
    """True iff segment p0-p1 has positive-length intersection with the
    interior of rect (rect shrunk by eps).  Liang–Barsky clip."""
    x0, y0 = p0
    x1, y1 = p1
    r = Bounds(rect.x0 + eps, rect.y0 + eps, rect.x1 - eps, rect.y1 - eps)
    if r.x0 >= r.x1 or r.y0 >= r.y1:
        return False
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r.x0), (dx, r.x1 - x0),
                 (-dy, y0 - r.y0), (dy, r.y1 - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return t1 - t0 > 1e-9
