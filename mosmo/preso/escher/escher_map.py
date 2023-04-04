"""Render an Escher map as SVG."""
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

ESCHER_CSS = """
    #eschermap .label {
      font-family: sans-serif;
      font-style: italic;
      font-weight: bold;
      stroke: none;
      text-rendering: optimizelegibility;
    }
    #eschermap #reactions .label {
      font-size: 30px;
      fill: #334E75;
    }
    #eschermap #metabolites .label {
      font-size: 20px;
      fill: black;
    }
    #eschermap #reactions .stoich {
      font-size: 8px;
      fill: #334E75;
    }
    #eschermap #reactions path {
      fill: none;
      stroke: #334E75;
      stroke-width: 10;
    }
    #eschermap #arrowhead path {
      fill: #334E75;
      stroke: none;
    }
    #eschermap #metabolites circle {
      fill: rgb(224, 134, 91);
      stroke: rgb(162, 69, 16);
      stroke-width: 2;
    }
"""


@dataclass(repr=False)
class _Element:
    """Lightweight generalized SVG document element."""
    name: str
    attrs: Mapping[str, Any] = field(default_factory=dict)
    content: str = None
    children: Sequence["_Element"] = field(default_factory=list)

    def render(self, indent: Optional[str] = None, level: int = 0):
        if indent is not None:
            linesep = "\n"
            prefix = indent * level
        else:
            linesep = ""
            prefix = ""

        attrs = [""]
        if self.attrs:
            for k, v in self.attrs.items():
                if v is not None:
                    attrs.append(f"{k}='{v}'")
        attrs_str = ' '.join(attrs)

        lines = []
        if self.content:
            lines.append(prefix + self.content)
        if self.children:
            lines.extend(child.render(indent=indent, level=level + 1) for child in self.children)
        content_str = linesep.join(lines)

        if content_str:
            return f"{prefix}<{self.name}{attrs_str}>{linesep}{content_str}{linesep}{prefix}</{self.name}>"
        else:
            return f"{prefix}<{self.name}{attrs_str}/>"

    def __str__(self):
        """In an interactive context, the element renders itself as a hierarchically indented structure."""
        return self.render(indent="  ")

    def __repr__(self):
        return str(self)


class MapNode:
    """Any node that can serve as an endpoint for a segment."""

    def __init__(self, node_json):
        self.center = (node_json["x"], node_json["y"])


class MapMetabolite(MapNode):
    """A node that designates a metabolite."""

    def __init__(self, node_json):
        super().__init__(node_json)
        self.metabolite_id = node_json["bigg_id"]
        self.primary = node_json["node_is_primary"]
        self.label_pos = (node_json["label_x"], node_json["label_y"])

    def size(self) -> float:
        if self.primary:
            return 20.
        else:
            return 12.

    def to_svg(self) -> _Element:
        return _Element("g", {"name": self.metabolite_id}, children=[
            _Element("circle", {"cx": self.center[0], "cy": self.center[1], "r": self.size()}),
            _Element("text",
                     {"x": self.label_pos[0], "y": self.label_pos[1], "class": "label"},
                     content=self.metabolite_id)
        ])


class MapSegment:
    """A single connection tying a node to a reaction."""

    def __init__(self, segment_json, reaction: "MapReaction", all_nodes: Mapping[str, MapNode]):
        self.from_node = all_nodes[segment_json["from_node_id"]]
        self.to_node = all_nodes[segment_json["to_node_id"]]
        self.b1 = (segment_json["b1"]["x"], segment_json["b1"]["y"]) if segment_json["b1"] else None
        self.b2 = (segment_json["b2"]["x"], segment_json["b2"]["y"]) if segment_json["b2"] else None
        if isinstance(self.to_node, MapMetabolite):
            self.has_arrow = reaction.reversible or reaction.stoich[self.to_node.metabolite_id] > 0

    def to_svg(self, padding=20., min_len=10.) -> _Element:
        start = self.from_node.center
        end = self.to_node.center
        if isinstance(self.to_node, MapMetabolite):
            # Adjust the endpoint to approach the metabolite node but stop at a padded distance from it.
            dx = end[0] - self.b2[0]
            dy = end[1] - self.b2[1]
            l = math.sqrt(dx * dx + dy * dy)
            ratio = min(l, max(min_len, l - self.to_node.size() - padding)) / l
            end = (self.b2[0] + dx * ratio, self.b2[1] + dy * ratio)
            attrs = {"d": f"M {start[0]:.1f} {start[1]:.1f} C {self.b1[0]:.1f} {self.b1[1]:.1f} {self.b2[0]:.1f}" +
                          f" {self.b2[1]:.1f} {end[0]:.1f} {end[1]:.1f}"}

            if self.has_arrow:
                attrs["marker-end"] = "url(#arrowhead)"
            return _Element("path", attrs)
        else:
            # Just a connector between "midmarker" and "multimarker". Uae a <path> so css finds it.
            return _Element("path", {"d": f"M {start[0]:.1f} {start[1]:.1f} L {end[0]:.1f} {end[1]:.1f}"})


class MapReaction:
    """A collection of segments associating metabolites with a reaction."""

    def __init__(self, reaction_json, all_nodes: Mapping[str, MapNode]):
        self.reaction_id = reaction_json["bigg_id"]
        self.stoich = {m["bigg_id"]: m["coefficient"] for m in reaction_json["metabolites"]}
        self.reversible = reaction_json["reversibility"]
        self.label_pos = (reaction_json["label_x"], reaction_json["label_y"])
        self.segments = []
        for segment_json in reaction_json["segments"].values():
            self.segments.append(MapSegment(segment_json, self, all_nodes))

    def to_svg(self) -> _Element:
        children = [segment.to_svg() for segment in self.segments]
        children.append(_Element("text", {"x": self.label_pos[0], "y": self.label_pos[1], "class": "label"},
                                 content=self.reaction_id))
        return _Element("g", {"name": self.reaction_id}, children=children)


class EscherMap:
    def __init__(self, map_json):
        canvas = map_json[1]["canvas"]
        self.origin = (canvas["x"], canvas["y"])
        self.size = (canvas["width"], canvas["height"])

        # All nodes may be referred to by some reaction segment. Only metabolite nodes will be rendered.
        all_nodes = {}
        self.metabolites = []
        for node_id, node in map_json[1]["nodes"].items():
            if node["node_type"] == "metabolite":
                metabolite = MapMetabolite(node)
                all_nodes[node_id] = metabolite
                self.metabolites.append(metabolite)
            else:
                all_nodes[node_id] = MapNode(node)

        self.reactions = []
        for reaction_id, reaction_json in map_json[1]["reactions"].items():
            self.reactions.append(MapReaction(reaction_json, all_nodes))

    def to_svg(self, width="20cm", height=None):
        defs_element = _Element(
            "defs",
            children=[
                _Element(
                    "marker",
                    {"id": "arrowhead", "viewBox": "0 -10 13 20", "refX": "2", "refY": "0",
                     "markerUnits": "strokeWidth", "markerWidth": "2", "markerHeight": "2", "orient": "auto"},
                    children=[
                        _Element("path", {"d": "M 0 -10 L 13 0 L 0 10 Z", "fill": "#334E75", "stroke": "none"})
                    ]),
                _Element("style", {"type": "text/css"}, content=ESCHER_CSS),
            ])

        map_element = _Element(
            "g",
            {"id": "eschermap"},
            children=[
                _Element("rect", {"id": "canvas", "x": self.origin[0], "y": self.origin[1], "width": self.size[0],
                                  "height": self.size[1], "fill": "white", "stroke": "#ccc"}),
                _Element("g", {"id": "reactions"}, children=[reaction.to_svg() for reaction in self.reactions]),
                _Element("g", {"id": "metabolites"}, children=[metabolite.to_svg() for metabolite in self.metabolites]),
            ])

        return _Element(
            "svg",
            {"width": width, "height": height,
             "viewBox": f"{self.origin[0]} {self.origin[1]} {self.size[0]} {self.size[1]}"},
            children=[defs_element, map_element])
