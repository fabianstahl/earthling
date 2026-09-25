"""Texture layers: generic, declarative surface colourings for the terrain.

A layer provides a GLSL function ``vec3 layer_<id>(LayerInput li)`` returning an sRGB albedo and a
list of properties (its own UI section). Two slots (A and B) choose layers and are crossfaded:

    color = mix(shade(layer_A), shade(layer_B), layers.mix)

``shade`` blends between lit (scene lighting) and flat (unlit) presentation with the layer's own
"shading" property. The dispatcher GLSL is generated from the registry and injected into the
terrain shader as the virtual include ``layers_generated.glsl``.

Adding a layer = one :class:`Layer` entry here (plus a tile source for imagery-type layers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from earthling.core.properties import PropertyDef, PType
from earthling.render.parameters import boolean, enum, flt, section

RAMPS = [("hypsometric", "Hypsometric"), ("viridis", "Viridis"), ("turbo", "Turbo"),
         ("grayscale", "Grayscale")]  # fmt: skip


@dataclass(frozen=True)
class Layer:
    id: str
    label: str
    glsl: str  # body of `vec3 layer_<id>(LayerInput li)`
    properties: list[PropertyDef] = field(default_factory=list)
    default_shading: float = 1.0
    tile_source: str | None = None  # "imagery" for layers sampling the node's imagery texture

    @property
    def shading_uniform(self) -> str:
        return f"u_{self.id}_shading"

    @property
    def gain_uniform(self) -> str:
        return f"u_{self.id}_gain"


def _layer(id_, label, glsl, *props, shading=1.0, gain=0.45, tile_source=None) -> Layer:
    shading_def = flt(f"layer_{id_}.shading", "Lighting", shading, 0.0, 1.0,
                      uniform=f"u_{id_}_shading",
                      tooltip="1 = lit by sun and sky, 0 = flat colours")  # fmt: skip
    gain_def = flt(f"layer_{id_}.gain", "Brightness", gain, 0.0, 2.0, uniform=f"u_{id_}_gain",
                   tooltip="Linear albedo multiplier (maps are brighter than ground)")  # fmt: skip
    defs = section(f"Layer: {label}", shading_def, gain_def, *props)
    return Layer(id_, label, glsl, defs, shading, tile_source)


LAYERS: list[Layer] = [
    _layer(
        "satellite",
        "Satellite",
        """
    vec3 c = li.imagery;
    c = (c - 0.5) * u_sat_contrast + 0.5 + u_sat_brightness;
    float grey = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return clamp(mix(vec3(grey), c, u_sat_saturation), 0.0, 1.0);""",
        flt("layer_satellite.brightness", "Brightness", 0.0, -0.5, 0.5, uniform="u_sat_brightness"),
        flt("layer_satellite.contrast", "Contrast", 1.0, 0.2, 2.5, uniform="u_sat_contrast"),
        flt("layer_satellite.saturation", "Saturation", 1.0, 0.0, 2.5, uniform="u_sat_saturation"),
        gain=1.0,
        tile_source="imagery",
    ),
    _layer(
        "elevation",
        "Elevation",
        """
    float t = (li.height - u_elev_min) / max(u_elev_max - u_elev_min, 1.0);
    if (u_elev_bands > 0.5) t = floor(t * u_elev_bands) / max(u_elev_bands - 1.0, 1.0);
    return colormap(u_elev_ramp, t);""",
        flt(
            "layer_elevation.min",
            "Minimum",
            500.0,
            -500.0,
            9000.0,
            step=10.0,
            decimals=0,
            unit="m",
            uniform="u_elev_min",
        ),
        flt(
            "layer_elevation.max",
            "Maximum",
            4500.0,
            -500.0,
            9000.0,
            step=10.0,
            decimals=0,
            unit="m",
            uniform="u_elev_max",
        ),
        enum("layer_elevation.ramp", "Colour ramp", "hypsometric", RAMPS, uniform="u_elev_ramp"),
        flt(
            "layer_elevation.bands",
            "Bands (0 = smooth)",
            0.0,
            0.0,
            40.0,
            step=1.0,
            decimals=0,
            uniform="u_elev_bands",
        ),
    ),
    _layer(
        "slope",
        "Slope",
        """
    if (u_slope_classes) return slope_classes(li.slope);
    return colormap(u_slope_ramp, li.slope / max(u_slope_max, 1.0));""",
        boolean(
            "layer_slope.classes",
            "Avalanche classes",
            True,
            uniform="u_slope_classes",
            tooltip="30-35° yellow, 35-40° orange, 40-45° red, > 45° violet",
        ),
        flt(
            "layer_slope.max",
            "Ramp maximum",
            60.0,
            5.0,
            90.0,
            step=1.0,
            decimals=0,
            unit="°",
            uniform="u_slope_max",
        ),
        enum("layer_slope.ramp", "Colour ramp", "turbo", RAMPS, uniform="u_slope_ramp"),
        shading=0.6,
    ),
    _layer(
        "aspect",
        "Aspect",
        """
    float flat_amount = 1.0 - smoothstep(2.0, 8.0, li.slope);
    vec3 c = hsv_to_rgb(vec3(li.aspect / 360.0, u_aspect_saturation, 0.95));
    return mix(c, vec3(0.9), flat_amount);""",
        flt("layer_aspect.saturation", "Saturation", 0.75, 0.0, 1.0, uniform="u_aspect_saturation"),
        shading=0.6,
    ),
    _layer(
        "hillshade",
        "Hillshade",
        """
    float az = radians(u_hs_azimuth), alt = radians(u_hs_altitude);
    vec3 l = vec3(sin(az) * cos(alt), cos(az) * cos(alt), sin(alt));
    vec3 n = normalize(vec3(li.normal_local.xy * u_hs_z, li.normal_local.z));
    float v = clamp(dot(n, l), 0.0, 1.0);
    return vec3(mix(0.08, 1.0, v));""",
        flt(
            "layer_hillshade.azimuth",
            "Light azimuth",
            315.0,
            0.0,
            360.0,
            step=1.0,
            decimals=0,
            unit="°",
            uniform="u_hs_azimuth",
        ),
        flt(
            "layer_hillshade.altitude",
            "Light altitude",
            45.0,
            5.0,
            90.0,
            step=1.0,
            decimals=0,
            unit="°",
            uniform="u_hs_altitude",
        ),
        flt("layer_hillshade.z", "Z factor", 1.5, 0.2, 5.0, uniform="u_hs_z"),
        shading=0.0,
    ),
]

LAYER_IDS = [layer.id for layer in LAYERS]


def layer_properties() -> list[PropertyDef]:
    options = [(layer.id, layer.label) for layer in LAYERS]
    slots = section(
        "Texture Layers",
        enum("layers.a", "Layer A", "satellite", options, uniform="u_layer_a"),
        enum("layers.b", "Layer B", "elevation", options, uniform="u_layer_b"),
        flt("layers.mix", "Blend A → B", 0.0, 0.0, 1.0, uniform="u_layer_mix"),
    )
    props = list(slots)
    for layer in LAYERS:
        props.extend(layer.properties)
    return props


def _uniform_declaration(d: PropertyDef) -> str:
    """GLSL uniform with the property default as initialiser (used when no store is bound)."""
    v = d.default
    if d.type is PType.FLOAT:
        return f"uniform float {d.uniform} = {float(v):.6g};"
    if d.type is PType.INT:
        return f"uniform int {d.uniform} = {int(v)};"
    if d.type is PType.BOOL:
        return f"uniform bool {d.uniform} = {'true' if v else 'false'};"
    if d.type is PType.ENUM:
        return f"uniform int {d.uniform} = {[o for o, _ in d.options].index(v)};"
    x, y, z = (float(c) ** 2.2 if d.type is PType.COLOR else float(c) for c in v)
    return f"uniform vec3 {d.uniform} = vec3({x:.6g}, {y:.6g}, {z:.6g});"


def generate_glsl(layers: list[Layer] = LAYERS) -> str:
    """Layer functions + dispatcher, included by the terrain shader."""
    out = ["// generated by earthling.render.layers - do not edit", '#include "colormaps.glsl"']
    for layer in layers:
        for d in layer.properties:
            if d.uniform:
                out.append(_uniform_declaration(d))
    for layer in layers:
        out.append(f"vec3 layer_{layer.id}(LayerInput li) {{{layer.glsl}\n}}")
    out.append("vec3 layer_albedo(int id, LayerInput li) {")
    for index, layer in enumerate(layers):
        out.append(f"    if (id == {index}) return layer_{layer.id}(li);")
    out.append("    return vec3(1.0, 0.0, 1.0);\n}")
    out.append("float layer_gain(int id) {")
    for index, layer in enumerate(layers):
        out.append(f"    if (id == {index}) return {layer.gain_uniform};")
    out.append("    return 1.0;\n}")
    out.append("float layer_shading(int id) {")
    for index, layer in enumerate(layers):
        out.append(f"    if (id == {index}) return {layer.shading_uniform};")
    out.append("    return 1.0;\n}")
    return "\n".join(out) + "\n"
