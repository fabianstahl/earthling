"""Mapbox vector tiles (MVT, protobuf): a small decoder for polygon layers.

Only what the water mask needs: layer names, feature tags and polygon / line geometry in tile
coordinates (0..extent, y down). See https://github.com/mapbox/vector-tile-spec.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field

# protobuf wire types
VARINT, FIXED64, BYTES, FIXED32 = 0, 1, 2, 5


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(data: bytes):
    """(field number, wire type, value) of a protobuf message; BYTES values are slices."""
    pos, end = 0, len(data)
    while pos < end:
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == VARINT:
            value, pos = _varint(data, pos)
        elif wire == BYTES:
            length, pos = _varint(data, pos)
            value = data[pos : pos + length]
            pos += length
        elif wire == FIXED32:
            value = data[pos : pos + 4]
            pos += 4
        elif wire == FIXED64:
            value = data[pos : pos + 8]
            pos += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _packed(data: bytes) -> list[int]:
    out, pos = [], 0
    while pos < len(data):
        value, pos = _varint(data, pos)
        out.append(value)
    return out


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _value(data: bytes):
    import struct

    for number, _wire, v in _fields(data):
        if number == 1:
            return bytes(v).decode("utf-8", "replace")
        if number == 2:
            return struct.unpack("<f", v)[0]
        if number == 3:
            return struct.unpack("<d", v)[0]
        if number in (4, 5):
            return v
        if number == 6:
            return _zigzag(v)
        if number == 7:
            return bool(v)
    return None


@dataclass
class Feature:
    type: int  # 1 point, 2 line string, 3 polygon
    tags: dict[str, object]
    rings: list[list[tuple[int, int]]]  # polygons: rings, lines: parts, points: one list

    @property
    def is_polygon(self) -> bool:
        return self.type == 3


@dataclass
class Layer:
    name: str
    extent: int = 4096
    features: list[Feature] = field(default_factory=list)


def decode_geometry(commands: list[int]) -> list[list[tuple[int, int]]]:
    """Command stream -> parts (lists of absolute tile coordinates)."""
    parts: list[list[tuple[int, int]]] = []
    x = y = 0
    i = 0
    current: list[tuple[int, int]] | None = None
    while i < len(commands):
        command = commands[i] & 0x7
        count = commands[i] >> 3
        i += 1
        if command == 7:  # ClosePath
            if current:
                current.append(current[0])
            continue
        for _ in range(count):
            x += _zigzag(commands[i])
            y += _zigzag(commands[i + 1])
            i += 2
            if command == 1:  # MoveTo
                current = [(x, y)]
                parts.append(current)
            elif current is not None:  # LineTo
                current.append((x, y))
    return parts


def ring_area(ring: list[tuple[int, int]]) -> float:
    """Signed area (surveyor's formula, tile coordinates): > 0 for exterior rings."""
    s = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:], strict=False):
        s += x0 * y1 - x1 * y0
    return s / 2.0


def decode_tile(data: bytes, layers: set[str] | None = None) -> dict[str, Layer]:
    """Decode the wanted layers (all with None) of a (possibly gzipped) tile."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    out: dict[str, Layer] = {}
    for number, _wire, layer_bytes in _fields(data):
        if number != 3:
            continue
        name, extent, keys, values, raw = "", 4096, [], [], []
        for n, _w, v in _fields(layer_bytes):
            if n == 1:
                name = bytes(v).decode("utf-8", "replace")
            elif n == 2:
                raw.append(v)
            elif n == 3:
                keys.append(bytes(v).decode("utf-8", "replace"))
            elif n == 4:
                values.append(v)
            elif n == 5:
                extent = v
        if layers is not None and name not in layers:
            continue
        values = [_value(v) for v in values]
        layer = Layer(name, extent)
        for feature_bytes in raw:
            ftype, tags, geometry = 0, {}, []
            for n, _w, v in _fields(feature_bytes):
                if n == 2:
                    t = _packed(v)
                    tags = {keys[t[k]]: values[t[k + 1]] for k in range(0, len(t) - 1, 2)}
                elif n == 3:
                    ftype = v
                elif n == 4:
                    geometry = _packed(v)
            layer.features.append(Feature(ftype, tags, decode_geometry(geometry)))
        out[name] = layer
    return out
