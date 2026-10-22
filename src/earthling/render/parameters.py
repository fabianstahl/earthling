"""Declarations of all scene/rendering properties, grouped by UI section.

Each render feature adds its section here. Keeping them in one module keeps ids unique and the
order of the UI sections stable.
"""

from __future__ import annotations

from datetime import datetime

from earthling.core.properties import (  # noqa: F401 (factories re-exported for scripts)
    PropertyDef,
    PropertyRegistry,
    PType,
    boolean,
    color,
    enum,
    flt,
    integer,
    section,
    shown_if,
    shown_unless,
)
from earthling.render.layers import layer_properties
from earthling.render.tracks import track_color
from earthling.render.vehicles import MARKER_OPTIONS
from earthling.render.weather import weather_properties

VIEW = section(
    "View",
    flt("view.fov", "Field of view", 50.0, 10.0, 120.0, step=0.5, decimals=1, unit="°"),
    boolean("view.show_outlines", "Area outlines", True, animatable=False, advanced=True,
            tooltip="Show the area of interest and resolution zone outlines"),
    boolean("view.preview", "Preview quality", False, animatable=False,
            tooltip="Faster viewport: half resolution, coarser terrain, smaller shadow maps. "
                    "Exports always use full quality."),
)  # fmt: skip

EXPORT = section(
    "Export",
    enum("export.resolution", "Resolution", "3840x2160",
         [("1280x720", "1280 × 720 (720p)"), ("1920x1080", "1920 × 1080 (1080p)"),
          ("2560x1440", "2560 × 1440 (1440p)"), ("3840x2160", "3840 × 2160 (4K UHD)"),
          ("1080x1920", "1080 × 1920 (portrait 9:16)"), ("2160x3840", "2160 × 3840 (portrait 4K)"),
          ("1080x1080", "1080 × 1080 (square)")],
         animatable=False),
    flt("export.detail", "Terrain detail", 0.75, 0.25, 4.0, step=0.05, unit="px/texel",
        logarithmic=True, animatable=False, advanced=True,
        tooltip="Detail threshold used for exported frames (smaller = sharper, slower)"),
    enum("export.shadow_resolution", "Shadow map size", "8192",
         [("4096", "4096"), ("8192", "8192")], animatable=False, advanced=True),
    enum("export.samples", "Anti-aliasing", "4",
         [("1", "Off"), ("4", "4 samples"), ("8", "8 samples"), ("16", "16 samples")],
         animatable=False,
         tooltip="Sub-pixel jittered renders averaged per exported frame (slower, smoother)"),
    flt("export.motion_blur", "Motion blur", 0.0, 0.0, 1.0, step=0.05, animatable=False,
        tooltip="Shutter as a fraction of the frame time (0.5 = 180° shutter). Uses the "
                "anti-aliasing samples, at least 8"),
)  # fmt: skip

FOLLOW = shown_if("camera.mode", "follow")

CAMERA = section(
    "Camera",
    PropertyDef("camera.pose", "Camera", PType.CAMERA, (0.0, -5000.0, 3000.0, 0.0, -30.0, 0.0),
                tooltip="Animated camera position and orientation (keyed with 'Add camera key')"),
    enum("camera.mode", "Camera mode", "keys",
         [("keys", "Keyframed path"), ("look_at", "Path + look at target"),
          ("follow", "Follow the hiker")],
         tooltip="Keyframe this to switch modes during the video"),
    flt("camera.transition", "Mode transition", 2.0, 0.0, 20.0, step=0.1, decimals=1, unit="s",
        animatable=False, advanced=True, tooltip="Blend time when the camera mode changes"),
    boolean("camera.constant_speed", "Constant camera speed", False,
            tooltip="Travel the camera path at constant speed between the first and last key"),
    PropertyDef("camera.target", "Look-at target", PType.VEC3, (0.0, 0.0, 0.0), decimals=0,
                when=shown_if("camera.mode", "look_at"),
                tooltip="ENU metres; use View > Pick Look-at Target (T) to click it on the map"),
    flt("follow.distance", "Follow distance", 600.0, 10.0, 20000.0, step=10.0, decimals=0,
        unit="m", logarithmic=True, when=FOLLOW),
    flt("follow.height", "Follow height", 250.0, -500.0, 10000.0, step=10.0, decimals=0,
        unit="m", when=FOLLOW),
    flt("follow.angle", "Follow angle", 0.0, -180.0, 180.0, step=1.0, decimals=0, unit="°",
        when=FOLLOW, tooltip="0 = straight behind the hiker, 90 = from the side"),
    flt("follow.lag", "Follow lag", 1.0, 0.0, 10.0, step=0.1, decimals=1, unit="s", when=FOLLOW),
    flt("follow.look_ahead", "Look ahead", 300.0, 0.0, 5000.0, step=10.0, decimals=0, unit="m",
        when=FOLLOW, advanced=True),
    flt("follow.bank", "Banking", 0.0, 0.0, 3.0, step=0.05, when=FOLLOW, advanced=True,
        tooltip="Lean into the turns (degrees of roll per degree of turn over 250 m), "
                "like a pilot or a cyclist"),
)  # fmt: skip

TERRAIN = section(
    "Terrain",
    flt("terrain.exaggeration", "Vertical exaggeration", 1.0, 0.1, 5.0, step=0.05, unit="×",
        uniform="u_exaggeration"),
    flt("terrain.detail_normals", "Close-range relief", 0.5, 0.0, 3.0, step=0.05,
        uniform="u_detail_normals",
        tooltip="Fine relief from the imagery near the camera (rocks, trees)"),
    flt("terrain.detail_distance", "Close-range relief distance", 3.0, 0.2, 30.0, step=0.1,
        decimals=1, unit="km", uniform="u_detail_distance_km", advanced=True),
    flt("terrain.detail", "Detail threshold", 1.0, 0.25, 8.0, step=0.05, unit="px/texel",
        logarithmic=True, animatable=False, advanced=True,
        tooltip="Refine terrain until one imagery texel covers at most this many screen pixels"),
    integer("terrain.memory_budget_mb", "GPU memory budget", 3000, 256, 16000, step=128,
            unit="MB", animatable=False, advanced=True),
    boolean("terrain.debug_lod", "Show LOD levels", False, uniform="u_debug_lod",
            animatable=False, advanced=True, tooltip="Debug: colour the terrain levels of detail"),
)  # fmt: skip

SUN = section(
    "Sun & Time",
    PropertyDef("sun.datetime", "Local date & time", PType.DATETIME, datetime(2026, 7, 1, 12, 0),
                when=shown_unless("sun.follow_track", True),
                tooltip="Local time in the project's timezone; drives the sun position"),
    flt("sun.azimuth_offset", "Azimuth offset", 0.0, -180.0, 180.0, step=1.0, decimals=1,
        unit="°", advanced=True, tooltip="Artistic rotation of the computed sun position"),
    flt("sun.elevation_offset", "Elevation offset", 0.0, -45.0, 45.0, step=0.5, decimals=1,
        unit="°", advanced=True),
    flt("sun.intensity", "Sun intensity", 1.4, 0.0, 6.0),
    color("sun.color", "Sun colour", (1.0, 0.96, 0.9), advanced=True),
    boolean("sun.follow_track", "Follow the GPX time", False,
            tooltip="The sun follows the recorded time at the head of the drawn track "
                    "(instead of the date & time above)"),
    flt("sun.track_offset", "GPX time offset", 0.0, -12.0, 12.0, step=0.25, unit="h",
        when=shown_if("sun.follow_track"),
        tooltip="Shift the GPX time, e.g. to start a day at sunrise"),
)  # fmt: skip

LIGHT = section(
    "Lighting",
    flt("light.ambient", "Sky ambient", 0.45, 0.0, 3.0),
    flt("light.night_ambient", "Night ambient", 1.0, 0.0, 10.0, advanced=True,
        tooltip="Brightness of moon/starlight on the terrain at night"),
    color("light.sky_color", "Sky ambient colour", (0.62, 0.72, 0.92)),
    color("light.ground_color", "Ground bounce colour", (0.45, 0.4, 0.33), advanced=True),
)  # fmt: skip

NIGHT = section(
    "Night Lights",
    flt("night.lights", "City lights", 0.0, 0.0, 10.0, step=0.1, uniform="u_night_lights",
        tooltip="Lights of the cities where it is night (NASA Black Marble, up to ~600 m "
                "detail: for views from far up)"),
    color("night.color", "Light colour", (1.0, 0.72, 0.42), uniform="u_night_color"),
)  # fmt: skip

WATER = section(
    "Water",
    boolean("water.enabled", "Water surface", False, uniform="u_water_enabled", switch=True,
            tooltip="Sea and lakes (OpenStreetMap water areas) reflect the sky and glint in the "
                    "sun, with waves close up"),
    flt("water.reflection", "Reflection", 1.0, 0.0, 2.0, uniform="u_water_reflection"),
    flt("water.waves", "Waves", 1.0, 0.0, 3.0, uniform="u_water_waves",
        tooltip="Height of the waves close up"),
    flt("water.wind", "Wind", 5.0, 0.0, 25.0, step=0.5, unit="m/s", uniform="u_water_wind",
        tooltip="Rougher water spreads the sun's glint"),
    flt("water.wind_dir", "Waves towards", 60.0, 0.0, 360.0, step=5.0, decimals=0, unit="°",
        uniform="u_water_wind_dir", advanced=True),
    flt("water.uniform", "Even open water", 0.0, 0.0, 1.0, uniform="u_water_body", advanced=True,
        tooltip="Open water (away from the shore) takes the colour below instead of the "
                "imagery: no seams between imagery tiles out at sea"),
    color("water.color", "Open water colour", (0.0014, 0.0132, 0.0358),
          uniform="u_water_body_color", advanced=True),
)  # fmt: skip

SHADOWS = section(
    "Shadows",
    boolean("shadows.enabled", "Terrain shadows", True, switch=True),
    flt("shadows.strength", "Shadow strength", 1.0, 0.0, 1.0, uniform="u_shadow_strength"),
    flt("shadows.distance", "Shadow distance", 40.0, 2.0, 200.0, step=1.0, decimals=0,
        unit="km", logarithmic=True, animatable=False, advanced=True,
        tooltip="Shadows are computed up to this distance from the camera"),
    enum("shadows.resolution", "Shadow map size", "4096",
         [("2048", "2048 (fast)"), ("4096", "4096"), ("8192", "8192 (export quality)")],
         animatable=False, advanced=True),
    flt("shadows.softness", "Softness", 1.5, 0.5, 4.0, uniform="u_shadow_softness",
        animatable=False),
    flt("shadows.bias", "Bias", 1.0, 0.0, 5.0, uniform="u_shadow_bias", animatable=False,
        advanced=True,
        tooltip="Increase if shadow acne (stripes) appears on slopes"),
)  # fmt: skip

SKY = section(
    "Sky & Atmosphere",
    flt("sky.illuminance", "Sky brightness", 20.0, 0.0, 100.0, step=0.5, decimals=1,
        uniform="u_sun_illuminance"),
    flt("sky.rayleigh", "Air density", 1.0, 0.0, 4.0, uniform="u_rayleigh_scale",
        tooltip="Rayleigh scattering: blue sky, red sunsets"),
    flt("sky.haze", "Haze", 1.0, 0.0, 10.0, uniform="u_mie_scale",
        tooltip="Mie scattering by aerosols: whitish haze and glow around the sun"),
    flt("haze.aerial", "Aerial perspective", 1.0, 0.0, 4.0, uniform="u_aerial_strength",
        keywords="fog distance",
        tooltip="Physically based haze between camera and terrain (uses the sky model)"),
    flt("sky.sun_disc", "Sun disc", 1.0, 0.0, 5.0, uniform="u_sun_disc", advanced=True),
    flt("sky.stars", "Star brightness", 1.0, 0.0, 10.0, uniform="u_star_brightness",
        advanced=True),
)  # fmt: skip

FOG = section(
    "Height Fog",
    boolean("fog.enabled", "Height fog", False, uniform="u_fog_enabled", switch=True),
    flt("fog.density", "Fog density", 0.3, 0.0, 20.0, step=0.01, unit="/km",
        logarithmic=False, tooltip="Extinction per kilometre at the fog base height"),
    flt("fog.base", "Fog base height", 1200.0, -500.0, 6000.0, step=10.0, decimals=0, unit="m",
        uniform="u_fog_base"),
    flt("fog.falloff", "Fog falloff", 250.0, 10.0, 5000.0, step=10.0, decimals=0, unit="m",
        logarithmic=True, uniform="u_fog_falloff",
        tooltip="Height over which the fog density drops to 37 %"),
    color("fog.color", "Fog colour", (0.85, 0.88, 0.92)),
    flt("fog.sun_scatter", "Sun glow", 1.0, 0.0, 5.0, advanced=True,
        tooltip="How strongly sunlight scatters in the fog (glow towards the sun)"),
    flt("fog.anisotropy", "Glow sharpness", 0.6, 0.0, 0.95, uniform="u_fog_glow", advanced=True),
)  # fmt: skip

POST = section(
    "Exposure",
    flt("post.exposure", "Exposure", 0.0, -8.0, 8.0, step=0.1, decimals=1, unit="EV"),
    enum("post.tonemap", "Tonemapping", "aces",
         [("aces", "ACES filmic"), ("reinhard", "Reinhard"), ("none", "None (clip)")],
         uniform="u_tonemap"),
)  # fmt: skip

RAMP = shown_unless("tracks.color_by", "none")

TRACKS = section(
    "Tracks",
    boolean("tracks.visible", "Show tracks", True, switch=True),
    flt("tracks.width", "Width", 5.0, 0.5, 200.0, step=0.5, decimals=1, logarithmic=True,
        uniform="u_track_width", tooltip="Pixels or metres, see width mode"),
    enum("tracks.width_mode", "Width mode", "pixels",
         [("pixels", "Constant on screen (px)"), ("meters", "In the world (m)")],
         uniform="u_track_width_mode"),
    flt("tracks.min_px", "Minimum width", 1.5, 0.0, 10.0, step=0.1, decimals=1, unit="px",
        uniform="u_track_min_px", advanced=True),
    enum("tracks.elevation", "Height source", "dem",
         [("dem", "Terrain (DEM)"), ("gpx", "GPX elevation")], animatable=False, advanced=True),
    flt("tracks.height_offset", "Height above ground", 3.0, -50.0, 500.0, step=0.5, decimals=1,
        unit="m", advanced=True),
    integer("tracks.smoothing", "Smoothing", 5, 1, 41, step=2, animatable=False, advanced=True,
            tooltip="Moving-average window in points"),
    enum("tracks.color_mode", "Colours", "per_track",
         [("per_track", "One colour per track / day"), ("single", "Single colour")]),
    color("tracks.color", "Colour", (1.0, 0.35, 0.15),
          when=shown_if("tracks.color_mode", "single")),
    enum("tracks.color_by", "Colour by", "none",
         [("none", "Track colours"), ("elevation", "Elevation"), ("gradient", "Gradient"),
          ("speed", "Speed"), ("heart_rate", "Heart rate")],
         tooltip="Colour the walked tracks by a value along them (a colour ramp)"),
    flt("tracks.color_min", "Ramp start", 0.0, -1000.0, 9000.0, step=0.5, decimals=1,
        when=RAMP, tooltip="Value at the start of the colour ramp (m, %, km/h or bpm)"),
    flt("tracks.color_max", "Ramp end", 12.0, -1000.0, 9000.0, step=0.5, decimals=1, when=RAMP),
    enum("tracks.color_ramp", "Colour ramp", "turbo",
         [("hypsometric", "Hypsometric"), ("viridis", "Viridis"), ("turbo", "Turbo"),
          ("grayscale", "Grayscale")], when=RAMP),
    flt("tracks.opacity", "Opacity", 1.0, 0.0, 1.0, uniform="u_track_opacity"),
    flt("tracks.emissive", "Self-illumination", 0.5, 0.0, 1.0, uniform="u_track_emissive",
        tooltip="0 = lit by sun and sky, 1 = glows on its own (visible at night)"),
    flt("tracks.outline", "Outline", 1.0, 0.0, 5.0, step=0.1, decimals=1, unit="px",
        uniform="u_track_outline", advanced=True),
    flt("tracks.glow", "Glow emission", 1.2, 0.0, 20.0, step=0.1, decimals=1,
        uniform="u_track_glow", tooltip="Light the track emits into the glow effect"),
    flt("tracks.depth_bias", "Stay on top", 0.002, 0.0, 0.05, step=0.001, decimals=3,
        uniform="u_track_depth_bias", advanced=True,
        tooltip="Share of the distance the track is lifted towards the camera, so it is not "
                "swallowed by the coarser terrain far away"),
    flt("tracks.casing", "Casing", 0.0, 0.0, 30.0, step=0.5, decimals=1, unit="px",
        uniform="u_track_casing",
        tooltip="Soft border around the tracks: sets them apart from busy imagery"),
    color("tracks.casing_color", "Casing colour", (0.0004, 0.0004, 0.0014),
          uniform="u_casing_color", advanced=True),
    flt("tracks.casing_opacity", "Casing opacity", 0.75, 0.0, 1.0, uniform="u_casing_opacity",
        advanced=True),
)  # fmt: skip

PROGRESS = section(
    "Hike Progress",
    flt("progress.head", "Progress", 1.0, 0.0, 1.0, step=0.001, decimals=4,
        tooltip="How much of the hike is drawn (0 = start, 1 = everything)"),
    flt("progress.tail", "Start", 0.0, 0.0, 1.0, step=0.001, decimals=4,
        tooltip="Hide the hike before this point (show only a part)"),
    enum("progress.mode", "Progress by", "distance",
         [("distance", "Distance"), ("time", "GPX time (pauses at night)"),
          ("clock", "Clock (all tracks at once)")],
         tooltip="Clock: every track is drawn up to the same time, each with its own marker "
                 "(a race of two runs: shift one group in time to start them together)"),
    flt("progress.head_fade", "Head highlight length", 150.0, 0.0, 5000.0, step=10.0,
        decimals=0, unit="m", logarithmic=False, uniform="u_head_fade", advanced=True),
    flt("progress.head_boost", "Head highlight", 1.5, 0.0, 10.0, uniform="u_head_boost",
        advanced=True),
) + section(
    "Marker",
    boolean("marker.visible", "Hiker marker", True, switch=True),
    flt("marker.size", "Marker size", 14.0, 2.0, 80.0, step=0.5, decimals=1, unit="px",
        uniform="u_marker_size"),
    color("marker.color", "Marker colour", (1.0, 0.95, 0.8)),
    flt("marker.pulse", "Pulse rate", 0.8, 0.0, 5.0, step=0.05, unit="Hz",
        uniform="u_marker_pulse"),
    flt("marker.glow", "Marker glow", 3.0, 0.0, 20.0, uniform="u_marker_glow", advanced=True),
)  # fmt: skip

GLOW = section(
    "Glow",
    boolean("glow.enabled", "Glow (bloom)", True, switch=True),
    flt("glow.intensity", "Intensity", 0.8, 0.0, 5.0, uniform="u_bloom_intensity"),
    flt("glow.radius", "Spread", 1.0, 0.3, 3.0),
    integer("glow.levels", "Size", 6, 1, 7, animatable=False, advanced=True,
            tooltip="Number of blur levels: larger values give a wider halo"),
)  # fmt: skip


LABELS_ON = shown_if("labels.visible")

# (sections are split for the inspector; the declaration order stays as it was)
LABELS = section(
    "Labels",
    boolean("labels.visible", "Show labels", True, switch=True,
            tooltip="Peaks, passes, places and huts from OpenStreetMap"),
    flt("labels.opacity", "Opacity", 1.0, 0.0, 1.0),
) + section(
    "Which Labels",
    boolean("labels.peaks", "Peaks", True, when=LABELS_ON),
    boolean("labels.passes", "Passes", True, when=LABELS_ON),
    boolean("labels.places", "Places", True, when=LABELS_ON),
    boolean("labels.huts", "Huts", False, when=LABELS_ON),
    boolean("labels.cities", "Large cities", False, when=LABELS_ON,
            tooltip="Natural Earth: cities above the minimum population, at any scale"),
    boolean("labels.countries", "Countries", False, when=LABELS_ON,
            tooltip="Natural Earth: country names in capitals"),
    flt("labels.min_population", "Minimum population (cities)", 1_000_000.0, 100_000.0,
        20_000_000.0, step=100_000.0, decimals=0, when=LABELS_ON + shown_if("labels.cities")),
    flt("labels.min_elevation", "Minimum elevation", 0.0, 0.0, 5000.0, step=50.0, decimals=0,
        unit="m", when=LABELS_ON, tooltip="Hide peaks, passes and huts below this elevation"),
    flt("labels.min_prominence", "Minimum prominence", 150.0, 0.0, 2000.0, step=10.0,
        decimals=0, unit="m", when=LABELS_ON,
        tooltip="Hide minor peaks (estimated drop to the nearest higher peak)"),
    flt("labels.max_track_distance", "Maximum distance to the track", 0.0, 0.0, 50.0, step=0.5,
        decimals=1, unit="km", when=LABELS_ON,
        tooltip="Only features near the tracks (0 = no limit)"),
    boolean("labels.follow_progress", "Appear when passed", False, when=LABELS_ON,
            tooltip="Labels near the drawn track appear once the head passes them (the towns "
                    "and villages of the way so far)"),
    flt("labels.progress_fade", "Appear over", 500.0, 0.0, 20000.0, step=50.0, decimals=0,
        unit="m", when=LABELS_ON + shown_if("labels.follow_progress"),
        tooltip="Track distance over which a passed label fades in"),
) + section(
    "Labels",
    flt("labels.max_distance", "Fade-out distance", 30.0, 1.0, 5000.0, step=1.0, decimals=0,
        unit="km", advanced=True, tooltip="Labels fade out towards this distance from the camera"),
    integer("labels.max_count", "Maximum labels", 40, 1, 300, advanced=True),
    flt("labels.size", "Text size", 18.0, 6.0, 72.0, step=0.5, decimals=1, unit="px",
        tooltip="At 1080p; scales with the output resolution"),
    boolean("labels.elevations", "Show elevations", True),
    flt("labels.leader", "Leader line", 36.0, 0.0, 200.0, step=1.0, decimals=0, unit="px",
        advanced=True),
    color("labels.color", "Colour", (1.0, 1.0, 1.0)),
    color("labels.outline_color", "Outline colour", (0.06, 0.06, 0.07), advanced=True),
    flt("labels.outline", "Outline", 2.0, 0.0, 8.0, step=0.1, decimals=1, unit="px",
        advanced=True),
)  # fmt: skip


STATS_ON = shown_if("stats.visible")
CORNERS = [("bottom_left", "Bottom left"), ("bottom_right", "Bottom right"),
           ("top_left", "Top left"), ("top_right", "Top right")]  # fmt: skip

STATS = section(
    "Stats Overlay",
    boolean("stats.visible", "Show stats", False, switch=True,
            tooltip="Day, date, distance, ascent, elevation and profile at the hike progress"),
    flt("stats.opacity", "Opacity", 1.0, 0.0, 1.0, advanced=True),
    enum("stats.position", "Position", "bottom_left", CORNERS),
    flt("stats.scale", "Size", 1.0, 0.4, 3.0, step=0.05),
    enum("stats.scope", "Totals and profile", "hike",
         [("hike", "Whole hike"), ("day", "Current day"), ("track", "Current track")]),
) + section(
    "Stats Values",
    boolean("stats.show_day", "Day number", True, when=STATS_ON),
    enum("stats.day_style", "Day as", "day", [("day", "Day 5"), ("count", "5 days")],
         when=STATS_ON + shown_if("stats.show_day"),
         tooltip="The current day, or the number of days so far (e.g. for a summary)"),
    boolean("stats.show_date", "Date", True, when=STATS_ON),
    boolean("stats.show_time", "Time of day", False, when=STATS_ON),
    enum("stats.timezone", "Time zone", "project",
         [("project", "The project's time zone"), ("position", "Where the head is")],
         when=STATS_ON + shown_if("stats.show_time"),
         tooltip="For journeys across time zones: the local time where the track is"),
    boolean("stats.show_utc_offset", "Show the UTC offset", False,
            when=STATS_ON + shown_if("stats.show_time"),
            tooltip="E.g. 14:05 (UTC+8) - shows when the clock jumps to another time zone"),
) + section(
    "Stats Overlay",
    enum("stats.language", "Language", "en", [("en", "English"), ("de", "Deutsch")],
         animatable=False, keywords="german deutsch sprache",
         tooltip="Words, dates and numbers in the overlays"),
) + section(
    "Stats Values",
    enum("stats.date_format", "Date format", "long",
         [("long", "14 July 2026"), ("short", "14 Jul"), ("iso", "2026-07-14")],
         when=STATS_ON + shown_if("stats.show_date")),
    boolean("stats.show_distance", "Distance", True, when=STATS_ON),
    boolean("stats.show_ascent", "Ascent", True, when=STATS_ON),
    boolean("stats.show_elevation", "Elevation", True, when=STATS_ON),
    boolean("stats.show_speed", "Speed", False, when=STATS_ON),
    enum("stats.speed_unit", "Speed as", "kmh",
         [("kmh", "km/h"), ("pace", "Pace (min/km)"), ("knots", "Knots")],
         when=STATS_ON + shown_if("stats.show_speed")),
    boolean("stats.show_gradient", "Gradient", False, when=STATS_ON),
    boolean("stats.show_gap", "Gap", False, when=STATS_ON + shown_if("progress.mode", "clock"),
            tooltip="Progress by clock: how far (in time) the second track is behind the first"),
    boolean("stats.show_heart_rate", "Heart rate", False, when=STATS_ON,
            tooltip="From the GPX files (Garmin / Strava heart rate extension)"),
    PropertyDef("stats.text", "Extra line", PType.TEXT, "", when=STATS_ON,
                tooltip="A line of your own under the title (keys switch it), e.g. a counter"),
) + section(
    "Elevation Profile",
    boolean("stats.profile", "Elevation profile", True, switch=True, when=STATS_ON),
    flt("stats.profile_width", "Profile width", 420.0, 100.0, 1600.0, step=10.0, decimals=0,
        unit="px", when=STATS_ON),
    flt("stats.profile_height", "Profile height", 90.0, 30.0, 400.0, step=5.0, decimals=0,
        unit="px", when=STATS_ON),
) + section(
    "Stats Overlay",
    color("stats.text_color", "Text colour", (1.0, 1.0, 1.0), advanced=True),
    color("stats.accent_color", "Accent colour", (1.0, 0.55, 0.2), advanced=True),
    flt("stats.background", "Background", 0.35, 0.0, 1.0, advanced=True),
    flt("stats.outline", "Text shadow", 0.6, 0.0, 1.0, advanced=True),
    boolean("stats.attribution", "Data attribution", False,
            tooltip="Credits for the imagery, elevation and label data (bottom right)"),
)  # fmt: skip


MINIMAP = section(
    "Mini-map",
    boolean("minimap.visible", "Show the mini-map", False, switch=True,
            tooltip="The whole journey on a small map: the route, the drawn part, the head"),
    enum("minimap.position", "Position", "top_right", CORNERS),
    flt("minimap.size", "Width", 300.0, 100.0, 1000.0, step=10.0, decimals=0, unit="px"),
    flt("minimap.scale", "Size", 1.0, 0.4, 3.0, step=0.05, advanced=True),
    flt("minimap.opacity", "Opacity", 0.9, 0.0, 1.0),
    color("minimap.land_color", "Land colour", (0.25, 0.27, 0.24), advanced=True),
    color("minimap.water_color", "Water colour", (0.12, 0.22, 0.34), advanced=True),
)  # fmt: skip


def track_group_properties(groups) -> list[PropertyDef]:
    """Keyframeable style properties of every track group ("trackgroup.<name>.*"); ``groups``:
    the session's TrackGroups."""
    defs = []
    for g in groups:
        planned = g.role == "planned"
        p = f"trackgroup.{g.name}."
        members = [("none", "None")] + [(t.name, t.name) for t in g.tracks]
        vehicle = shown_unless(p + "marker", "dot")
        defs += section(
            f"Tracks: {g.label}",
            boolean(p + "visible", "Show", True, switch=True),
            flt(p + "opacity", "Opacity", 0.9 if planned else 1.0, 0.0, 1.0),
            enum(p + "color_mode", "Colours", "per_track" if planned else "auto",
                 [("auto", "As in the Tracks section"), ("per_track", "One colour per track"),
                  ("single", "Group colour")]),
            color(p + "color", "Group colour",
                  (0.95, 0.95, 0.95) if planned else (1.0, 0.35, 0.15),
                  when=shown_if(p + "color_mode", "single")),
            flt(p + "width", "Width factor", 0.6 if planned else 1.0, 0.05, 5.0, step=0.05),
            flt(p + "dash", "Dashes", 10.0 if planned else 0.0, 0.0, 100.0, step=1.0,
                decimals=0, unit="px", tooltip="Dash length on screen (0 = solid line)"),
            boolean(p + "dots", "Dotted", False, advanced=True,
                    tooltip="Round dots instead of dashes; Dashes sets their spacing"),
            flt(p + "glow", "Glow factor", 0.3 if planned else 1.0, 0.0, 5.0),
            enum(p + "highlight", "Highlight", "none", members,
                 tooltip="Show one track in the highlight colour and dim the others"),
            color(p + "highlight_color", "Highlight colour", (0.25, 0.95, 0.35),
                  when=shown_unless(p + "highlight", "none")),
            flt(p + "dim", "Others while highlighting", 0.35, 0.0, 1.0,
                when=shown_unless(p + "highlight", "none")),
            enum(p + "marker", "Head marker", "dot", MARKER_OPTIONS,
                 tooltip="The glowing dot, or a vehicle (turned to the direction of travel) "
                         "while the head is on this group's tracks"),
            flt(p + "marker_size", "Vehicle size", 46.0, 8.0, 200.0, step=1.0, decimals=0,
                unit="px", when=vehicle),
            color(p + "marker_tint", "Vehicle colour", (1.0, 1.0, 1.0), when=vehicle,
                  tooltip="Multiplies the vehicle icon (e.g. gold and silver runners)"),
            PropertyDef(p + "marker_label", "Vehicle label", PType.TEXT, "", when=vehicle,
                        tooltip="A caption moving with the vehicle, e.g. the year of a run"),
            color(p + "water_color", "On water: colour", (0.35, 0.75, 1.0), advanced=True,
                  tooltip="Groups split by the water mask (water = true): the parts on "
                          "water"),
            flt(p + "water_dash", "On water: dashes", 0.0, 0.0, 100.0, step=1.0,
                decimals=0, unit="px", advanced=True),
            boolean(p + "water_dots", "On water: dotted", False, advanced=True),
            flt(p + "time_offset", "Time shift", 0.0, -1e6, 1e6, step=0.01, decimals=4,
                unit="h", animatable=False, when=shown_if("progress.mode", "clock"),
                tooltip="Progress by clock: shift the group's times (e.g. start last year's "
                        "run together with this year's)"),
        )  # fmt: skip
        # one colour per track (used with "One colour per track")
        defs += section(
            f"Tracks: {g.label} – colours",
            *(color(p + f"track{i}.color", t.name, track_color(i))
              for i, t in enumerate(g.tracks)),
        )  # fmt: skip
    return defs


def build_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    for group in (
        VIEW,
        CAMERA,
        EXPORT,
        POST,
        TERRAIN,
        SUN,
        LIGHT,
        NIGHT,
        WATER,
        SHADOWS,
        SKY,
        FOG,
        TRACKS,
        PROGRESS,
        GLOW,
        LABELS,
        STATS,
        MINIMAP,
    ):
        registry.extend(group)
    registry.extend(layer_properties())
    registry.extend(weather_properties())
    return registry
