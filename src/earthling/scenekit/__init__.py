"""The scene kit: helpers for *scene scripts*, Python files that build a scene (camera path,
progress, light, POIs) from a project's tracks and save it as ``scenes/<name>.json`` for
``earthling render`` / ``earthling stills``.

A minimal script::

    from earthling.scenekit import follow_shots, head_keys_for, key_shots, load_journey, \\
        new_scene, save

    journey = load_journey("path/to/project")          # tracks on the terrain, no GPU needed
    scene = new_scene(journey, duration=20.0)
    head_of = head_keys_for(scene, journey, 1, 1.0, 19.0, realtime=False)  # draw track 1
    times = [i / 30 for i in range(601)]
    key_shots(scene, follow_shots(journey, times, head_of, distance=2500, pitch=-35, side=30))
    save(scene, "s01_first", holds=True)                 # still seconds at both ends

Project specifics (names of places, track-numbering conventions, the look of a series of
videos) stay in the project's own helper module, which usually starts with
``from earthling.scenekit import *``. Positions are ENU metres in the scene's local frame,
times are timeline seconds, epochs are UTC seconds.
"""

from earthling.core.animation import Interp
from earthling.core.easing import gaussian, lerp, pchip, pchip_array, smoother, smoother_soft
from earthling.scenekit.camera import (
    KEY_STEP,
    View,
    bearing,
    blend_shots,
    follow_shots,
    geo_orbit_eye,
    heading_of,
    key_shots,
    level_roll,
    line_of_sight,
    line_shots,
    map_shots,
    orbit_eye,
    orbit_shots,
    plan_camera,
    plan_look,
    pose,
    still_shots,
)
from earthling.scenekit.journey import (
    Journey,
    current,
    distance_near,
    epoch_at,
    first_reach,
    load_journey,
    local,
    nearest_on,
    set_current,
    stop_near,
    stops,
    track_highest,
    track_point_near,
    track_time_at,
)
from earthling.scenekit.scene import (
    HOLD_S,
    add_holds,
    clamp_sun,
    clear_keys,
    daylight_keys,
    draw_keys,
    head_distance_of,
    head_keys_for,
    key,
    key_head,
    new_scene,
    night_light,
    poi,
    reached_at,
    save,
    show,
    sun_keys,
    travel_scene,
    travel_time_of,
)

__all__ = [
    "add_holds", "bearing", "blend_shots", "clamp_sun", "clear_keys", "current",
    "daylight_keys", "distance_near", "draw_keys", "epoch_at", "first_reach", "follow_shots",
    "gaussian", "geo_orbit_eye", "head_distance_of", "head_keys_for", "heading_of", "HOLD_S",
    "Interp", "Journey", "key", "key_head", "key_shots", "KEY_STEP", "lerp", "level_roll",
    "line_of_sight", "line_shots", "load_journey", "local", "map_shots", "nearest_on",
    "new_scene", "night_light", "orbit_eye", "orbit_shots", "pchip", "pchip_array",
    "plan_camera", "plan_look", "poi", "pose", "reached_at", "save", "set_current", "show",
    "smoother", "smoother_soft", "still_shots", "stop_near", "stops", "sun_keys",
    "track_highest", "track_point_near", "track_time_at", "travel_scene", "travel_time_of",
    "View",
]  # fmt: skip
