# Showcase: ten very different scenes

Ten synthetic scenes, each pushing a different part of Earthling (roadmap phase 19). The GPX
tracks are generated (`tools/gen_tracks.py`): real paths, roads and waterways from
OpenStreetMap, plausible times from simple speed models, a heart rate for the bike and the runs.

| Scene | Where | Moves by | Weather / light | Shows |
|---|---|---|---|---|
| `s01_patagonia` | Torres del Paine | on foot | wind, gusting clouds | a dive from space into the valley, the globe |
| `s02_hurtigruten` | Bergen – Kirkenes | ship | drizzle, midnight sun | the water surface, a ship marker, knots |
| `s03_stelvio` | Stelvio pass | road bike | valley fog | speed, gradient, heart rate, colour by gradient, a hairpin counter |
| `s04_grand_canyon` | rim to rim | on foot | monsoon storm, lightning | texture layers, a locked-off time-lapse |
| `s05_kilimanjaro` | summit night | on foot | sea of clouds, night to sunrise | the sun from the GPX time, a glowing track |
| `s06_iceland` | ring road | car + hikes | four seasons a day, a rainbow | several track groups, fast weather changes |
| `s07_berlin` | marathon | running | overcast, drizzle | two runs at once (by clock), 9:16 portrait |
| `s08_everest` | Lukla – Base Camp | plane + on foot | snowfall | a banking first-person flight, snow cover, peak labels |
| `s09_boundary_waters` | lakes of Minnesota | canoe | morning mist | paddling and portages split by the water mask |
| `s10_trans_siberian` | Moscow – Beijing | train | winter | day and night side, city lights, time zones, border map |

## Running

From the repository root:

```powershell
uv run python examples/showcase/tools/gen_tracks.py s03        # (re)generate GPX (network)
uv run earthling fetch examples/showcase/s03_stelvio            # download the data
uv run python examples/showcase/tools/s03_stelvio.py            # write s03_stelvio/scenes/s03_stelvio.json
uv run python examples/showcase/tools/stills.py s03_stelvio/scenes/s03_stelvio.json 5 12   # preview
bash examples/showcase/tools/render_all.sh s03_stelvio          # 1080p60 video + QA into renders/showcase
```

`tools/sc_common.py` holds the camera helpers shared by the scene scripts (smooth chase and
orbit shots, holds, POIs), `tools/qa_video.py` makes a contact sheet and finds jumps in a video.
