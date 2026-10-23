# Showcase: eleven very different scenes

Eleven synthetic scenes (and a few variants for the trailer), each pushing a different part of
Earthling (roadmap phase 19). The GPX tracks are generated (`tools/gen_tracks.py`): real paths,
roads and waterways from OpenStreetMap, plausible times from simple speed models, a heart rate
for the bike and the runs.

| Scene | Where | Moves by | Weather / light | Shows |
|---|---|---|---|---|
| `s01_patagonia` | Torres del Paine | on foot | wind, gusting clouds | a dive from space into the valley, the globe |
| `s02_hurtigruten` | Bergen – Kirkenes | ship | drizzle, midnight sun | a calm north-up map view with a mini-map, the water surface, a ship marker, knots |
| `s03_stelvio` | Stelvio pass | road bike | valley fog | speed, gradient, heart rate, colour by gradient, a hairpin counter |
| `s04_grand_canyon` | rim to rim | on foot | monsoon storm, lightning | texture layers, a locked-off time-lapse |
| `s04_grand_canyon_storm` | Bright Angel Canyon | (the trail, drawn) | dusk, the storm closing in | a low flight along the trail below the rims, lightning ahead (`tools/s04_grand_canyon_storm.py`, for the trailer) |
| `s05_kilimanjaro` | summit night | on foot | sea of clouds, night to sunrise | the sun from the GPX time, a glowing track |
| `s06_iceland` | ring road | car + hikes | cloud changing by the day | several track groups, a map view with pauses at every change, a mini-map, photos of the sights appearing when the line gets there, still seconds at both ends |
| `s07_berlin` | marathon | running | overcast, drizzle | two runs at once (by clock): gold and silver runners with their year, km badges and water stations, 9:16 portrait |
| `s08_everest` | Lukla – Base Camp | plane + on foot | snowfall, clearing | a follow-mode flight, snow cover, peak labels (Everest from Kala Patthar) |
| `s08_everest_sunrise` | Everest from the west | (the trek, drawn) | dawn, a sea of clouds | the sun breaking out beside Lhotse, a glide towards the South Col (`tools/s08_everest_sunrise.py`) |
| `s08_everest_summit` | Base Camp – summit (South Col route) | on foot | a thunderstorm below, first light | the route drawn up to the summit, lightning inside the cloud deck, one rising arc, then a fast pull-out to the globe (`tools/s08_everest_summit.py`, the trailer's finale) |
| `s09_boundary_waters` | lakes of Minnesota | canoe | late afternoon sun | paddling and portages split by the water mask, glint on the lakes |
| `s10_trans_siberian` | Moscow – Beijing | train | winter | day and night side, city lights, time zones, border map, country and city labels |
| `s11_k2` | Askole – Concordia (Baltoro) | on foot | snowfall, clearing | the snow clouds part and K2 appears, a long lens, Copernicus GLO-30 heights (`tools/s11_k2.py`) |

## Running

From the repository root:

```powershell
uv run python examples/showcase/tools/gen_tracks.py s03        # (re)generate GPX (network)
uv run earthling fetch examples/showcase/s03_stelvio            # download the data
uv run python examples/showcase/tools/s03_stelvio.py            # write s03_stelvio/scenes/s03_stelvio.json
uv run earthling stills examples/showcase/s03_stelvio --scene scenes/s03_stelvio.json 5 12 --out renders/showcase/stills   # preview
bash examples/showcase/tools/render_all.sh s03_stelvio          # 1080p60 video + QA into renders/showcase
```

The photos of `s06_iceland` are public domain / CC0 pictures from Wikimedia Commons
(`s06_iceland/photos/CREDITS.md`).

The scene scripts use the scene kit (`earthling.scenekit`: smooth chase, orbit and calm map
shots, holds, POIs); `tools/sc_common.py` only adds where the scene projects live and English
stats. `earthling qa <video>` makes a contact sheet and finds jumps and flashes in a video.
