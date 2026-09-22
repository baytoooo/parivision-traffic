# The scene

The organisers' `camera.md` was not in the starter kit, so this is our own
description, written from the four sample clips. All coordinates are in the
1920x1080 reference frame (the 4K clips downscaled by 2). The shapes live in
`src/parivision/scene.py`, and `docs/scene_overlay.jpg` shows them drawn on a
median background.

![overlay](scene_overlay.jpg)

## Camera

A Sony camcorder on a tripod, high up on a building at the south-west corner
of a signalised four-way junction in Tashkent. It looks north-west, straight
up the avenue. Files are XAVC S: 3840x2160, 29.97 fps, H.264 High 4:2:2
10-bit at about 140 Mbit/s. There is no camera motion inside a clip and all
four samples have the same framing to within a pixel (checked with a
homography between median backgrounds), but we still register every clip
against the reference in case a test clip is framed slightly differently.

| clip  | length  | local time (Tashkent)   | light                         |
|-------|---------|-------------------------|-------------------------------|
| C3896 | 340.3 s | 18 Sep 2026, 11:18      | midday sun, hard shadows      |
| C3897 | 317.8 s | 18 Sep 2026, 11:24      | midday sun, hard shadows      |
| C3902 | 317.8 s | 18 Sep 2026, 16:58      | low sun, long shadows         |
| C3905 | 127.6 s | 18 Sep 2026, 17:22      | dusk, underexposed, headlights on |

## Roads

* **Avenue, north arm.** Runs from the top-left of the frame down to the
  junction. A raised concrete median splits it.
  * **Southbound (SB) carriageway**, left of the median: traffic comes towards
    the camera. Four lanes at the stop line. The white stop line runs from
    (285, 527) to (930, 457), just before the north crossing.
  * **Northbound (NB) carriageway**, right of the median: traffic leaves the
    junction and drives away from the camera. There is a bus stop on its far
    kerb near the top of the frame.
* **Junction box.** The open asphalt in the lower half of the frame.
* **West arm** of the cross street: leaves the frame at the bottom-left,
  past three pink channelising islands.
* **East arm** of the cross street: leaves the frame on the right, below the
  NB crossing.
* **South arm** of the avenue: under the camera, off the bottom edge.

Southbound drivers at the stop line can go straight (exit bottom or
bottom-right), turn right into the west arm (they drive over the west
crossing), or turn left into the east arm.

## Crossings

* **North crossing**, zebra across both avenue carriageways, split by the
  median nose (the round kerb with the blue keep-right sign at about
  (1235, 555)). `north_sb` covers the SB half, `north_nb` the NB half.
* **West crossing**, a long zebra across the west arm, from the corner
  sidewalk at the left down to the bottom edge, between the pink islands.

Nothing else in view is a legal place to cross.

## Signals

* The gantry over the SB lanes carries the signal heads for SB traffic. They
  face away from the camera, so we only see their backs.
* Two heads **face the camera**: one on the pole at the left corner (about
  (280, 520)) and one on the median nose (about (1165, 385)). They are
  readable in every clip, including dusk. In all four samples they follow the
  avenue phase: red while pedestrians use the north crossing and the SB queue
  waits at the stop line, green while the SB queue moves.
* No pedestrian signal heads are readable.

We read the phase from those two heads (HSV colour of the lit lamp) and use
it as the SB signal state. If the junction ever gave the south approach its
own phase this would be wrong, but nothing in the samples suggests it does.

## Things that are easy to get wrong

* Parked cars along the left edge (x < 170, y 370 to 520) are off the
  carriageway.
* Buses sit at the NB bus stop for 20 to 60 s. They are at a stop, not a
  stopped vehicle.
* The SB queue at a red light can stand still for a minute or more. That is
  a queue at a signal, not congestion and not a stopped vehicle, unless it
  still does not move when the light is green.
* Cyclists and delivery riders are detected as a person plus a bicycle or
  motorcycle. They are vehicles, not pedestrians.
