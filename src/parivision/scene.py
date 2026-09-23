"""Scene layout of the intersection in reference coordinates.

Reference = the 1920x1080 downscale of the morning sample clips. The
afternoon clips are framed up to 60 px differently, which registration.py
takes care of. The organisers did not ship camera.md, so everything here was
traced by hand on a median background image (docs/scene.md has the picture
and the reasoning). At runtime the shapes are mapped into each clip with the
homography from ``registration.align``.

Naming: the avenue runs from the top-left of the frame towards the camera.
"SB" is the southbound carriageway (left of the median, traffic comes towards
the camera), "NB" the northbound one (right of the median, traffic goes away).
The cross street runs left-right across the lower part of the frame (west arm
at the bottom-left, east arm off the right edge).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REF_SIZE = (1920, 1080)

# Everything a vehicle can legally drive on, as one outline, minus the holes below.
ROAD = [
    (70, 100), (95, 45), (450, 95), (700, 185), (900, 240), (1100, 265), (1240, 295), (1480, 362),
    (1640, 440), (1820, 458), (1870, 478), (1920, 510), (1920, 1080), (0, 1080), (0, 835),
    (180, 740), (270, 700), (330, 650), (330, 600), (290, 527), (170, 380), (100, 200),
]
MEDIAN = [(100, 118), (130, 112), (1000, 432), (1188, 500), (1190, 516), (1150, 514), (990, 452), (110, 132)]
ISLANDS = [
    [(500, 768), (608, 668), (752, 742), (748, 760)],             # pink triangle, centre
    [(120, 930), (335, 852), (420, 912), (415, 925)],             # pink triangle, bottom-left
    [(680, 860), (700, 830), (870, 810), (955, 875), (740, 895)],  # pink block, bottom-centre
    [(1170, 540), (1300, 540), (1300, 575), (1170, 575)],         # median nose with the keep-right sign
]

MEDIAN_NOSE = (1235, 555)  # centre of the round kerb with the keep-right sign

CROSSWALKS = {
    # north crossing, split by the median nose into the SB and NB halves
    "north_sb": [(333, 607), (750, 560), (1150, 530), (1162, 557), (750, 604), (390, 650)],
    "north_nb": [(1195, 503), (1880, 475), (1905, 498), (1300, 543), (1215, 548)],
    # crossing over the west arm of the cross street
    "west": [(158, 738), (342, 711), (425, 769), (508, 807), (600, 852), (692, 886), (767, 957),
             (858, 1032), (900, 1080), (604, 1080), (567, 1015), (517, 940), (400, 882), (250, 798)],
}

# Junction box between the north crossing, the pink islands and the west
# crossing. Few vehicles drive over parts of it, but it is carriageway and it
# is exactly where people cut across from one crossing to the other.
JUNCTION_BOX = [(400, 655), (1160, 575), (1160, 1080), (600, 1080), (500, 960), (400, 880), (300, 790), (360, 720)]

# Far kerb of the NB carriageway (bus stop, taxis, people waiting at the kerb
# edge). Pedestrians inside this band are not counted as on the carriageway.
NB_KERB = [(95, 45), (450, 95), (700, 185), (900, 240), (1100, 265), (1240, 295), (1480, 362),
           (1640, 440), (1820, 458)]
NB_KERB_BAND_PX = 30

# Southbound stop line, drawn left (kerb) to right (median).
STOP_LINE_SB = [(285, 527), (930, 457)]

# Vehicle signal head on the median nose, facing the camera (it serves the
# approach from the south, which runs in the same phase as the SB approach).
# Lamp centres, top to bottom. A 2-lamp pedestrian head on the left corner
# pole (walking man for the west crossing) is not used for the vehicle phase;
# docs/scene.md gives its lamp positions and the reason.
SIGNAL_LAMPS = {"red": (1157, 371), "yellow": (1157, 388), "green": (1156, 406)}


def poly(points) -> np.ndarray:
    return np.asarray(points, np.float32)


def signed_side(points: np.ndarray, line) -> np.ndarray:
    """>0 on the camera side of a left-to-right line (below it in the image), <0 beyond it."""
    (x1, y1), (x2, y2) = line
    p = np.asarray(points, np.float64).reshape(-1, 2)
    return (x2 - x1) * (p[:, 1] - y1) - (y2 - y1) * (p[:, 0] - x1)


# Apparent height of a standing person in reference pixels, as a linear
# function of position (fitted on ~150k pedestrian boxes from the samples,
# residual 4 px). 1.7 m / this = metres per pixel at that spot.
PERSON_HEIGHT_PX = (25.07, -0.00444, 0.11707)  # c0 + cx * x + cy * y


def metres_per_px(x: float, y: float) -> float:
    c0, cx, cy = PERSON_HEIGHT_PX
    return 1.7 / max(15.0, c0 + cx * x + cy * y)


DRIVABLE = Path(__file__).resolve().parent / "assets" / "drivable.png"


def masks(size=REF_SIZE) -> dict[str, np.ndarray]:
    """Rasterised regions in reference pixels (uint8 0/1).

    ``walk_check`` is where a pedestrian counts as being on the carriageway:
    the hand-drawn road, restricted to where vehicles were actually seen
    driving in the sample clips (``assets/drivable.png``, built by
    tools/drivable_mask.py from vehicle tracks), plus the junction box. That drops the
    kerb strips, the bus-stop edge and the median edge, where people stand
    all day without being in anyone's way.
    """
    w, h = size
    road = np.zeros((h, w), np.uint8)
    cv2.fillPoly(road, [poly(ROAD).astype(np.int32)], 1)
    for hole in [MEDIAN, *ISLANDS]:
        cv2.fillPoly(road, [poly(hole).astype(np.int32)], 0)
    cross = np.zeros((h, w), np.uint8)
    for cw in CROSSWALKS.values():
        cv2.fillPoly(cross, [poly(cw).astype(np.int32)], 1)
    box = np.zeros((h, w), np.uint8)
    cv2.fillPoly(box, [poly(JUNCTION_BOX).astype(np.int32)], 1)
    drivable = cv2.imread(str(DRIVABLE), cv2.IMREAD_GRAYSCALE)
    drivable = (drivable > 127).astype(np.uint8) if drivable is not None else road
    kerb = np.zeros((h, w), np.uint8)
    band = np.asarray(NB_KERB, np.float32)
    band = np.concatenate([band, band[::-1] + [0, NB_KERB_BAND_PX]])
    cv2.fillPoly(kerb, [band.astype(np.int32)], 1)
    walk_check = road & (drivable | box) & (1 - kerb)
    return {"road": road, "crosswalk": cross, "walk_check": walk_check}
