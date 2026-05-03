"""Photogrammetry-first track reconstruction pipeline.

End-to-end shape:
    video + GPX
        → extract.py        : frames + per-frame GPS (OpenSfM project layout)
        → sfm.py            : OpenSfM (Docker) → poses + dense GPS-anchored cloud
        → semantic.py       : SegFormer-Mapillary masks projected to cloud (TODO)
        → cones.py          : YOLO-World cones, back-projected via SfM depth (TODO)
        → export.py         : track.obj (per-class submeshes) + track.xodr (TODO)

Each module is standalone-runnable for incremental verification.
"""
