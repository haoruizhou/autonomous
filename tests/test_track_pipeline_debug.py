import json
import math

import cv2
import numpy as np

from track_pipeline import (
    _cone_mask_hsv_debug,
    _cone_projection_records,
    _cone_boxes_to_centers,
    _gps_offset_for_frame,
    _select_debug_sample_frames,
    _write_debug_html_report,
)


def test_select_debug_sample_frames_is_deterministic_and_sorted():
    frames = list(range(0, 100, 5))

    selected = _select_debug_sample_frames(frames, sample_count=5, seed=7)

    assert selected == sorted(selected)
    assert len(selected) == 5
    assert selected == _select_debug_sample_frames(frames, sample_count=5, seed=7)
    assert all(frame in frames for frame in selected)


def test_select_debug_sample_frames_returns_all_when_sample_exceeds_available():
    frames = [30, 10, 20]

    selected = _select_debug_sample_frames(frames, sample_count=10, seed=7)

    assert selected == [10, 20, 30]


def test_cone_mask_hsv_debug_reports_raw_and_clean_components():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 55), (70, 95), (0, 120, 255), -1)
    cv2.rectangle(frame, (5, 5), (7, 7), (0, 120, 255), -1)

    debug = _cone_mask_hsv_debug(frame)

    assert debug["raw_mask"].shape == (120, 160)
    assert debug["clean_mask"].shape == (120, 160)
    assert debug["raw_count"] >= 2
    assert debug["clean_count"] == 1
    assert len(debug["components"]) == 1
    component = debug["components"][0]
    assert 50 <= component["cx"] <= 60
    assert 70 <= component["cy"] <= 80
    assert component["area"] >= 1000


def test_cone_boxes_to_centers_uses_bottom_center_for_depth_projection():
    boxes = [
        {"xyxy": [10.0, 20.0, 50.0, 100.0], "conf": 0.91, "cls": 0, "label": "traffic cone"},
        {"xyxy": [70.0, 30.0, 120.0, 90.0], "conf": 0.73, "cls": 0, "label": "traffic cone"},
    ]

    centers = _cone_boxes_to_centers(boxes)

    assert centers == [
        {"cx": 30, "cy": 100, "area": 3200, "conf": 0.91, "bbox": [10, 20, 40, 80], "source": "model", "label": "traffic cone"},
        {"cx": 95, "cy": 90, "area": 3000, "conf": 0.73, "bbox": [70, 30, 50, 60], "source": "model", "label": "traffic cone"},
    ]


def test_cone_boxes_to_centers_discards_low_confidence_boxes():
    boxes = [
        {"xyxy": [10.0, 20.0, 50.0, 100.0], "conf": 0.19, "cls": 0, "label": "traffic cone"},
        {"xyxy": [70.0, 30.0, 120.0, 90.0], "conf": 0.51, "cls": 0, "label": "traffic cone"},
    ]

    centers = _cone_boxes_to_centers(boxes, min_conf=0.5)

    assert len(centers) == 1
    assert centers[0]["cx"] == 95
    assert centers[0]["cy"] == 90


def test_gps_offset_for_frame_uses_first_sample_as_origin():
    gps_list = [
        {"frame": 0, "lat": 42.0, "lon": -71.0},
        {"frame": 10, "lat": 42.0001, "lon": -70.9999},
    ]

    offset = _gps_offset_for_frame(gps_list, 10)

    expected_x = math.radians(0.0001) * math.cos(math.radians(42.0)) * 6_371_000.0
    expected_z = math.radians(0.0001) * 6_371_000.0
    np.testing.assert_allclose(offset, np.array([expected_x, 0.0, expected_z], dtype=np.float32), rtol=1e-6)


def test_cone_projection_records_include_world_coordinates_and_depth():
    depth = np.full((100, 200), 0.5, dtype=np.float32)
    cone_centers = [{"cx": 100, "cy": 50, "area": 250}, {"cx": 150, "cy": 75, "area": 300}]
    offset = np.array([10.0, 0.0, 20.0], dtype=np.float32)

    records = _cone_projection_records(
        frame_name="NH1_f000010_g000002",
        video_stem="NH1",
        src_frame=10,
        cone_centers_px=cone_centers,
        depth_arr=depth,
        gps_offset=offset,
    )

    assert len(records) == 2
    assert records[0]["frame"] == "NH1_f000010_g000002"
    assert records[0]["video"] == "NH1"
    assert records[0]["src_frame"] == 10
    assert records[0]["depth_rel"] == 0.5
    assert records[0]["camera_xyz"] == [0.0, 0.0, 1.999996]
    assert records[0]["world_xyz"] == [10.0, 0.0, 21.999996]
    assert records[1]["world_xyz"][0] > 10.0
    assert records[1]["world_xyz"][2] > 20.0


def test_write_debug_html_report_links_frames_and_world_summary(tmp_path):
    debug_dir = tmp_path / "debug_vis"
    frames_dir = debug_dir / "frames"
    world_dir = debug_dir / "world"
    frames_dir.mkdir(parents=True)
    world_dir.mkdir(parents=True)
    (frames_dir / "NH1_f000000_g000000_debug.png").write_bytes(b"frame")
    (world_dir / "NH1_f000000_g000000_world.png").write_bytes(b"world")
    (world_dir / "cone_world_summary.png").write_bytes(b"summary")
    (debug_dir / "cone_projection_records.json").write_text(json.dumps([{"frame": "NH1_f000000_g000000"}]))

    html_path = _write_debug_html_report(debug_dir, title="Track Debug")

    html = html_path.read_text()
    assert html_path == debug_dir / "index.html"
    assert "Track Debug" in html
    assert "NH1_f000000_g000000_debug.png" in html
    assert "NH1_f000000_g000000_world.png" in html
    assert "cone_world_summary.png" in html
    assert "cone_projection_records.json" in html
