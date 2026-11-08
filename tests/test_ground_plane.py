"""Test mặt sàn: fit tĩnh bằng đường bao dưới + ràng buộc gương, RANSAC tự do, YAML."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gaitmirror.calibration.ground_plane import (  # noqa: E402
    GroundPlane, contact_candidates, fit_ground_plane,
)

THETA = np.deg2rad(30.5)  # góc giữa trục quang học và mặt gương (như phòng thật)


def make_scene(seed=0, pitch_deg=7.5, roll_deg=2.0, noise=0.03, cam_h=1.39,
               n_frames=700, outlier_frac=0.05):
    """Người đi song song gương, camera nhìn dọc gương, chúi + nghiêng."""
    rng = np.random.default_rng(seed)
    s, c = np.sin(THETA), np.cos(THETA)
    R_H = np.array([[-s, 0, c], [0, -1, 0], [c, 0, s]])  # thế giới (X, Y lên, Z) -> ngang
    a, r = np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    R = Rz @ Rx @ R_H
    C_w = np.array([0.0, cam_h, 0.0])
    cyc, stance, marker = 1.4, 0.6, 0.02
    pts = []
    for foot, (z_off, phase0) in enumerate([(-0.09, 0.0), (0.09, 0.5)]):
        for t in range(n_frames):
            pos = (t / 60.0) * 1.2 / cyc + phase0 + 2.0 / cyc
            k, u = np.floor(pos), pos - np.floor(pos)
            x0 = k * cyc
            if u < stance:
                fx, fy = x0, marker
            else:
                sw = (u - stance) / (1 - stance)
                sm = sw * sw * (3 - 2 * sw)
                fx, fy = x0 + cyc * sm, marker + 0.12 * np.sin(np.pi * sw)
            for dx in (-0.07, 0.12, 0.10):  # gót, ngón cái, ngón út
                pts.append([fx + dx, fy, 1.64 + z_off])
    Xw = np.array(pts)
    Xc = (Xw - C_w) @ R.T
    Xc = Xc + rng.normal(0, noise, Xc.shape)
    bad = rng.random(len(Xc)) < outlier_frac
    Xc[bad] += rng.normal(0, 0.15, (bad.sum(), 3))
    return {"pts": Xc, "up": R @ np.array([0.0, 1.0, 0.0]),
            "n_m": R @ np.array([0.0, 0.0, -1.0]), "walk": R @ np.array([1.0, 0, 0]),
            "cam_h": cam_h}


def _rot(v, axis, deg):
    axis = axis / np.linalg.norm(axis)
    a = np.deg2rad(deg)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * (axis @ v) * (1 - np.cos(a))


def _ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def test_envelope_with_mirror_constraint_recovers_up_and_height():
    sc = make_scene()
    # gợi ý sai: lệch 5° dọc hướng đi (quanh n_m) + 3° ngang (quanh hướng đi)
    hint = _rot(_rot(sc["up"], sc["n_m"], 5.0), sc["walk"], 3.0)
    assert _ang(hint, sc["up"]) > 5.0
    gp = fit_ground_plane(sc["pts"], up_hint=hint, mirror_normal=sc["n_m"])
    assert _ang(gp.normal_up, sc["up"]) < 0.8
    assert gp.offset == pytest.approx(sc["cam_h"], abs=0.05)
    assert abs(gp.diagnostics["slope_after_deg"]) < 0.3
    assert abs(gp.diagnostics["slope_before_deg"]) > 2.0  # sàn "trôi dốc" trước khi fit


def test_envelope_without_mirror_fixes_along_track_only():
    sc = make_scene(seed=1)
    hint = _rot(sc["up"], sc["n_m"], 5.0)  # chỉ lệch dọc hướng đi
    gp = fit_ground_plane(sc["pts"], up_hint=hint)
    assert _ang(gp.normal_up, sc["up"]) < 1.0


def test_wrong_hint_drift_visible_as_height_slope():
    """Gợi ý lệch 5° dọc hướng đi làm sàn trôi cỡ L*tan(5°) trên 6 m."""
    sc = make_scene(seed=2, noise=0.005, outlier_frac=0.0)
    hint = _rot(sc["up"], sc["n_m"], 5.0)
    wrong = GroundPlane.from_vertical_down(-hint, sc["cam_h"])
    h = wrong.height(sc["pts"])
    x = sc["pts"] @ sc["walk"]
    drift = np.polyfit(x, h, 1)[0] * (x.max() - x.min())
    assert abs(drift) > 0.3  # > 30 cm trên đoạn đi
    gp = fit_ground_plane(sc["pts"], up_hint=hint, mirror_normal=sc["n_m"])
    good = gp.height(sc["pts"])
    low = good < np.percentile(good, 25)
    assert abs(np.polyfit(x[low], good[low], 1)[0] * (x.max() - x.min())) < 0.05


def test_ransac_free_plane_flags_ill_conditioning_for_straight_walk():
    sc = make_scene(seed=3)
    gp = fit_ground_plane(sc["pts"], up_hint=sc["up"], method="ransac")
    assert gp.diagnostics["ill_conditioned"] == 1.0
    assert _ang(gp.normal_up, sc["up"]) < 12.0  # chỉ để đảm bảo nghiệm hợp lệ


def test_yaml_roundtrip_legacy_keys_and_rescale():
    sc = make_scene(seed=4)
    gp = GroundPlane(sc["up"], 1.39, d_ref=2.4, method="envelope")
    gp2 = GroundPlane.from_config(gp.to_yaml_dict())
    assert np.allclose(gp.normal_up, gp2.normal_up) and gp2.offset == pytest.approx(1.39)
    legacy = {"mirror": {"vertical_down": gp.legacy_keys()["vertical_down"], "d": 2.4},
              "setup": {"camera_height": 1.39}}
    gp3 = GroundPlane.from_config(legacy)
    assert np.allclose(gp3.normal_up, gp.normal_up) and gp3.d_ref == pytest.approx(2.4)
    assert gp.rescaled(2.0).offset == pytest.approx(1.39 * 2.0 / 2.4)
    assert np.allclose(gp.height(np.zeros((1, 3))), 1.39)


def test_contact_candidates_filters_nan():
    pose = np.full((10, 23, 3), np.nan)
    pose[:, 17:23] = 1.0
    assert contact_candidates(pose).shape == (60, 3)