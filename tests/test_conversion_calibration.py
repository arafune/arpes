import numpy as np
import xarray as xr

from arpes.utilities.conversion.calibration import DetectorCalibration, build_edge_from_list


def test_build_edge_from_list():
    points = [
        {"eV": 1.0, "phi": 0.1},
        {"eV": 2.0, "phi": 0.2},
        {"eV": 3.0, "phi": 0.3},
    ]
    dataset = build_edge_from_list(points)

    assert isinstance(dataset, xr.Dataset)
    assert "eV" in dataset
    assert "phi" in dataset
    assert list(dataset["eV"].values) == [1.0, 2.0, 3.0]
    assert list(dataset["phi"].values) == [0.1, 0.2, 0.3]


def test_detector_calibration_init():
    left_points = [
        {"eV": 1.0, "phi": 0.1},
        {"eV": 2.0, "phi": 0.2},
    ]
    right_points = [
        {"eV": 1.5, "phi": 0.3},
        {"eV": 2.5, "phi": 0.4},
    ]

    calibration = DetectorCalibration(left_points, right_points)

    assert isinstance(calibration._left_edge, xr.Dataset)
    assert isinstance(calibration._right_edge, xr.Dataset)
    assert calibration._left_edge.phi.mean() <= calibration._right_edge.phi.mean()


def test_detector_calibration_repr():
    left_points = [
        {"eV": 1.0, "phi": 0.1},
        {"eV": 2.0, "phi": 0.2},
    ]
    right_points = [
        {"eV": 1.5, "phi": 0.3},
        {"eV": 2.5, "phi": 0.4},
    ]

    calibration = DetectorCalibration(left_points, right_points)
    repr_str = repr(calibration)

    assert "<DetectorCalibration>" in repr_str
    assert "Left Edge" in repr_str
    assert "RightEdge" in repr_str


def test_detector_calibration_correct_detector_angle():
    left_points = [
        {"eV": 1.0, "phi": 0.1},
        {"eV": 2.0, "phi": 0.2},
    ]
    right_points = [
        {"eV": 1.5, "phi": 0.3},
        {"eV": 2.5, "phi": 0.4},
    ]

    calibration = DetectorCalibration(left_points, right_points)

    eV = np.array([1.0, 1.5, 2.0])
    phi = np.array([0.1, 0.2, 0.3])

    corrected_phi = calibration.correct_detector_angle(eV, phi)

    assert corrected_phi.shape == eV.shape
    assert np.all(np.isfinite(corrected_phi))


def test_detector_calibration_init_swap_edges():
    left_points = [
        {"eV": 1.0, "phi": 0.3},
        {"eV": 2.0, "phi": 0.4},
    ]
    right_points = [
        {"eV": 1.5, "phi": 0.1},
        {"eV": 2.5, "phi": 0.2},
    ]

    calibration = DetectorCalibration(left_points, right_points)

    assert calibration._left_edge.phi.mean() <= calibration._right_edge.phi.mean()


def test_detector_calibration_correct_detector_angle_empty():
    left_points = [
        {"eV": 1.0, "phi": 0.1},
        {"eV": 2.0, "phi": 0.2},
    ]
    right_points = [
        {"eV": 1.5, "phi": 0.3},
        {"eV": 2.5, "phi": 0.4},
    ]

    calibration = DetectorCalibration(left_points, right_points)

    eV = np.array([])
    phi = np.array([])

    corrected_phi = calibration.correct_detector_angle(eV, phi)

    assert corrected_phi.shape == eV.shape
    assert corrected_phi.size == 0
