"""Tests for anomaly and hypothesis detectors."""

from nn_toolbox.detectors.collapse import CollapseDetector
from nn_toolbox.detectors.data import DataDetector
from nn_toolbox.detectors.exploding import ExplodingDetector
from nn_toolbox.detectors.instability import InstabilityDetector
from nn_toolbox.detectors.optimization import OptimizationDetector
from nn_toolbox.detectors.saturation import SaturationDetector
from nn_toolbox.detectors.vanishing import VanishingDetector


def test_exploding_detector():
    det = ExplodingDetector()
    context = {
        "forward_analysis": {
            "amplification_events": [
                {"module": "block11", "ratio_to_prev": 18.2, "ratio_to_median": 12.0, "std": 50.0}
            ]
        },
        "backward_analysis": {
            "exploding_params": [
                {"param": "head.weight", "grad_norm": 50000.0, "ratio_to_median": 60.0}
            ]
        },
    }
    findings = det.detect(context)
    assert len(findings) == 2
    assert any(f.module == "block11" for f in findings)
    assert any(f.module == "head.weight" for f in findings)
    assert any("amplification" in f.interpretation for f in findings)


def test_vanishing_detector():
    det = VanishingDetector()
    context = {
        "forward_analysis": {
            "attenuation_events": [
                {"module": "layer2", "ratio_to_prev": 0.01, "ratio_to_median": 0.02, "std": 1e-7}
            ]
        },
        "backward_analysis": {
            "vanishing_params": [
                {"param": "layer1.weight", "grad_norm": 1e-12, "ratio_to_median": 1e-9}
            ],
            "zero_grad_params": ["unhooked.bias"],
        },
    }
    findings = det.detect(context)
    assert len(findings) == 3
    assert any(f.severity == "critical" for f in findings)


def test_saturation_detector():
    det = SaturationDetector()
    context = {
        "activation_stats": {
            "relu1": {"zero_fraction": 0.98},
            "relu2": {"zero_fraction": 0.10},
        }
    }
    findings = det.detect(context)
    assert len(findings) == 1
    assert findings[0].module == "relu1"
    assert "dead units" in findings[0].observation


def test_collapse_detector():
    det = CollapseDetector()
    context = {
        "representation_collapse": {
            "is_collapsed": True,
            "mean_pairwise_similarity": 0.999,
            "effective_rank": 1.05,
            "rank_ratio": 0.01,
        }
    }
    findings = det.detect(context)
    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert "collapse" in findings[0].observation.lower()


def test_instability_and_optimization_detector():
    inst_det = InstabilityDetector()
    opt_det = OptimizationDetector()

    context = {
        "backward_analysis": {
            "recent_cosine_similarity": -0.85,
            "params_with_gradients": 10,
        },
        "update_stats": {
            "global_update_ratio": 0.45,
            "global_update_norm": 2.0,
            "updates": {"layer.weight": {"update_ratio": 0.6}},
        },
        "parameter_info": {
            "total_parameters": 100,
            "trainable_parameters": 0,
        },
    }

    inst_findings = inst_det.detect(context)
    assert len(inst_findings) >= 2  # oscillation and high update ratio

    opt_findings = opt_det.detect(context)
    assert any("frozen" in f.observation for f in opt_findings)


def test_data_detector():
    det = DataDetector()
    context = {
        "data_sanity": {
            "inputs_finite": False,
            "input_is_constant": True,
            "input_min": 0.0,
            "input_max": 255.0,
            "input_shape": [4, 64, 64, 3],  # NHWC shape
            "targets_finite": True,
            "target_is_constant": True,
        }
    }
    findings = det.detect(context)
    assert len(findings) >= 4  # NaN, constant input, constant target, 255 range, NHWC
