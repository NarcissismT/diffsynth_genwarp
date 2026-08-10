from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


_MODULE_PATH = Path(__file__).resolve().parent / "diffsynth" / "utils" / "flow_guidance.py"
_SPEC = importlib.util.spec_from_file_location("flow_guidance_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class FlowGuidanceTest(unittest.TestCase):
    def test_identity_flow_is_identity(self) -> None:
        image = torch.randn(2, 4, 7, 9)
        flow = torch.zeros(1, 2, 14, 18)
        result = _MODULE.backward_warp_latents(image, flow, source_size=(14, 18))
        self.assertTrue(torch.allclose(result, image, atol=1e-6))

    def test_flow_scaling_uses_coordinate_canvas(self) -> None:
        flow = torch.zeros(2, 14, 18)
        flow[0].fill_(2.0)
        flow[1].fill_(-1.0)
        result = _MODULE.resize_backward_flow_to_latents(
            flow, (7, 9), source_size=(14, 18)
        )
        self.assertTrue(torch.allclose(
            result[0, 0], (2.0 * 8.0 / 17.0) * torch.ones(7, 9)
        ))
        self.assertTrue(torch.allclose(
            result[0, 1], (-1.0 * 6.0 / 13.0) * torch.ones(7, 9)
        ))

    def test_single_flow_broadcasts_to_batch(self) -> None:
        image = torch.randn(3, 4, 7, 9)
        flow = torch.zeros(2, 14, 18)
        result = _MODULE.backward_warp_latents(image, flow, source_size=(14, 18))
        self.assertEqual(tuple(result.shape), tuple(image.shape))
        self.assertTrue(torch.allclose(result, image, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
