from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from docgrid_flow.analysis.mmdit_correspondence import (
    EvaluationContext,
    ManifestSample,
    SourceResizeTransform,
    STRUCTURE_TO_ID,
    evaluate_baselines,
    evaluate_similarity,
    identity_source_token_coordinates,
    make_token_grid,
    pixel_to_token_coordinates,
    sample_gt_at_target_tokens,
    token_to_pixel_coordinates,
)
from docgrid_flow.providers.qwen_diffusers import (
    DiffusersQwenCorrespondenceProbe,
    ImageTokenLayout,
    _normalize_lora_state_dict,
    apply_qwen_rotary,
)


def _canonical_map(target_size: tuple[int, int], source_size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_size
    source_h, source_w = source_size
    y, x = torch.meshgrid(
        torch.arange(target_h, dtype=torch.float32),
        torch.arange(target_w, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack(
        (
            (x + 0.5) * source_w / target_w - 0.5,
            (y + 0.5) * source_h / target_h - 0.5,
        )
    )


def _sample() -> ManifestSample:
    return ManifestSample(
        sample_id="synthetic",
        document_id="doc",
        warped_image=Path("warped.png"),
        rectified_image=Path("rectified.png"),
        backward_map=Path("map.npy"),
        valid_mask=None,
        horizontal_structure=None,
        vertical_structure=None,
        boundary_structure=None,
        input_size=(64, 80),
        output_size=(64, 80),
        warp_severity="hard",
        split="test",
        subset_tags={},
        source_record={},
    )


def test_pixel_token_round_trip_arbitrary_resolution() -> None:
    coordinates = torch.tensor(
        [[-0.5, -0.5], [0.0, 0.0], [127.25, 63.75], [639.5, 479.5]]
    )
    token = pixel_to_token_coordinates(coordinates, (480, 640), (31, 47))
    recovered = token_to_pixel_coordinates(token, (31, 47), (480, 640))
    assert torch.allclose(recovered, coordinates, atol=1.0e-4)


def test_identity_gt_sampling_matches_canonical_token_map() -> None:
    native_target = (73, 91)
    native_source = (61, 107)
    target_grid = (7, 11)
    source_grid = (13, 17)
    work_size = (256, 320)
    backward = _canonical_map(native_target, native_source)
    valid = torch.ones((1, *native_target), dtype=torch.bool)
    transform = SourceResizeTransform.create(native_source, work_size, "stretch")
    pixel, keep, _ = sample_gt_at_target_tokens(
        backward, valid, target_grid=target_grid, source_transform=transform
    )
    token = pixel_to_token_coordinates(pixel, work_size, source_grid)
    expected = identity_source_token_coordinates(target_grid, source_grid)
    assert bool(keep.all())
    assert torch.allclose(token, expected, atol=1.0e-4)


def test_letterbox_transform_round_trip_and_offsets() -> None:
    transform = SourceResizeTransform.create((100, 200), (300, 300), "letterbox")
    assert transform.resized_size == (150, 300)
    assert transform.padding_xy == (0, 75)
    coordinates = torch.tensor([[0.0, 0.0], [199.0, 99.0], [55.5, 42.25]])
    transformed = transform.apply_coords(coordinates)
    assert math.isclose(float(transformed[0, 1]), 75.25, abs_tol=1.0e-6)
    assert torch.allclose(transform.invert_coords(transformed), coordinates, atol=1.0e-5)


def test_letterbox_gt_sampling_matches_explicit_token_lattice() -> None:
    native_target = (73, 91)
    native_source = (100, 200)
    target_grid = (7, 11)
    source_grid = (15, 15)
    work_size = (300, 300)
    backward = _canonical_map(native_target, native_source)
    transform = SourceResizeTransform.create(native_source, work_size, "letterbox")
    pixel, keep, _ = sample_gt_at_target_tokens(
        backward,
        torch.ones((1, *native_target), dtype=torch.bool),
        target_grid=target_grid,
        source_transform=transform,
    )
    token = pixel_to_token_coordinates(pixel, work_size, source_grid)
    target_xy = make_token_grid(target_grid)
    native_x = (target_xy[:, 0] + 0.5) * native_source[1] / target_grid[1] - 0.5
    native_y = (target_xy[:, 1] + 0.5) * native_source[0] / target_grid[0] - 0.5
    expected_pixel = transform.apply_coords(torch.stack((native_x, native_y), dim=1))
    expected_token = pixel_to_token_coordinates(expected_pixel, work_size, source_grid)
    assert bool(keep.all())
    assert torch.allclose(token, expected_token, atol=1.0e-4)


def test_masked_bilinear_renormalizes_fractional_valid_support() -> None:
    backward = torch.zeros((2, 2, 2), dtype=torch.float32)
    backward[:, 0, 0] = torch.tensor([10.0, 20.0])
    backward[:, 0, 1] = torch.tensor([30.0, 40.0])
    backward[:, 1, 0] = torch.tensor([50.0, 60.0])
    backward[:, 1, 1] = float("nan")
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    pixel, keep, _ = sample_gt_at_target_tokens(
        backward,
        valid,
        target_grid=(1, 1),
        source_transform=SourceResizeTransform.create((100, 100), (100, 100)),
    )
    # The one target centre samples all four native pixels equally. The NaN
    # corner has zero support, so the remaining three values are renormalized.
    assert bool(keep.item())
    assert torch.allclose(pixel[0], torch.tensor([30.0, 40.0]), atol=1.0e-5)


def test_xy_channel_order_is_preserved() -> None:
    backward = torch.zeros((2, 4, 5))
    backward[0] = 17.0
    backward[1] = 29.0
    pixel, keep, _ = sample_gt_at_target_tokens(
        backward,
        torch.ones((1, 4, 5), dtype=torch.bool),
        target_grid=(2, 3),
        source_transform=SourceResizeTransform.create((40, 40), (40, 40)),
    )
    assert bool(keep.all())
    assert torch.all(pixel[:, 0] == 17.0)
    assert torch.all(pixel[:, 1] == 29.0)


def test_invalid_and_out_of_source_points_are_filtered() -> None:
    backward = _canonical_map((8, 10), (20, 30))
    backward[:, :4, :5] = float("nan")
    backward[0, 4:, 5:] = 1000.0
    valid = torch.ones((1, 8, 10), dtype=torch.bool)
    valid[:, 3:5, 3:5] = False
    _, keep, _ = sample_gt_at_target_tokens(
        backward,
        valid,
        target_grid=(8, 10),
        source_transform=SourceResizeTransform.create((20, 30), (20, 30)),
    )
    keep_grid = keep.reshape(8, 10)
    assert not bool(keep_grid[:4, :5].any())
    assert bool(keep_grid[:3, 5:].all())
    assert int(keep.sum()) < 40


def _perfect_context() -> tuple[EvaluationContext, torch.Tensor, torch.Tensor]:
    target_grid = (2, 3)
    source_grid = (4, 5)
    source_count = source_grid[0] * source_grid[1]
    gt_indices = torch.tensor([0, 3, 7, 11, 16, 19])
    gt_token = torch.stack(
        (
            (gt_indices % source_grid[1]).float(),
            torch.div(gt_indices, source_grid[1], rounding_mode="floor").float(),
        ),
        dim=1,
    )
    work_size = (80, 100)
    gt_pixel = token_to_pixel_coordinates(gt_token, source_grid, work_size)
    identity = identity_source_token_coordinates(target_grid, source_grid)
    identity_pixel = token_to_pixel_coordinates(identity, source_grid, work_size)
    displacement = torch.linalg.vector_norm(gt_pixel - identity_pixel, dim=-1)
    context = EvaluationContext(
        sample=_sample(),
        target_grid=target_grid,
        source_grid=source_grid,
        source_work_size=work_size,
        gt_source_pixel_xy_all=gt_pixel,
        gt_source_token_xy_all=gt_token,
        valid_all=torch.ones(6, dtype=torch.bool),
        identity_source_token_xy_all=identity,
        displacement_px_all=displacement,
        displacement_ids_all=torch.tensor([0, 1, 2, 3, 2, 1]),
        structure_ids_all=torch.full(
            (6,), STRUCTURE_TO_ID["text_or_horizontal"], dtype=torch.long
        ),
        target_indices=torch.arange(6),
        structure_is_pseudo=False,
    )
    key = torch.eye(source_count)
    query = key[gt_indices]
    return context, query, key


def test_perfect_cost_has_exact_hard_correspondence() -> None:
    context, query, key = _perfect_context()
    rows, artifact = evaluate_similarity(
        query,
        key,
        context,
        temperatures=(0.01, 0.1),
        source_chunk_size=3,
        artifact_query_positions=torch.tensor([0, 5]),
    )
    assert len(rows) == 2
    for row in rows:
        assert row["metrics"]["recall_at_1_r0"] == 1.0
        assert row["metrics"]["hard_epe_mean_px"] == 0.0
        assert row["metrics"]["hard_pck_token_0p5"] == 1.0
    assert artifact is not None
    assert artifact["cost"].shape == (2, 20)


def test_streaming_chunks_do_not_change_metrics() -> None:
    context, query, key = _perfect_context()
    small, _ = evaluate_similarity(
        query, key, context, temperatures=(0.07,), source_chunk_size=2
    )
    full, _ = evaluate_similarity(
        query, key, context, temperatures=(0.07,), source_chunk_size=100
    )
    def assert_close_tree(value: object, other: object, path: str) -> None:
        if isinstance(value, float):
            # Streaming log-sum-exp changes floating-point reduction order but
            # must remain well below a thousandth of a source pixel.  Quantile
            # sketches contain nested floating-point centroid lists, so apply
            # the same bound recursively rather than demanding bit identity.
            assert isinstance(other, float), path
            assert math.isclose(value, other, rel_tol=1.0e-4, abs_tol=1.0e-4), path
        elif isinstance(value, dict):
            assert isinstance(other, dict) and value.keys() == other.keys(), path
            for key, nested in value.items():
                assert_close_tree(nested, other[key], f"{path}.{key}")
        elif isinstance(value, list):
            assert isinstance(other, list) and len(value) == len(other), path
            for index, nested in enumerate(value):
                assert_close_tree(nested, other[index], f"{path}[{index}]")
        else:
            assert value == other, path

    for name, value in small[0]["metrics"].items():
        assert_close_tree(value, full[0]["metrics"].get(name), name)


def test_random_baseline_recall_is_probability_not_sample_noise() -> None:
    context, _, _ = _perfect_context()
    rows = evaluate_baselines(context, random_seed=123)
    random_row = next(row for row in rows if row["baseline"] == "random_candidate")
    assert math.isclose(
        random_row["metrics"]["recall_at_1_r0"], 1.0 / 20.0, abs_tol=1.0e-7
    )
    assert math.isclose(random_row["metrics"]["recall_at_10_r0"], 0.5, abs_tol=1.0e-7)


def test_unequal_target_and_source_token_counts_supported() -> None:
    context, query, key = _perfect_context()
    assert query.shape[0] != key.shape[0]
    rows, _ = evaluate_similarity(query, key, context, temperatures=(0.03,))
    assert rows[0]["metrics"]["valid_tokens"] == query.shape[0]


def test_nonfinite_query_or_key_is_rejected() -> None:
    context, query, key = _perfect_context()
    bad_query = query.clone()
    bad_query[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="query features"):
        evaluate_similarity(bad_query, key, context, temperatures=(0.03,))
    bad_key = key.clone()
    bad_key[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="key features"):
        evaluate_similarity(query, bad_key, context, temperatures=(0.03,))


def test_img_shapes_are_the_only_segment_source() -> None:
    layout = ImageTokenLayout.parse([[(1, 2, 3), (1, 4, 5)]])
    assert layout.target_grid == (2, 3)
    assert layout.source_grid == (4, 5)
    assert layout.offsets == (0, 6, 26)
    values = torch.arange(26).reshape(1, 26, 1, 1)
    assert layout.segment(values, 0).shape[1] == 6
    assert layout.segment(values, 1).shape[1] == 20


def test_rope_replay_identity_frequencies_is_noop() -> None:
    tensor = torch.randn(1, 7, 3, 8, dtype=torch.float32)
    frequencies = torch.ones((7, 4), dtype=torch.complex64)
    rotated = apply_qwen_rotary(tensor, frequencies)
    assert torch.allclose(rotated, tensor, atol=1.0e-6)


def test_rope_replay_rotates_each_token_without_shape_change() -> None:
    tensor = torch.randn(1, 5, 2, 8, dtype=torch.float32)
    phase = torch.linspace(0, math.pi, 5)[:, None].expand(5, 4)
    frequencies = torch.polar(torch.ones_like(phase), phase)
    rotated = apply_qwen_rotary(tensor, frequencies)
    assert rotated.shape == tensor.shape
    assert torch.allclose(rotated[:, 0], tensor[:, 0], atol=1.0e-6)
    assert not torch.allclose(rotated[:, -1], tensor[:, -1])


def test_diffsynth_lora_keys_are_normalized_and_rank_is_inferred() -> None:
    state = {
        "pipe.dit.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(32, 3072),
        "pipe.dit.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(3072, 32),
        "pipe.dit.transformer_blocks.0.img_mod.1.lora_A.default.weight": torch.zeros(
            32, 3072
        ),
        "pipe.dit.transformer_blocks.0.img_mod.1.lora_B.default.weight": torch.zeros(
            18432, 32
        ),
    }
    normalized, rank, targets = _normalize_lora_state_dict(state)
    assert rank == 32
    assert targets == ("img_mod.1", "to_q")
    assert all(key.startswith("transformer_blocks.0.") for key in normalized)
    assert all(".default.weight" in key for key in normalized)


def test_non_lora_tensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="pure PEFT LoRA"):
        _normalize_lora_state_dict({"transformer_blocks.0.weight": torch.zeros(1)})


class _FakePosEmbed(nn.Module):
    def forward(self, img_shapes, max_txt_seq_len, device=None):
        del max_txt_seq_len
        layout = ImageTokenLayout.parse(img_shapes)
        image = torch.ones((layout.total_tokens, 2), dtype=torch.complex64, device=device)
        text = torch.ones((1, 2), dtype=torch.complex64, device=device)
        return image, text


class _FakeAttention:
    def __init__(self) -> None:
        self.heads = 2
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _FakeAttention()


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_FakeBlock(), _FakeBlock()])
        self.pos_embed = _FakePosEmbed()
        self.inner_dim = 8
        self.config = {"fake": True}
        self.dtype = torch.float32

    def forward(self, hidden_states, timestep, img_shapes, **_kwargs):
        del hidden_states, timestep
        layout = ImageTokenLayout.parse(img_shapes)
        self.pos_embed(img_shapes, max_txt_seq_len=1, device=torch.device("cpu"))
        value = torch.arange(layout.total_tokens * 8, dtype=torch.float32).reshape(
            1, layout.total_tokens, 2, 4
        )
        for block in self.transformer_blocks:
            block.attn.norm_q(value)
            block.attn.norm_k(value + 1)
        return (torch.zeros(1, layout.total_tokens, 1),)


class _FakePipeline:
    _callback_tensor_inputs = ["latents"]

    def __init__(self) -> None:
        self.transformer = _FakeTransformer()
        self.scheduler = SimpleNamespace(
            sigmas=torch.tensor([1.0, 0.5, 0.1, 0.0]), config={"name": "fake"}
        )
        self._execution_device = torch.device("cpu")
        self._current_timestep = None

    def set_progress_bar_config(self, disable=True):
        del disable

    def to(self, device):
        del device
        return self

    def __call__(
        self,
        image,
        prompt,
        height,
        width,
        generator,
        num_inference_steps,
        true_cfg_scale,
        num_images_per_prompt,
        output_type,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        guidance_scale=None,
        negative_prompt=None,
        max_sequence_length=None,
    ):
        del (
            image,
            prompt,
            height,
            width,
            generator,
            true_cfg_scale,
            num_images_per_prompt,
            output_type,
            callback_on_step_end_tensor_inputs,
            guidance_scale,
            negative_prompt,
            max_sequence_length,
        )
        img_shapes = [[(1, 2, 3), (1, 4, 5)]]
        for index in range(num_inference_steps):
            timestep = torch.tensor(float(1000 - index * 100))
            self._current_timestep = timestep
            self.transformer(
                hidden_states=torch.zeros(1, 26, 1),
                timestep=(timestep / 1000).reshape(1),
                img_shapes=img_shapes,
            )
            callback_on_step_end(
                self, index, timestep, {"latents": torch.zeros(1)}
            )
        self._current_timestep = None
        return SimpleNamespace(images=torch.zeros(1, 1))


class _FeatureCollector:
    capture_current_query = True

    def __init__(self) -> None:
        self.features = []
        self.layouts = []
        self.trace = []

    def on_layout(self, layout):
        self.layouts.append(layout)

    def on_feature(self, metadata, query_target, key_source):
        self.features.append((metadata, query_target.clone(), key_source.clone()))

    def on_step_end(self, trace):
        self.trace.append(dict(trace))


def test_online_probe_tracks_steps_segments_and_pre_post_rope(monkeypatch) -> None:
    fake = _FakePipeline()
    monkeypatch.setattr(
        DiffusersQwenCorrespondenceProbe, "_load_pipeline", lambda _self: fake
    )
    config = {
        "model_id": "Qwen/Qwen-Image-Edit-2511",
        "dtype": "float32",
        "prompt": "rectify",
        "height": 32,
        "width": 32,
        "num_inference_steps": 3,
        "true_cfg_scale": 1.0,
        "guidance_scale": None,
    }
    collector = _FeatureCollector()
    with DiffusersQwenCorrespondenceProbe(
        config,
        device=torch.device("cpu"),
        selections={1: {0: ("pre", "post")}},
    ) as probe:
        probe.run_sample(Image.new("RGB", (32, 32)), seed=0, consumer=collector)
    assert len(collector.trace) == 3
    assert len(collector.layouts) == 1
    assert [feature[0].rope_state for feature in collector.features] == ["pre", "post"]
    for metadata, query, key in collector.features:
        assert metadata.step_index == 1
        assert metadata.branch == "conditional"
        assert query.shape == (6, 8)
        assert key.shape == (20, 8)
    assert torch.allclose(collector.features[0][1], collector.features[1][1])
    assert torch.allclose(collector.features[0][2], collector.features[1][2])
