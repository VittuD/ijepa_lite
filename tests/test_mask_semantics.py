from __future__ import annotations

import torch

from ijepa_lite.masking.base import (
    MaskOutput,
    MaskPartition,
    NWayAssignment,
    TargetScoreAssignment,
    ThreeWayAssignment,
)
from ijepa_lite.masking.metrics import mask_diagnostics


def test_three_way_assignment_exposes_expected_role_counts() -> None:
    probabilities = torch.tensor(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.1, 0.5]],
            [[0.5, 0.2, 0.3], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]],
        ]
    )
    output = MaskOutput(
        partition=MaskPartition(
            context_idx=torch.tensor([[0], [0]]),
            target_idx=torch.tensor([[1], [1]]),
        ),
        assignment=ThreeWayAssignment(
            context=probabilities[..., 0],
            target=probabilities[..., 1],
            ignore=probabilities[..., 2],
        ),
    )

    stats = mask_diagnostics(output, num_patches=3)

    assert stats["mask/expected_nctx"] == probabilities[..., 0].sum(1).mean().item()
    assert stats["mask/expected_ntgt"] == probabilities[..., 1].sum(1).mean().item()
    assert stats["mask/expected_nign"] == probabilities[..., 2].sum(1).mean().item()


def test_nway_partition_uses_real_block_counts_not_padding() -> None:
    probabilities = torch.full((2, 6, 4), 0.25)
    output = MaskOutput(
        partition=MaskPartition(
            context_idx=torch.tensor([[0, 1], [0, 1]]),
            target_idx=torch.tensor(
                [
                    [[2, 2, 2], [3, 4, 5]],
                    [[2, 2, 2], [3, 4, 5]],
                ]
            ),
            target_block_counts=torch.tensor([1, 3]),
        ),
        assignment=NWayAssignment(probabilities=probabilities),
    )

    stats = mask_diagnostics(output, num_patches=6)

    assert stats["mask/ntgt"] == 4.0
    assert stats["mask/hard_tgt_0"] == 1.0
    assert stats["mask/hard_tgt_1"] == 3.0


def test_target_scores_are_not_exposed_as_role_probabilities() -> None:
    scores = torch.tensor([[0.2, 0.8]])
    output = MaskOutput(
        partition=MaskPartition(
            context_idx=torch.tensor([[0]]),
            target_idx=torch.tensor([[1]]),
        ),
        assignment=TargetScoreAssignment(target=scores),
    )

    assert output.context_soft is None
    assert output.target_soft is scores
