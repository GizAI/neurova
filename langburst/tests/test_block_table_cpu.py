from __future__ import annotations

import torch

from langburst.engines.native.block_table import KVBlockTable


def test_kv_block_table_allocates_logical_positions():
    table = KVBlockTable(num_blocks=4, block_size=8)
    req = table.ensure_tokens("r1", 17)

    assert req.block_ids == [0, 1, 2]
    assert table.summary()["used_blocks"] == 3
    assert req.logical_to_physical(0, block_size=8).block_id == 0
    ref = req.logical_to_physical(16, block_size=8)
    assert ref.block_id == 2
    assert ref.offset == 0


def test_request_block_table_wraps_logical_positions_inside_ring_capacity():
    table = KVBlockTable(num_blocks=2, block_size=4)
    req = table.ensure_tokens("r1", 8)

    assert req.logical_to_physical(8, block_size=4).block_id == 0
    assert req.logical_to_physical(9, block_size=4).offset == 1
    assert req.logical_to_physical(15, block_size=4).block_id == 1


def test_kv_block_table_releases_blocks():
    table = KVBlockTable(num_blocks=2, block_size=4)
    table.ensure_tokens("r1", 8)

    assert table.release("r1") == 2
    assert table.summary()["free_blocks"] == 2
    assert table.release("r1") == 0


def test_kv_block_table_truncates_speculative_tail_blocks():
    table = KVBlockTable(num_blocks=4, block_size=4)
    req = table.ensure_tokens("r1", 9)

    assert req.block_ids == [0, 1, 2]
    assert table.summary()["used_blocks"] == 3

    truncated = table.truncate_tokens("r1", 5)

    assert truncated.block_ids == [0, 1]
    assert table.summary()["used_blocks"] == 2
    assert table.summary()["free_blocks"] == 2


def test_kv_block_table_builds_block_and_slot_tensors():
    table = KVBlockTable(num_blocks=4, block_size=4)
    table.ensure_tokens("a", 6)
    table.ensure_tokens("b", 2)

    blocks = table.block_table_tensor(["a", "b"])
    slots = table.slot_mapping_tensor(
        ["a", "b"],
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        positions=torch.tensor([0, 5, 1], dtype=torch.long),
    )

    assert blocks.tolist() == [[0, 1], [2, 0]]
    assert slots.tolist() == [0, 5, 9]
