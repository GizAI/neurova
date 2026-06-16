from __future__ import annotations

from langburst.core.block_table import KVBlockTable
from langburst.core.cuda_graph import CudaGraphBucketPlanner
from langburst.core.scheduler import AdmissionController, ContinuousBatchScheduler


def test_admission_controller_tracks_stats():
    scheduler = AdmissionController(max_active_requests=1)
    assert scheduler.stats().active_requests == 0
    with scheduler.acquire():
        stats = scheduler.stats()
        assert stats.active_requests == 1
        assert stats.total_admitted == 1
    stats = scheduler.stats()
    assert stats.active_requests == 0
    assert stats.total_completed == 1


def test_admission_controller_rejects_when_queue_is_full():
    scheduler = AdmissionController(max_active_requests=1, max_queued_requests=0)
    with scheduler.acquire():
        try:
            with scheduler.acquire(timeout_s=0):
                raise AssertionError("second request should not be admitted")
        except TimeoutError as exc:
            assert "queue is full" in str(exc)
    stats = scheduler.stats()
    assert stats.total_admitted == 1
    assert stats.total_completed == 1
    assert stats.total_rejected == 1


def test_continuous_batch_scheduler_chunks_prefill():
    scheduler = ContinuousBatchScheduler(max_num_requests=2, max_num_batched_tokens=4, prefill_chunk_size=3)
    scheduler.add_request("a", [1, 2, 3, 4, 5])
    scheduler.add_request("b", [10, 11])

    batch = scheduler.schedule()

    assert batch is not None
    assert batch.request_ids == ["a", "b"]
    assert batch.input_ids.tolist() == [1, 2, 3, 10]
    assert batch.positions.tolist() == [0, 1, 2, 0]
    assert batch.query_start_loc.tolist() == [0, 3, 4]
    assert batch.num_scheduled_tokens == [3, 1]


def test_continuous_batch_scheduler_prioritizes_decode_rows():
    scheduler = ContinuousBatchScheduler(max_num_requests=2, max_num_batched_tokens=4, prefill_chunk_size=4)
    decode = scheduler.add_request("decode", [1, 2])
    decode.computed_tokens = 2
    decode.last_sampled_token = 3
    decode.draft_token_ids = [4]
    scheduler.add_request("prefill", [10, 11, 12, 13])

    batch = scheduler.schedule()

    assert batch is not None
    assert batch.request_ids == ["decode"]
    assert batch.input_ids.tolist() == [3, 4]
    assert batch.num_draft_tokens_per_request == [1]
    assert batch.cu_num_logits.tolist() == [0, 2]

    decode.computed_tokens += 1
    decode.last_sampled_token = 5
    scheduler.finish_request("decode")
    next_batch = scheduler.schedule()

    assert next_batch is not None
    assert next_batch.request_ids == ["prefill"]
    assert next_batch.input_ids.tolist() == [10, 11, 12, 13]


def test_continuous_batch_scheduler_updates_block_table_and_graph_bucket():
    blocks = KVBlockTable(num_blocks=8, block_size=4)
    graphs = CudaGraphBucketPlanner(batch_sizes=(1, 2, 4), query_lens=(1, 2, 4))
    scheduler = ContinuousBatchScheduler(
        max_num_requests=2,
        max_num_batched_tokens=4,
        prefill_chunk_size=3,
        block_table=blocks,
        cuda_graph_planner=graphs,
    )
    scheduler.add_request("a", [1, 2, 3, 4, 5])
    scheduler.add_request("b", [10])

    batch = scheduler.schedule()

    assert batch is not None
    assert batch.cuda_graph_bucket == (2, 4, 0)
    assert batch.block_tables is not None
    assert batch.slot_mapping is not None
    assert batch.block_tables.tolist() == [[0, 1], [2, 0]]
    assert batch.slot_mapping.tolist() == [0, 1, 2, 8]
    assert blocks.summary()["used_blocks"] == 3
    assert scheduler.finish_request("a") is not None
    assert blocks.summary()["used_blocks"] == 1


def test_kv_block_table_refcounts_cached_prefix_blocks():
    blocks = KVBlockTable(num_blocks=4, block_size=2)
    blocks.ensure_tokens("a", 4)
    assert blocks.summary()["used_blocks"] == 2

    pinned = blocks.pin_prefix_blocks("a", 4)
    assert pinned == (0, 1)
    assert blocks.summary()["pinned_or_shared_blocks"] == 2

    blocks.release("a")
    assert blocks.summary()["used_blocks"] == 2
    blocks.attach_prefix_blocks("b", pinned)
    assert blocks.summary()["used_blocks"] == 2
    blocks.release("b")
    assert blocks.summary()["used_blocks"] == 2
    blocks.release_pinned_blocks(pinned)
    assert blocks.summary()["used_blocks"] == 0


def test_kv_block_table_can_replace_preallocated_blocks_with_prefix_blocks():
    blocks = KVBlockTable(num_blocks=8, block_size=2)
    blocks.ensure_tokens("cached", 4)
    pinned = blocks.pin_prefix_blocks("cached", 4)
    blocks.ensure_tokens("new", 6)

    blocks.reset_to_prefix_blocks("new", pinned)

    assert blocks.get("new").block_ids == [0, 1]
    assert blocks.summary()["used_blocks"] == 2


def test_continuous_batch_scheduler_reuses_input_buffers():
    scheduler = ContinuousBatchScheduler(max_num_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2)
    scheduler.add_request("a", [1, 2])
    scheduler.add_request("b", [3])

    first = scheduler.schedule()
    second = scheduler.schedule()

    assert first is not None and second is not None
    assert first.input_ids.data_ptr() == second.input_ids.data_ptr()
    assert first.positions.data_ptr() == second.positions.data_ptr()
    assert first.query_start_loc.data_ptr() == second.query_start_loc.data_ptr()
