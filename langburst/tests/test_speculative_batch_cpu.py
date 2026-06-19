from __future__ import annotations

import torch

from langburst.speculative_batch import (
    DecodeInputBuffers,
    DecodeRequestState,
    NativeSpecDecodeMetadata,
    apply_decode_post_update,
    build_decode_batch_plan,
    resolve_speculative_gpu,
    sampled_and_rejected_counts,
)
from langburst.ops import CPUFallbackOps
from langburst.research.speculative_oracle import (
    make_speculative_batch_plan,
    resolve_greedy_speculative_metadata,
    resolve_speculative_batch,
    tensor_token_ids,
)


def test_speculative_batch_resolves_full_accept_with_bonus_ready_contract():
    first = torch.tensor(10)
    drafts = torch.tensor([11, 12])
    plan = make_speculative_batch_plan(first, drafts)

    result = resolve_speculative_batch(plan, target_token_ids=torch.tensor([11, 12]))

    assert result.accepted_draft_tokens == 2
    assert result.verified_draft_tokens == 2
    assert result.rejected_tokens == 0
    assert result.all_draft_accepted
    assert tensor_token_ids(result.commit_tokens) == [10, 11, 12]
    assert tensor_token_ids(result.emitted_tokens) == [10, 11, 12]
    assert result.num_sampled == 3


def test_speculative_batch_resolves_prefix_reject_with_correct_token():
    first = torch.tensor(10)
    drafts = torch.tensor([11, 99, 100])
    plan = make_speculative_batch_plan(first, drafts, emit_first=False)

    result = resolve_speculative_batch(plan, target_token_ids=torch.tensor([11, 12, 13]))

    assert result.accepted_draft_tokens == 1
    assert result.verified_draft_tokens == 3
    assert result.rejected_tokens == 2
    assert not result.all_draft_accepted
    assert tensor_token_ids(result.commit_tokens) == [10, 11, 12]
    assert tensor_token_ids(result.emitted_tokens) == [11, 12]


def test_build_decode_batch_plan_combines_prefill_and_spec_decode_rows():
    reqs = [
        DecodeRequestState("prefill", 0, [101, 102, 103, 104], computed_tokens=1),
        DecodeRequestState("decode", 1, [201, 202], computed_tokens=2, last_sampled_token=203, draft_token_ids=[204, 205]),
    ]

    batch = build_decode_batch_plan(reqs, max_prefill_tokens_per_request=2)

    assert batch.request_ids == ["prefill", "decode"]
    assert batch.num_scheduled_tokens == [2, 3]
    assert batch.num_draft_tokens_per_request == [0, 2]
    assert batch.input_ids.tolist() == [102, 103, 203, 204, 205]
    assert batch.positions.tolist() == [1, 2, 2, 3, 4]
    assert batch.query_start_loc.tolist() == [0, 2, 5]
    assert batch.seq_lens.tolist() == [3, 5]
    assert batch.logits_indices.tolist() == [1, 2, 3, 4]
    assert batch.cu_num_logits.tolist() == [0, 1, 4]
    assert batch.spec_decode_metadata is not None
    assert batch.spec_decode_metadata.num_draft_tokens == [0, 2]
    assert batch.spec_decode_metadata.draft_token_ids.tolist() == [204, 205]


def test_build_decode_batch_plan_keeps_tensor_drafts_in_metadata():
    req = DecodeRequestState(
        "decode",
        1,
        [201, 202],
        computed_tokens=2,
        last_sampled_token=203,
        draft_token_ids_tensor=torch.tensor([204, 205], dtype=torch.long),
    )
    buffers = DecodeInputBuffers(max_num_requests=1, max_num_tokens=3)

    batch = build_decode_batch_plan([req], buffers=buffers)

    assert req.draft_token_ids is None
    assert req.num_draft_tokens == 2
    assert batch.input_ids.tolist() == [203, 204, 205]
    assert batch.num_draft_tokens_per_request == [2]
    assert batch.spec_decode_metadata is not None
    assert batch.spec_decode_metadata.draft_token_ids.data_ptr() != 0
    assert batch.spec_decode_metadata.draft_token_ids.tolist() == [204, 205]


def test_sampled_and_rejected_counts_match_vllm_shape_contract():
    sampled, rejected = sampled_and_rejected_counts(
        sampled_counts=torch.tensor([0, 2, 3]),
        cu_num_logits=torch.tensor([0, 1, 4, 8]),
        is_prefill=torch.tensor([True, False, False]),
    )

    assert sampled.tolist() == [0, 2, 3]
    assert rejected.tolist() == [0, 1, 1]


def test_native_spec_decode_metadata_matches_vllm_index_contract():
    meta = NativeSpecDecodeMetadata.from_draft_rows([[10, 11, 12], [], [20, 21], [30]])

    assert meta.num_draft_tokens == [3, 0, 2, 1]
    assert meta.draft_token_ids.tolist() == [10, 11, 12, 20, 21, 30]
    assert meta.cu_num_draft_tokens.tolist() == [3, 3, 5, 6]
    assert meta.cu_num_sampled_tokens.tolist() == [4, 5, 8, 10]
    assert meta.logits_indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert meta.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 8]
    assert meta.bonus_logits_indices.tolist() == [3, 4, 7, 9]
    assert meta.max_spec_len == 3
    assert meta.select_rows([0, 2]).num_draft_tokens == [3, 2]
    assert meta.select_rows([0, 2]).draft_token_ids.tolist() == [10, 11, 12, 20, 21]


def test_resolve_greedy_speculative_metadata_is_the_batch_decision_ssot():
    meta = NativeSpecDecodeMetadata.from_draft_rows([[10, 11, 12], [], [20, 21]])

    decisions = resolve_greedy_speculative_metadata(
        meta,
        target_token_ids=torch.tensor([10, 99, 12, 20, 21]),
        bonus_token_ids=torch.tensor([13, 30, 22]),
        scheduled_token_counts=[4, 1, 3],
    )

    assert [decision.token_ids for decision in decisions] == [[10, 99], [30], [20, 21, 22]]
    assert [decision.sampled_count for decision in decisions] == [2, 1, 3]
    assert [decision.rejected_count for decision in decisions] == [2, 0, 0]
    assert [decision.accepted_draft_tokens for decision in decisions] == [1, 0, 2]
    assert [decision.all_drafts_accepted for decision in decisions] == [False, True, True]


def test_resolve_speculative_gpu_contract_matches_reference():
    meta = NativeSpecDecodeMetadata.from_draft_rows([[10, 11, 12], [], [20, 21]])
    reference = resolve_greedy_speculative_metadata(
        meta,
        target_token_ids=torch.tensor([10, 99, 12, 20, 21]),
        bonus_token_ids=torch.tensor([13, 30, 22]),
        scheduled_token_counts=[4, 1, 3],
    )

    resolved = resolve_speculative_gpu(
        meta,
        target_token_ids=torch.tensor([10, 99, 12, 20, 21]),
        bonus_token_ids=torch.tensor([13, 30, 22]),
        scheduled_token_counts=[4, 1, 3],
    )

    assert resolved.output_rows() == [decision.token_ids for decision in reference]
    assert resolved.output_length_list() == [decision.sampled_count for decision in reference]
    assert resolved.rejected_count_list() == [decision.rejected_count for decision in reference]
    assert resolved.accepted_draft_tokens.cpu().tolist() == [
        decision.accepted_draft_tokens for decision in reference
    ]


def test_resolve_speculative_gpu_exposes_commit_and_output_contract():
    meta = NativeSpecDecodeMetadata.from_draft_rows([[10, 11, 12], [], [20, 21]])

    resolved = resolve_speculative_gpu(
        meta,
        target_token_ids=torch.tensor([10, 99, 12, 20, 21]),
        bonus_token_ids=torch.tensor([13, 30, 22]),
        scheduled_token_counts=[4, 1, 3],
    )

    assert resolved.output_rows() == [[10, 99], [30], [20, 21, 22]]
    assert resolved.output_length_list() == [2, 1, 3]
    assert resolved.rejected_count_list() == [2, 0, 0]
    assert resolved.accepted_draft_count_list() == [1, 0, 2]
    assert resolved.commit_tokens.tolist() == [2, 1, 3]
    assert resolved.next_input_ids.tolist() == [99, 30, 22]


def test_copy_selected_trajectory_contract_updates_state_slots():
    trajectory = torch.arange(2 * 3 * 2 * 2, dtype=torch.float16).reshape(2, 3, 2, 2)
    dest = torch.full((4, 2, 2), -1.0, dtype=torch.float16)
    state_indices = torch.tensor([3, 1], dtype=torch.long)
    commit_tokens = torch.tensor([2, 3], dtype=torch.int32)

    CPUFallbackOps.copy_selected_trajectory_out(trajectory, dest, state_indices, commit_tokens)

    assert torch.equal(dest[3], trajectory[0, 1])
    assert torch.equal(dest[1], trajectory[1, 2])
    assert torch.equal(dest[0], torch.full((2, 2), -1.0, dtype=torch.float16))


def test_apply_decode_post_update_uses_query_len_minus_rejected_delta():
    req = DecodeRequestState("decode", 0, [1, 2], computed_tokens=2, last_sampled_token=3, draft_token_ids=[4, 5])
    batch = build_decode_batch_plan([req])

    apply_decode_post_update(
        [req],
        batch=batch,
        sampled_token_ids=[[3, 4]],
        sampled_counts=[2],
        rejected_counts=[1],
    )

    assert req.token_ids == [1, 2, 3, 4]
    assert req.last_sampled_token == 4
    assert req.computed_tokens == 4
    assert req.draft_token_ids is None


def test_generated_tokens_do_not_reopen_prefill_phase():
    req = DecodeRequestState("decode", 0, [1, 2], computed_tokens=0)
    batch = build_decode_batch_plan([req], scheduled_tokens_per_request=[2])
    apply_decode_post_update(
        [req],
        batch=batch,
        sampled_token_ids=[[3]],
        sampled_counts=[1],
        rejected_counts=[0],
    )

    assert req.prefill_len == 2
    assert req.token_ids == [1, 2, 3]
    assert req.computed_tokens == 2
    assert not req.is_prefilling


def test_decode_input_buffers_are_reused_by_batch_plan():
    buffers = DecodeInputBuffers(max_num_requests=2, max_num_tokens=4)
    reqs = [
        DecodeRequestState("a", 0, [1, 2]),
        DecodeRequestState("b", 1, [3]),
    ]

    first = build_decode_batch_plan(reqs, buffers=buffers)
    second = build_decode_batch_plan(reqs, buffers=buffers)

    assert first.input_ids.data_ptr() == buffers.input_ids.data_ptr()
    assert second.input_ids.data_ptr() == buffers.input_ids.data_ptr()
    assert second.query_start_loc.data_ptr() == buffers.query_start_loc.data_ptr()
    assert second.input_ids.tolist() == [1, 2, 3]


def test_decode_batch_plan_exposes_vllm_compatible_aliases():
    row = DecodeRequestState("r1", 7, [1, 2], computed_tokens=2, last_sampled_token=3, draft_token_ids=[4, 5])
    batch = build_decode_batch_plan([row])

    assert batch.num_reqs == 1
    assert batch.idx_mapping.tolist() == [7]
    assert batch.num_draft_tokens == 2
    assert batch.num_logits == 3
    assert batch.row_spans == ((0, 3),)
