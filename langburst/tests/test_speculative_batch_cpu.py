from __future__ import annotations

import torch

from langburst.speculative_batch import (
    DecodeInputBuffers,
    DecodeRequestState,
    apply_decode_post_update,
    build_decode_batch_plan,
    make_speculative_batch_plan,
    resolve_speculative_batch,
    sampled_and_rejected_counts,
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


def test_sampled_and_rejected_counts_match_vllm_shape_contract():
    sampled, rejected = sampled_and_rejected_counts(
        sampled_counts=torch.tensor([0, 2, 3]),
        cu_num_logits=torch.tensor([0, 1, 4, 8]),
        is_prefill=torch.tensor([True, False, False]),
    )

    assert sampled.tolist() == [0, 2, 3]
    assert rejected.tolist() == [0, 1, 1]


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
