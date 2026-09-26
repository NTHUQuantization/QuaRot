"""Use the runtime's synchronous PARD2 generator with shared phase timers."""
import inspect

from e2e import synchronous_pard
from e2e.benchmark_batched_ar import BatchedAR
from e2e.benchmark_pard2_phases import UntimedStages


class BatchedPARD(BatchedAR):
    def __init__(self, runtime, prompt, tokens):
        if runtime.mode not in ('pard2-ti', 'pard2-td') or not runtime.ignore_eos:
            raise ValueError('Batched PARD2 requires TI/TD and fixed-length generation')
        self.runtime, self.prompt, self.tokens = runtime, prompt, tokens
        self.expected_ids, self.checks = None, []
        self.generated_source = inspect.getsource(synchronous_pard)

    def begin(self):
        state = synchronous_pard.generate_synchronous(
            self.runtime, self.prompt, self.tokens, UntimedStages())
        assert next(state) == 'prefill_complete'
        return state

    def validate(self, result):
        super().validate(result)
        self.checks[-1].update(draft_forwards=result.draft_forwards,
            proposed_draft_tokens=result.proposed_draft_tokens,
            accepted_draft_tokens=result.accepted_draft_tokens,
            accepted_lengths_by_round=result.accepted_lengths_by_round,
            common_accepted_by_round=result.common_accepted_by_round,
            emitted_tokens_per_sequence_by_round=result.emitted_tokens_per_sequence_by_round)
