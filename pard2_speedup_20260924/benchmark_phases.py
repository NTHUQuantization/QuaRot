"""Add phase measurements using the existing runtime's HIP events.

No extra event, synchronization, graph, or generation operation is introduced.
Prefill is an event span; steady decode and E2E retain the official wall timers.
"""
import hashlib
import json
from pathlib import Path

from e2e import speculative
from e2e.benchmark_pard2 import main, parser


class PhaseTimer(speculative._StageTimer):
    def totals(self):
        result = super().totals()
        if not self.events:
            return result
        if self.events[0][0] == "target":
            # AR uses the same stage name for prefill and subsequent M1 decode.
            prefill = self.events[:1]
        else:
            names = {"target_prefill", "target_feature_prefill", "draft_prefill",
                     "td_feature_project_prefill"}
            prefill = [entry for entry in self.events if entry[0] in names]
        if not prefill:
            raise RuntimeError("cannot identify prefill in existing stage events")
        result["phase_prefill_gpu_span"] = prefill[0][1].elapsed_time(prefill[-1][2])
        result["phase_target_prefill_gpu"] = prefill[0][1].elapsed_time(prefill[0][2])
        result["phase_draft_prefill_gpu"] = sum(
            start.elapsed_time(end) for name, start, end in prefill
            if name == "draft_prefill")
        return result


if __name__ == "__main__":
    args = parser().parse_args()
    speculative._StageTimer = PhaseTimer
    main()
    path = Path(args.output)
    payload = json.loads(path.read_text())
    payload["phase_measurement"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prefill": "HIP event span from target prefill start through last prefill end; "
                   "PARD includes draft prefix prefill and intervening work; "
                   "TD includes target-feature materialization. No proposal/verification included.",
        "steady_decode": "Official wall-time throughput excluding all tokens/time of first emitted round.",
        "e2e": "Official generated tokens / generate wall time; includes prefill and first round, "
               "excludes loading/tokenization/cache construction before runtime timer and warmed graph capture.",
        "instrumentation": "Derive spans from existing HIP events after the original synchronization; "
                           "no extra GPU events or synchronization. Adds only post-run CPU scalar processing.",
    }
    for row in payload["runs"]:
        row["prefill_gpu_span_ms"] = row["stage_ms"]["phase_prefill_gpu_span"]
        row["prefill_input_tokens_per_s"] = (
            1000 * row["input_token_count"] / row["prefill_gpu_span_ms"])
    path.write_text(json.dumps(payload, indent=2) + "\n")
