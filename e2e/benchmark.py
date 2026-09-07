import argparse
import gc
import pprint
import numpy as np
import torch
import time
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.model_registry import runtime_types
import torch
import transformers

model_configs = [
    # "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-2-13b-hf",
    # "meta-llama/CodeLlama-34b-hf",
    "Qwen/Qwen3-32B",
    # "Qwen/Qwen2.5-32B",
    # "meta-llama/Llama-3.1-8B",
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 3
num_bench_steps = 5
memory_safe_cleanup = False

def repeated_run(num_repeats=3):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

@repeated_run()
def module_benchmark(module):
    # Warm up kernels and libraries. For memory-heavy cases, release only
    # unused allocator blocks between complete forwards.
    for _ in range(num_warmup_steps):
        out = module()
        torch.cuda.synchronize()
        del out
        if memory_safe_cleanup:
            _cleanup()

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    elapsed_ms = []
    peak_memory = torch.cuda.memory_allocated()
    for _ in range(num_bench_steps):
        # Synchronization makes each wall-clock interval GPU-complete. Cleanup
        # happens after the end timestamp and is excluded from reported latency.
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        out = module()
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start_time) * 1000)
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        del out
        if memory_safe_cleanup:
            _cleanup()

    return np.mean(elapsed_ms), peak_memory


def get_model_quantized(config_name):
    config_cls, int4_cls, _ = runtime_types(config_name)
    config = config_cls.from_pretrained(
        config_name, attn_implementation="flash_attention_2")
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights():
        model = int4_cls(config=config)
    torch.set_default_dtype(dtype_old)
    return model


def get_model_hf(config_name):
    return transformers.LlamaForCausalLM.from_pretrained(
        config_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )

def get_model_fp16(config_name):
    _, _, fp16_cls = runtime_types(config_name)
    return fp16_cls.from_pretrained(
        config_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )


def run_prefill(model, bsz, prefill_length):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    return module_benchmark(lambda: model(test_input, use_cache=True))


def run_decode(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    model._expected_max_length = prefill_length + decode_steps
    out = model(test_input)
    past_key_values = out.past_key_values
    del out
    _cleanup()
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _decode_for_multiple_steps():
        past_key_values.length = prefill_length
        for _ in range(decode_steps):
            model(next_input, past_key_values=past_key_values)
    return module_benchmark(_decode_for_multiple_steps)


def run_e2e(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _prefill_and_decode_for_multiple_steps():
        model._expected_max_length = prefill_length + decode_steps
        out = model(test_input)
        for _ in range(decode_steps):
            model(next_input, past_key_values=out.past_key_values)
    return module_benchmark(_prefill_and_decode_for_multiple_steps)


def _wait_for_input():
    print("Press enter")
    input()

@torch.no_grad
def run_all_for_model(model, bsz, prefill, decode):
    model.eval()
    model = model.cuda()
    print(f"[benchmark] prefill start: B={bsz}, S={prefill}", flush=True)
    time_prefill, _ = run_prefill(model, bsz, prefill)
    print(f"[benchmark] prefill complete: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak", flush=True)
    _cleanup()
    if decode is not None:
        print(f"[benchmark] decode start: B={bsz}, cache={prefill}, steps={decode}", flush=True)
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        print(f"[benchmark] decode complete: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak", flush=True)
        _cleanup()
        print(f"[benchmark] e2e start: B={bsz}, S={prefill}, steps={decode}", flush=True)
        time_e2e, _ = run_e2e(model, bsz, prefill, decode)
        print(f"[benchmark] e2e complete: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak", flush=True)
        _cleanup()
    else:
        time_decode = time_e2e = memory_decode = None
    return time_prefill, time_decode, time_e2e, memory_decode

def benchmark(args):
    global memory_safe_cleanup
    memory_safe_cleanup = (
        args.memory_safe_cleanup or
        args.batch_size * args.prefill_seq_len >= 16384
    )
    if memory_safe_cleanup:
        print("Memory-safe cleanup enabled between complete benchmark forwards.",
              flush=True)

    for config_name in model_configs:
        model = get_model_quantized(config_name)
        model.cache_dtype = args.kv_cache_dtype
        time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
            model, args.batch_size, args.prefill_seq_len, args.decode_steps)
        del model
        _cleanup()
        if not args.int4_only:
            model = get_model_fp16(config_name)
            time_prefill_f16, time_decode_f16, time_e2e_f16, mem_f16 = run_all_for_model(
                model, args.batch_size, args.prefill_seq_len, args.decode_steps)
            del model
            _cleanup()
        else:
            time_prefill_f16 = time_decode_f16 = time_e2e_f16 = mem_f16 = None

        print(f'{config_name} & {args.batch_size} & {args.prefill_seq_len}')
        print('---------------------------------------------------------------------')
        print(f"Prefill Int4 time: {np.mean(time_prefill_i4):.3f} +- {1.96 * np.std(time_prefill_i4):.3f}ms")
        if time_prefill_f16 is not None:
            print(f"Prefill FP16 time: {np.mean(time_prefill_f16):.3f} +- {1.96 * np.std(time_prefill_f16):.3f}ms")
            print(f"Speedup: {np.mean(time_prefill_f16) / np.mean(time_prefill_i4):.3f}x")

        if args.decode_steps is not None:
            print('---------------------------------------------------------------------')
            print(f"Decode Int4 time: {np.mean(time_decode_i4):.3f} +- {1.96 * np.std(time_decode_i4):.3f}ms")
            if time_decode_f16 is not None:
                print(f"Decode FP16 time: {np.mean(time_decode_f16):.3f} +- {1.96 * np.std(time_decode_f16):.3f}ms")
                print(f"Speedup: {np.mean(time_decode_f16) / np.mean(time_decode_i4):.3f}x")

            print('---------------------------------------------------------------------')
            print(f"E2E Int4 time: {np.mean(time_e2e_i4):.3f} +- {1.96 * np.std(time_e2e_i4):.3f}ms")
            if time_e2e_f16 is not None:
                print(f"E2E FP16 time: {np.mean(time_e2e_f16):.3f} +- {1.96 * np.std(time_e2e_f16):.3f}ms")
                print(f"Speedup: {np.mean(time_e2e_f16) / np.mean(time_e2e_i4):.3f}x")

        # table-style output
        print('---------------------------------------------------------------------')
        if mem_i4 is not None:
            print(f"Int4 memory: {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_i4):.3f}")
            if mem_f16 is not None:
                print(f"FP16 memory: {np.mean(mem_f16) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_f16):.3f}")
                print(f"Memory saving: {np.mean(mem_f16) / np.mean(mem_i4):.3f}x")

        print('---------------------------------------------------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        required=False,
        default=None,
    )
    parser.add_argument(
        '--kv_cache_dtype', '--kv-cache-dtype', choices=('int4', 'float16'),
        default='int4', help='KV-cache storage for the INT4 model.')
    parser.add_argument(
        '--int4_only', '--int4-only', action='store_true',
        help='Benchmark only INT4 and do not load the FP16 comparison model.',
    )
    parser.add_argument(
        '--memory_safe_cleanup', '--memory-safe-cleanup',
        action='store_true',
        help='Release unused GPU allocator blocks between complete forwards.',
    )

    args = parser.parse_args()
    pprint.pprint(vars(args))
    benchmark(args)
