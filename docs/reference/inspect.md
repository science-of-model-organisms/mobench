# Inspect AI and the eval layer

The eval layer is built on [Inspect AI](https://inspect.aisi.org.uk/) (docs index for
agents: <https://inspect.aisi.org.uk/llms.txt>). Check the installed `inspect_ai` version
before relying on a docs page; the published docs can run ahead of the pin.

## Why Inspect

- Every registered eval is a native Inspect task: a real `@solver` and `@scorer`, scheduled
  by `eval_set` over the organism x eval matrix, with results read back from `.eval` logs.
- Anyone who knows Inspect can use the suite directly: `inspect eval mo_evals/<name>` after
  install, `inspect view` for transcripts and scores, `--model` to bring any Inspect
  provider, `--model-role grader=` to swap the judge.
- Retries, concurrency, caching, epochs, `--limit`/`--sample-id`, multiple-choice and
  model-graded scorers, log tooling and `inspect score` come from upstream. Custom code is
  written only where the research differs (per-sample Parquet, our axes, the
  organism-as-model provider), with the reason in a comment.

```bash
uv run inspect eval mo_evals/mmlu -T persona=impulsive -T method=oct_demos   # one cell
uv run inspect eval mo_evals/mmlu --model openai/gpt-4o --limit 5            # your own model
```

## The seam

- **One provider.** `@modelapi("mo_organism")` (`EndpointModelAPI`, `evals/_inspect.py`)
  wraps any `Endpoint` as an Inspect model. Evals never see how the organism was built
  (CONTRIBUTING.md, Non-negotiables and Contracts).
- **Registration is the task.** One `register_task` call per eval (CONTRIBUTING.md, Adding an eval); the
  entry-point group `inspect_ai` exposes the registry as `mo_evals/<name>`. `core/registry.py`
  keeps only `VALID_AXES` and `EvalSkipped`.
- **Logs to Parquet.** One bridge turns `.eval` logs into per-sample `SampleResult` Parquet
  (`raw_output` always kept). Both runners stamp `bridged`, and the collision guard is
  closure-aware.
- **Bring your own model.** `--model` runs an eval in external mode
  (`spec_hash="external"`); combining an organism with `--model` raises instead of silently
  ignoring one.
- **Judge as a model role.** Judges read the `grader` role, so `--model-role grader=...`
  and `inspect score <log> --model-role grader=<model>` re-judge frozen samples per
  candidate judge (see `judges/calibration.py`).
- **Resume is config-safe.** A task's `version` hashes the eval config and rendered
  dataset, so a changed config re-runs instead of resuming stale logs.
- **Metrics.** Every scorer reports `mean`/`stderr`, so `inspect view` shows real scores.
- **Scheduling.** Inspect dispatches model-balanced, so with one model per organism each
  organism runs its cells in list order. `_SCHEDULE_FIRST` in `serving/inspect_runner.py`
  fronts the known long cells so they start while short cells fill the other slots;
  override with `schedule_first:` in the sweep config.
- **Fake mode.** `FakeEndpoint` still drives the whole pipeline with no GPU. Fake `.eval`
  logs are pruned once bridged.

## Endpoint contract extension

`core/` is a frozen contract. It was extended once, additively, to let the Inspect
provider report real telemetry instead of estimates.

**Why.** `Endpoint.complete` returns a bare `str`, so the provider had nowhere to get token
counts or a finish reason and had to estimate from `len(text) // 4`. Four things need real
values:

- Honest usage in logs and in `results/_usage/` (cost per organism, eval and sweep).
- `--token-limit`, which Inspect enforces against reported usage.
- Truncation visibility: a judged response cut off by the output cap scores the cap, not
  the organism. The chars/4 estimate under-reports and misses truncations; a real finish
  reason does not.
- Inspect's stock `perplexity()` and `target_perplexity()` scorers, which read
  `state.output.choices[0].prompt_logprobs`, teacher-forced over the prompt.
  `Endpoint.logprobs` covers the completion, so there was no route and they scored NaN.

**What changed.** One dataclass and one optional method in `core/endpoint.py`:

```python
@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None = None      # None means NOT MEASURED, never zero
    output_tokens: int | None = None
    stop_reason: str | None = None       # "stop" | "max_tokens" | "unknown"
    prompt_logprobs: list[float] | None = None   # only when asked for

class Endpoint(Protocol):
    def complete_full(self, messages, **kw) -> Completion: ...   # may raise NotImplementedError
```

`complete`, `logprobs` and `hidden_states` are unchanged, so existing implementations and
evals work as before. `text` must equal what `complete` returns for the same call and seed.
The provider prefers `complete_full` by presence (`getattr(endpoint, "complete_full", None)`),
never by a trial call. An endpoint without it falls back to the estimate path, still marked
`usage_estimated` / `stop_reason_estimated`. An endpoint that has it and raises is a
configuration error and propagates. Wrappers in `serving/resolve.py` forward it with the
bound prompt or ICL prefix applied, so counts include what the method actually costs.
`FakeEndpoint` implements it (word counts, `stop_reason: "stop"`, prompt logprobs on
request), so the fake gate takes the measured branch.

**Why it does not break the seam.** The rule is that nothing method-specific goes on
`Endpoint`:

- Token counts, a finish reason and prompt logprobs are facts about one generation. Prompting,
  a LoRA, a steering hook and edited weights all report them in the same shape.
- It is read-only: a measurement, not a control. Method knobs stay in `method_config`.
- Its consumer is the serving layer, not an eval. No eval calls `complete_full`.
- It is additive and optional.

## Tool calling for the organism

For agentic evals where the organism is the agent, `EndpointModelAPI` forwards tool schemas
as `complete(..., tools=[...])` (HF/OpenAI shape). `core/` only documents this; `**kw`
already admitted the argument.

- `serving/vllm_backend.py` routes `tools` into `apply_chat_template(tools=...)`. Other
  backends refuse loudly rather than let an organism answer in prose with the tools
  silently missing.
- The call returns in the completion text (`<tool_call>{...}</tool_call>` for Qwen, a bare
  JSON object for Llama 3.1) and is parsed by Inspect's own `HFHandler`.
- `evals/_inspect._endpoint_message` keeps `tool_calls` on assistant turns and `name` /
  `tool_call_id` on tool turns, so an agent sees that it already called a tool.
- `tool_choice` cannot be enforced on a raw-completion organism; a forced choice is warned
  about.
- `FakeEndpoint` never emits tool syntax, so an agentic cell needs its own scripted
  `mockllm` for fake mode (the shared one, `evals/_agent_models`, was deleted
  with `petri_audit`; no registered eval uses one). Keep the `[FAKE-ORGANISM` banner
  in its output so `analysis` refuses to report it.

## Watching a running sweep

Launch a sweep under a named terminal-multiplexer session teed to a logfile. `inspect eval --detach` is not
used: it only wraps `inspect eval`/`eval-set`/`eval-retry`, while a sweep is `run_all.sh`
into `serving.inspect_runner`, several `eval_set()` calls, a report render and an HF push.
The runner passes no `ctl_server=`, so each `eval_set()` binds a control socket by default,
and `inspect ctl task --json`, `sample errors`, `sample cancel` and `config --max-tasks`
work against a running sweep from a second shell. Do not use `--ctl-server=keep` on a remote machine:
it parks a process after the eval finishes. The completion signal is the Parquet tree.
