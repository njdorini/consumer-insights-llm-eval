"""
Shared helpers used by every script.

The main thing here is `generate()`: give it a provider ("anthropic" or
"openai"), a model id and a prompt, and it returns the model's answer plus
some bookkeeping (tokens used, how long it took, why it stopped).
Keeping this in one place means run_models.py and judge.py don't each need
to know the details of two different APIs.
"""

import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = the folder that contains this "src" folder.
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Read OPENAI_API_KEY / ANTHROPIC_API_KEY from .env into the environment.
# The SDKs pick them up from there automatically, so keys never appear in code.
load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_prompts() -> list[dict]:
    """Load data/prompts.yaml and check it's well-formed."""
    with open(ROOT / "data" / "prompts.yaml") as f:
        prompts = yaml.safe_load(f)

    seen = set()
    for p in prompts:
        for key in ("id", "category", "prompt", "rubric"):
            if key not in p:
                raise ValueError(f"Prompt {p.get('id', '?')} is missing '{key}'")
        # Rubric items can be plain strings; number them c1, c2, ... in order.
        # (Don't reorder criteria after scoring, or the numbers will shift.)
        p["rubric"] = [
            c if isinstance(c, dict) else {"id": f"c{i}", "criterion": c}
            for i, c in enumerate(p["rubric"], 1)
        ]
        if p["id"] in seen:
            raise ValueError(f"Duplicate prompt id: {p['id']}")
        seen.add(p["id"])
        if not 3 <= len(p["rubric"]) <= 5:
            print(f"  note: {p['id']} has {len(p['rubric'])} rubric criteria (plan was 3-5)")
        ids = [c["id"] for c in p["rubric"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate criterion ids in {p['id']}")
    return prompts


# Create each API client once and reuse it.
_clients = {}


def _client(provider: str):
    if provider not in _clients:
        if provider == "anthropic":
            import anthropic
            _clients[provider] = anthropic.Anthropic(max_retries=4)
        elif provider == "openai":
            import openai
            _clients[provider] = openai.OpenAI(max_retries=4)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _clients[provider]


def generate(provider, model_id, prompt, *, system=None, effort=None,
             max_output_tokens=16000, json_schema=None) -> dict:
    """
    Send one prompt to one model and return a dict:
        text, stop_reason, input_tokens, output_tokens, latency_s

    json_schema (Anthropic only): force the reply to be JSON matching this
    schema. The judge uses it so every verdict parses cleanly.
    """
    start = time.time()

    if provider == "anthropic":
        kwargs = dict(
            model=model_id,
            max_tokens=max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        output_config = {}
        if effort:
            output_config["effort"] = effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            kwargs["output_config"] = output_config

        # Streaming avoids HTTP timeouts on long answers; we just wait for the end.
        with _client("anthropic").messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

        # The reply is a list of blocks (thinking, text, ...). Keep only the text.
        text = "".join(b.text for b in msg.content if b.type == "text")
        return dict(
            text=text,
            stop_reason=msg.stop_reason,  # "end_turn" is normal; "refusal" / "max_tokens" are worth noticing
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_s=round(time.time() - start, 1),
        )

    if provider == "openai":
        if json_schema:
            raise NotImplementedError("json_schema is only wired up for the Anthropic judge")
        kwargs = dict(model=model_id, input=prompt, max_output_tokens=max_output_tokens)
        if system:
            kwargs["instructions"] = system
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        resp = _client("openai").responses.create(**kwargs)
        stop = resp.status
        if resp.incomplete_details is not None:
            stop = f"incomplete:{resp.incomplete_details.reason}"
        return dict(
            text=resp.output_text,
            stop_reason=stop,  # "completed" is normal
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=round(time.time() - start, 1),
        )

    raise ValueError(f"Unknown provider: {provider}")
