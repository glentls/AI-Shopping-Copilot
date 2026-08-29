# LLM Message Parser

`LLMMessageParser` is a drop-in replacement for the regex-based `MessageParser`.
It sends the customer message to an OpenAI-compatible LLM endpoint via
**Docker Model Runner** (built into Docker Desktop) and extracts the same
structured output:

- `keywords` — product-relevant BM25 query terms
- `attributes` — structured slots: `category`, `material`, `color`, `size`,
  `style`, `brand`, `budget`, `feature`, `use_case`
- `signals` — `is_override`, `is_no_preference`, `is_vague`

---

## Prerequisites

1. **Python 3.10+** with the `openai` package:
   ```bash
   pip install openai
   ```
   OR

   install all deps via `environment.yml`

2. **Docker Desktop** with Docker Model Runner enabled (see below).

---

## Setup: Docker Model Runner

Docker Model Runner is a feature built into Docker Desktop — no separate
container image is needed.

### 1. Enable Docker Model Runner with TCP access

```bash
docker desktop enable model-runner --tcp=12434
```

This exposes the OpenAI-compatible API on `http://localhost:12434`.

### 2. Pull a model

```bash
docker model pull ai/llama3.1
```

Other available models: `ai/llama3.2`, `ai/smollm2`, `ai/qwen2.5-coder`, etc.
Browse the full list at [hub.docker.com/u/ai](https://hub.docker.com/u/ai).

### 3. Verify the endpoint is reachable

```bash
curl http://localhost:12434/engines/v1/models
```

Expected response:
```json
{"object":"list","data":[{"id":"ai/llama3.1","object":"model",...}]}
```

### 4. List locally available models

```bash
docker model list
```

---

## Configuration

All connection details are read from **environment variables** — never
hard-code credentials.

| Variable | Description | Value for Docker Model Runner |
|---|---|---|
| `DOCKER_MODEL_BASE_URL` | OpenAI-compatible base URL | `http://localhost:12434/engines/v1` |
| `DOCKER_MODEL_API_KEY` | API key (DMR does not require one) | `none` |
| `DOCKER_MODEL_NAME` | Model identifier | `ai/llama3.1` (or whichever you pulled) |

### Set variables in your shell

```bash
export DOCKER_MODEL_BASE_URL="http://localhost:12434/engines/v1"
export DOCKER_MODEL_API_KEY="none"
export DOCKER_MODEL_NAME="ai/llama3.1"
```

Or inline for a single command:

```bash
DOCKER_MODEL_BASE_URL="http://localhost:12434/engines/v1" \
DOCKER_MODEL_API_KEY="none" \
DOCKER_MODEL_NAME="ai/llama3.1" \
python -m src.message_parser.try_it
```

> **Note:** Use `python` (your conda env), not `python3` (system Python).


---

## Running the Interactive REPL

`try_it.py` uses `MessageParser` by default. To test the LLM parser, swap
the two lines in `try_it.py`:

```python
# Replace this:
from .parser import MessageParser
parser = MessageParser(known_categories=categories, known_brands=brands)

# With this:
from .llm_parser import LLMMessageParser
parser = LLMMessageParser(known_categories=categories, known_brands=brands)
```

Then run from the repo root:

```bash
python -m src.message_parser.try_it
```

---

## Output Schema

`ParsedMessage.to_dict()` returns:

```json
{
  "raw_text": "black leather boots, size 9, under $80",
  "keywords": ["black", "leather", "boots", "size", "9", "80"],
  "attributes": {
    "color": "black",
    "material": "leather",
    "size": "9",
    "budget": "80"
  },
  "signals": {
    "is_override": false,
    "is_no_preference": false,
    "is_vague": false
  }
}
```

### Attribute reference

| Key | Type | Description |
|---|---|---|
| `category` | string | Product type, e.g. `"running shoes"`, `"hoodies"` |
| `material` | string | Fabric/material, e.g. `"leather"`, `"cotton"` |
| `color` | string | Color preference, e.g. `"black"`, `"navy blue"` |
| `size` | string | Size value only, e.g. `"9"`, `"M"`, `"XL"` |
| `style` | string | Fit/silhouette/pattern, e.g. `"slim fit"`, `"floral"` |
| `brand` | string | Brand or store name, e.g. `"nike"` |
| `budget` | string | Price ceiling as a number, e.g. `"80"` |
| `feature` | string | Free-text catch-all when no structured slot matched |
| `use_case` | string | Activity/occasion, e.g. `"running"`, `"office"` |

---

## Relation to the Regex-Based `MessageParser`

| Feature | `MessageParser` | `LLMMessageParser` |
|---|---|---|
| Dependency | stdlib only | `openai` package + Docker Model Runner |
| Speed | ~1 ms | Latency of LLM inference (~1–5 s locally) |
| Attribute coverage | Regex vocab lists | Prompt-engineered, generalises better |
| Negation handling | Clause-bounded regex | Instructed via prompt |
| Interface | `.parse(text)` | `.parse(text)` (identical) |
| Output | `ParsedMessage` | `ParsedMessage` (identical) |

Both are exported from `src.message_parser`:

```python
from src.message_parser import MessageParser, LLMMessageParser
```

---

## Troubleshooting

**`RuntimeError: LLMMessageParser requires the following environment variables...`**  
Set the three env vars as shown in the Configuration section above.

**`LLM API call failed: Connection error.`**  
TCP access to Docker Model Runner is not enabled or the port is wrong. Run:
```bash
docker desktop enable model-runner --tcp=12434
curl http://localhost:12434/engines/v1/models   # should return JSON
```

**`DOCKER_MODEL_NAME` not found / model not available**  
The model hasn't been pulled yet. Run `docker model pull ai/llama3.1` (or your
chosen model), then verify with `docker model list`.

**Parser returns `is_vague=True` for everything**  
The model may be returning non-JSON text. Enable debug logging to inspect:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
Smaller models sometimes ignore the JSON-only instruction; try a larger model
like `ai/llama3.1` (8B) for better instruction-following.
