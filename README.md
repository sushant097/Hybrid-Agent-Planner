# Hybrid-Agent-Planner
A fully agentic Hybrid Planning system using LLM reasoning + heuristic guardrails + adaptive Python plans + memory-aware decision making. Includes conservative strategy execution, tool sandboxing, historical conversation indexing, and introspection-driven multi-step planning.

## 🧠 System Architecture

#### 1. High level

This project is a **Hybrid Planning Agent**:

> User query → Perception (LLM) → Tool selection (MultiMCP) → Decision (LLM plan) → Python sandbox → MCP tools → Memory → Final answer

The agent runs in **discrete steps**. At each step it:

1. Understands the query and picks tool servers (**Perception**).
2. Chooses which tool to call and writes Python glue code (**Decision / Planning**).
3. Executes that Python code in a sandbox that can only talk to tools (**Action**).
4. Updates its **Memory** and either:

   * finishes with `FINAL_ANSWER: ...`, or
   * continues with `FURTHER_PROCESSING_REQUIRED: ...` and another step.

#### 2. Architecture diagram (Mermaid)

```mermaid
flowchart TD
    User --> Main

    Main --> Config
    Main --> Context
    Main --> MultiMCP
    Main --> Loop

    Context --> Memory

    subgraph AgentStep
        Loop --> Perception
        Perception --> MultiMCP
        MultiMCP --> Loop

        Loop --> Decision
        Decision --> Sandbox

        Sandbox --> MultiMCP
        Sandbox --> Loop
        Loop --> Memory
    end

    Loop --> Main
    Main --> User

```

### 🔍 What each node stands for 

| Node           | Meaning                                                     |
| -------------- | ----------------------------------------------------------- |
| **User**       | CLI user typing a query                                     |
| **Main**       | `agent.py`                                                  |
| **Config**     | `models.json` + `profiles.yaml`                             |
| **Context**    | `AgentContext` (state, strategy, memory linkage)            |
| **MultiMCP**   | Tool dispatcher loading all MCP servers                     |
| **Loop**       | `AgentLoop.run()` (perception → decision → action → memory) |
| **Perception** | `modules/perception.py` LLM intent + server selection       |
| **Decision**   | `modules/decision.py` LLM writes Python solve() code        |
| **Sandbox**    | `modules/action.py` executes plan safely                    |
| **Memory**     | `modules/memory.py` logging, success history                |

---

#### 3. Components

* **`agent.py`**
  CLI entry point.

  * Loads `profiles.yaml` and `models.json`.
  * Creates a shared `MultiMCP` dispatcher and calls `initialize()` to discover tools from all MCP servers.
  * For each user query:

    * Builds an `AgentContext` (user input, session id, strategy profile, dispatcher, memory manager).
    * Creates an `AgentLoop(context)` and awaits `run()`.
    * Handles:

      * `FINAL_ANSWER: ...` → print and finish.
      * `FURTHER_PROCESSING_REQUIRED: ...` → feed that back in as the next `user_input` and keep going.

* **`core/context.py`**
  Holds **all state** for a run:

  * `AgentProfile` and `StrategyProfile` are loaded from `profiles.yaml` (planning mode, max steps, lifelines, memory flags).
  * `AgentContext` tracks:

    * `user_input`, `session_id`, `step`
    * `agent_profile` (strategy and persona)
    * `memory = MemoryManager(session_id=...)`
    * `dispatcher` (the shared `MultiMCP`)
    * `mcp_server_descriptions` (for perception prompt)
    * `task_progress` (simple introspection: per step tool + status)
    * `final_answer`
  * Provides helpers like `add_memory`, `log_subtask`, and `update_subtask_status`.

* **`core/session.py`** – MCP dispatch layer

  * `MCP` wraps a single FastMCP server, using `ClientSession` over stdio to:

    * `list_tools()`
    * `call_tool(name, arguments)`
  * `MultiMCP` aggregates all servers from `profiles.yaml`:

    * At `initialize()` time it starts each server once, calls `list_tools()`, and builds:

      * `tool_map[tool_name] = {config, tool}`
      * `server_tools[server_id] = [tools...]`
    * At runtime:

      * `call_tool(tool_name, arguments)` runs the specific server script via stdio and calls that tool.
      * `get_tools_from_servers(selected_servers)` returns tool objects that perception decided to use.

* **`modules/perception.py`** – Perception

  * Builds a prompt from `perception_prompt.txt` with the list of available MCP servers and the user query.
  * Uses `ModelManager` (Gemini or Ollama) to generate a JSON block with:

    * `intent`, `entities`, `tool_hint`, `tags`, and `selected_servers` (list of server ids).
  * `run_perception(...)` returns a `PerceptionResult` that drives tool selection.

* **`modules/decision.py`** – Decision and plan generation

  * Reads the appropriate decision prompt (conservative or exploratory) using `load_prompt`.
  * Injects `tool_descriptions` and `user_input` into that template.
  * Uses `ModelManager` to generate **Python code** that defines `solve()`.
  * Cleans fenced blocks and checks that a `def solve(...):` actually exists.
  * Returns the raw code string (the “plan”).

* **`modules/action.py`** – Python sandbox

  * Defines `run_python_sandbox(plan, dispatcher)` which:

    * Wraps the real `MultiMCP` in a `SandboxMCP` that:

      * exposes `await mcp.call_tool(tool_name, input_dict)`
      * enforces a max number of tool calls per plan.
    * Creates a fresh `ModuleType` and injects:

      * `mcp = SandboxMCP(dispatcher)`, `json`, `re`.
    * `exec`s the generated code to load `solve()` and then runs it.
    * Normalises return values to a string:

      * `dict` → `result["result"]` or JSON string
      * `list` → `" ".join(list)`
      * everything else → `str(value)`
    * Captures exceptions and returns `[sandbox error: ...]`.

* **`modules/memory.py` + `core/context.py`** – Memory

  * `MemoryManager` writes JSON files under `memory/<year>/<month>/<day>/session-<session_id>.json`.
  * Each `MemoryItem` has a type:

    * `run_metadata`, `tool_call`, `tool_output`, `final_answer`
  * The agent stores:

    * tool calls and outputs (plus success flag)
    * final answers
    * simple run metadata.
  * This will later power your historical index and memory MCP server.

* **`core/loop.py`** – AgentLoop controller

  * Uses `max_steps` and `max_lifelines_per_step` from the strategy profile.
  * For each step:

    1. **Perception**

       * `run_perception(context, user_input_override or user_input)`
       * Use `selected_servers` to fetch tools via `MultiMCP`.
    2. **Planning**

       * `summarize_tools(selected_tools)`
       * `select_decision_prompt_path(...)`
       * `generate_plan(...)` to get Python code for `solve()`.
    3. **Execution**

       * If plan defines `solve()`, run it via `run_python_sandbox(...)`.
       * Interpret result:

         * `"FINAL_ANSWER:"` → store in memory, return.
         * `"FURTHER_PROCESSING_REQUIRED:"` → build a new meta `user_input_override` instructing the next step what to do.
         * `"[sandbox error:"` or invalid plan → mark failure and use another lifeline.
  * When steps or lifelines are exhausted, returns
    `FINAL_ANSWER: [Max steps reached]`.

---

# 2. Bug Fixed Report: [Bug-Fixed-Report.md](Bug-Fixed-Report.md)

---

# 3. 10 Heuristics for Effective Hybrid Agent Planning and Implementation

## Ten Heuristics (for query + results)

### Query-side heuristics

1. **Empty / too-short query guard**

   * **Scope:** Query
   * **Rule:** If the user input is empty or < 3 non-stopword tokens, don’t run perception/decision.
   * **Action:** Return `FINAL_ANSWER: Please add a bit more detail to your request.`

2. **Max query length / token budget**

   * **Scope:** Query
   * **Rule:** If query length > N chars (e.g. 3000) or estimated tokens > limit, running tools/LLM may blow context.
   * **Action:** Ask user to narrow: `FINAL_ANSWER: Your request is quite long. Please specify a smaller part you want me to work on.`

3. **Blocked vs allowed domains (URLs)**

   * **Scope:** Query (URLs, web tools)
   * **Rule:** If URL domain matches a deny-list such as `gmail.com`, `drive.google.com`, `localhost`, internal IPs, or custom “private” domains, do not fetch.
   * **Action:** Reject call and explain: `FINAL_ANSWER: For privacy reasons I can’t access that site. Please paste the relevant text instead.`

4. **Suspicious / harmful script detection**

   * **Scope:** Query
   * **Rule:** If query asks to generate or execute shell commands, malware, exploits (`rm -rf`, `powershell -enc`, “bypass antivirus”, etc.), block.
   * **Action:** Refuse and respond safely; do **not** call any tool.

5. **Confidential or highly sensitive content**

   * **Scope:** Query & document content
   * **Rule:** If text contains patterns like credit-card numbers, SSN-like patterns, API keys, “confidential / internal only” banners, or passwords, treat as sensitive.
   * **Action:** Mask or reject: `FINAL_ANSWER: It looks like this text may contain confidential or secret information, so I can’t process it. Please remove secrets and try again.`

6. **File size / content size guard for RAG tools**

   * **Scope:** Query → before indexing or summarizing docs/webpages
   * **Rule:** If extracted document or webpage text exceeds a safe size (e.g. > 200k chars), don’t fully embed/process.
   * **Action:** Either sample a smaller chunk or ask the user which section to focus on; return `FURTHER_PROCESSING_REQUIRED` with a short message instead of brute-forcing.

### Result-side heuristics

7. **Tool name & call validation**

   * **Scope:** Plan → before sandbox run
   * **Rule:** If the generated plan tries to call a tool that’s not in the selected tool list or uses a non-string name (`mcp.call_tool(add, ...)`), treat as invalid.
   * **Action:** Reject that plan, burn one lifeline, and ask decision model for a new plan.

8. **Output length + type sanity**

   * **Scope:** Result from sandbox
   * **Rule:** If final result is extremely long, mostly HTML, JSON, or binary-ish (huge base64 chunks), it’s not a clean answer.
   * **Action:** Wrap as `FURTHER_PROCESSING_REQUIRED` and let the next step summarize instead of returning raw blobs.

9. **Banned words / profanity filter**

   * **Scope:** Final answer string
   * **Rule:** If `FINAL_ANSWER:` contains profanity, slurs, or custom banned phrases, do not return it as-is.
   * **Action:** Either mask them (`****`) or re-ask the LLM to regenerate a neutral answer; always keep the prefix `FINAL_ANSWER:`.

10. **Numeric & factual sanity check**

    * **Scope:** Result
    * **Rule:** When the answer is supposed to be a numeric result (sum, log, EMI etc.), ensure it parses as a number and is not `"nan"`, `"Infinity"`, or obviously nonsense. Also, if tools didn’t supply any numeric value, the agent must not invent one.
    * **Action:** If check fails, convert to `FURTHER_PROCESSING_REQUIRED` and include the raw tool output so the next step (or user) can decide.

---

### Optional code skeleton [`heuristics.py`](modules/heuristics.py)

---

## 4. New `decision_prompt_conservative.txt` (297 words < 300 words)

```text

You are an LLM planner. Your job is to generate a single valid async Python function named `solve()` that uses ONLY the tools listed below.

Tool Catalog:
{tool_descriptions}

User Query:
"{user_input}"

RULES:
- Output ONLY Python code.
- Define exactly one function: `async def solve():`
- Use at most one tool call inside `solve()`.
- Use only tools listed in the Tool Catalog. Do not invent tool names.
- Call tools with the injected `mcp` object using POSITIONAL args ONLY.
  Do NOT use keyword arguments like `input=...`.
- Always pass a dict as `input`, matching the documented schema.
- Before the call, include the tool’s provided docstring inside triple quotes (""").
- To read a JSON tool result, do:

    data = json.loads(result.content[0].text)["result"]

  Never inline `json.loads(...)` inside f-strings. Never reference `result` before assigning it.
- Never access attributes like `mcp.last_tool_result` or `mcp.last_tool_call`. You only have `mcp.call_tool(...)` and local variables.
- Never return raw tool objects or JSON. `FINAL_ANSWER` must be a clean, human-readable string.
- Use `FINAL_ANSWER` only when you can answer directly from parsed `data`. Never invent or approximate numeric values. If the tool result does not contain a clear, parsable answer, return `FURTHER_PROCESSING_REQUIRED` with the raw `result`.
- If the tool returns search results, URLs, or long text that still needs summarizing, return `FURTHER_PROCESSING_REQUIRED` with `result`.
- If the user already gave clean text content, do NOT call any tool; answer directly and return `FINAL_ANSWER`.

Example pattern (guidance only):


async def solve():
    # FUNCTION_CALL: 1
    """Compute the base-10 logarithm of a number. Usage: input={{"input": {{"a": 100}}}} result = await mcp.call_tool('log10', input)"""
    input = {{"input": {{"a": 1000}}}}
    result = await mcp.call_tool('log10', input)
    return f"FINAL_ANSWER: The base-10 logarithm is {{result}}"

Tips:

If a monetary amount is given in words, first extract that number then convert correctly.

```

Answer obtained on given new `decision_prompt_conservative.txt`. Full log is in [`questions_answers_log2.txt`](questions_answers_log2.txt):

1. What do you know about Don Tapscott and Anthony Williams?

Don Tapscott and Anthony Williams have argued for the use of open source strategies in respect of clean technologies. In their 2010 book MacroWikinomics, they called for the establishment of a green technology commons.

2. What is the relationship between Gensol and Go-Auto?

Gensol transferred funds to Go-Auto, an auto dealership, to purchase electric vehicles for BluSmart.

3. What is the log value of the amount that Anmol singh paid for his DLF apartment via Capbridge? 

The log value of the amount Anmol Singh paid is 8.63286204010023

4. How much Anmol singh paid for his DLF apartment via Capbridge? 

Anmol Singh paid ₹42.94 crore for his DLF apartment via Capbridge.

5. Summarize this page: https://theschoolof.ai/

TSAI offers courses on AI/ML, focusing on practical implementation and advanced concepts. Key courses include ERA (training LLMs), EAG (Agentic AI), and EPAi (Python for AI). The courses cover a wide range of topics from foundational concepts to advanced techniques like multi-agent systems, reinforcement learning, and MLOps. Registrations for some courses are scheduled for April 2026. The program emphasizes hands-on learning and preparing students for leadership in AI development.


## 🧠 Historical Memory + Semantic Cache: How It Works

I implemented a semantic query cache on top of my historical conversation index. Each past query is stored with normalized keywords. For any new user query, I compute a Jaccard similarity over keyword sets against all previous queries. If any match exceeds a threshold (for example 0.90), I immediately return the stored `FINAL_ANSWER` from memory and completely bypass the perception–planning–action loop. In addition, I inject the top-k similar examples into the planner prompt, so the LLM can reuse prior `FINAL_ANSWER`s even when the similarity is slightly below the hard threshold.

This section explains how the agent:

1. **Indexes past conversations** into a global memory file.  
2. **Finds semantically similar queries** using Jaccard similarity over keywords.  
3. **Directly returns a cached `FINAL_ANSWER`** when the new query is “close enough.”  
4. **Injects similar examples into the planner prompt** so the LLM can reuse answers without re-planning.

---

## 1. Data store: `historical_conversation_store.json`

All historical Q&A pairs are stored in:

```text
memory/historical_conversation_store.json
````

Each entry looks like this:

```json
{
  "session_id": "2025/11/28/session-1764386898-6bd8e8",
  "turn_index": 0,
  "user_query": "How much Anmol singh paid for his DLF apartment via Capbridge?",
  "final_answer": "FINAL_ANSWER: A key transaction flagged by SEBI involved DLF...",
  "tools_used": ["solve_sandbox"],
  "successful_tools": ["solve_sandbox"],
  "tags": ["run_start", "sandbox"],
  "keywords": ["anmol", "singh", "paid", "dlf", "apartment", "capbridge"]
}
```

Important:

* Only **successful runs** with a clean `FINAL_ANSWER:` are indexed.
* Junk answers like `FINAL_ANSWER: [Could not generate valid solve()]` are **ignored**.
* `keywords` are the tokenized, de-stopworded version of the query.
  Example:
  `"How much Anmol singh paid for his DLF apartment via Capbridge?"` →
  `["how", "much", "anmol", "singh", "paid", "dlf", "apartment", "capbridge"]`
  After stopword removal it might become: `["anmol", "singh", "paid", "dlf", "apartment", "capbridge"]`.

---

## 2. Indexing: `update_index_for_session(...)`

Every time a run finishes, `loop.py` calls:

```python
update_index_for_session(mm.memory_path, mm.session_id)
```

This function:

1. Reads the **per session memory JSON** (from `MemoryManager`).

2. Walks through events to find:

   * a `run_metadata` entry with
     `"Started new session with input: ..."` → this gives the `user_query`
   * the following `tool_output` entries that contain a `FINAL_ANSWER: ...`.

3. Builds a `HistoricalExample`:

   ```python
   HistoricalExample(
       session_id=session_id,
       turn_index=turn_index,
       user_query=user_query,
       final_answer=final_answer,
       tools_used=...,
       successful_tools=...,
       tags=...,
       keywords=_normalize_text(user_query),
   )
   ```

4. Appends this to `historical_conversation_store.json`, deduped by `(session_id, turn_index)`.

So the memory file is gradually filled with **real successful Q&A pairs** across sessions.

---

## 3. Keyword extraction and normalization

Function used: `_normalize_text(text: str) -> List[str>`

Algorithm:

1. Lowercase the text.
2. Extract `[a-z0-9]+` tokens using regex.
3. Remove English stopwords like `"the", "is", "a", "how", "what", "much", "when", "where"...`.

Example:

```text
Input: "relationship between Gensol and Go-Auto?"
Tokens: ["relationship", "between", "gensol", "and", "go", "auto"]
After stopword removal: ["relationship", "gensol", "go", "auto"]
```

These keyword lists are stored with each historical example and used to compute similarity.

---

## 4. Similarity metric: Jaccard over keywords

Similarity is computed with `_jaccard_similarity(a, b)`:

```python
sa = set(a)
sb = set(b)
similarity = len(sa ∩ sb) / len(sa ∪ sb)
```

Intuition:

* If two queries share many important words and have similar sets of keywords, the Jaccard score is high (close to 1.0).
* If they share very few or no keywords, the score is low.

Example:

```text
Q1 keywords: ["anmol", "singh", "paid", "dlf", "apartment", "capbridge"]
Q2 keywords: ["how", "much", "anmol", "singh", "paid", "dlf", "apartment", "capbridge"]
          → after stopword removal, Q2 keywords ≈ Q1 keywords

Set1 = {"anmol", "singh", "paid", "dlf", "apartment", "capbridge"}
Set2 = {"anmol", "singh", "paid", "dlf", "apartment", "capbridge"}

Intersection = 6, Union = 6
Jaccard = 6 / 6 = 1.00
```

So variations like:

* `"How much Anmol singh paid for his DLF apartment via Capbridge?"`
* `"How much Anmol Singh paid for DLF apartment via Capbridge"`

will get **very high similarity**, even if punctuation or small words differ.

---

## 5. Semantic cache API: `find_best_cached_answer(...)`

This is the main entry point for cache hits:

```python
def find_best_cached_answer(
    user_query: str,
    min_similarity: float = 0.90,
    memory_root: Optional[str] = None,
) -> Optional[str]:
    ...
```

Algorithm:

1. Load all historical items from `historical_conversation_store.json`.
2. Compute `q_keywords = _normalize_text(user_query)`.
3. For each item:

   * Skip if `final_answer` is invalid or “junk.”
   * Get its `keywords` (or recompute from `user_query` if missing).
   * Compute `sim = jaccard(q_keywords, keywords)`.
4. Track the **best match** `(best_sim, best_answer, best_query)`.
5. If `best_sim >= min_similarity`:

   * Log:
     `✅ Semantic cache hit for 'X' -> 'Y' (sim=0.95)`
   * Return `best_answer` (which starts with `FINAL_ANSWER:`).
6. Otherwise return `None`.

So the cache is **semantic** but still uses a simple, deterministic similarity measure that is cheap and fast.

---

## 6. Using the cache in the loop: bypassing the whole agent

At the very top of `AgentLoop.run()` in `loop.py`:

```python
original_query = self.context.user_input or ""
semantic_hit = find_best_cached_answer(original_query, min_similarity=0.90)

if semantic_hit:
    log("loop", "⚡ Semantic memory hit — returning cached FINAL_ANSWER.")
    self.context.final_answer = semantic_hit.strip()
    self.context.memory.add_final_answer(self.context.final_answer)
    self._update_historical_index()
    return {"status": "done", "result": self.context.final_answer}
```

Effect:

* If a **similar question** has already been answered, the agent:

  * **Skips perception**
  * **Skips planning**
  * **Skips all tool calls and sandbox execution**
* It simply returns the previous `FINAL_ANSWER:` from memory.

For example: Queries below matched with similarity 0.86 and the cached answer is returned instantly.:

* `relationship between Gensol and Go-Auto?`
* `What is the relationship between Gensol and Go-Auto?`

Once one of them is answered, the other one hits the cache and returns instantly.

---

## 7. Injecting historical examples into the planner prompt

Besides the hard cache logic, the planner is also given “soft” historical context.

In `decision.py`, when building the planner prompt, we:

1. Call:

   ```python
   similar_examples = load_similar_examples(user_query, top_k=3)
   ```

   which returns a short list of high similarity past queries.

2. Format them in the prompt as:

   ```text
   Past similar interactions:
   - Past query: What is the relationship between Gensol and Go-Auto?
     Tools: solve_sandbox
     Outcome: FINAL_ANSWER: Gensol transferred funds to Go-Auto, ...
   - Past query: relationship between Don Tapscott and Anthony Williams?
     Tools: solve_sandbox
     Outcome: FINAL_ANSWER: Don Tapscott and Anthony Williams are the authors of ...
   ```

3. Add a **very explicit instruction**:

   ```text
   ---
   IF the 'User query' is a CLOSE PARAPHRASE of any 'Past query'
   AND the 'Outcome' contains a `FINAL_ANSWER:`:
   * ACTION: STOP NOW. Do NOT write any Python code (`async def solve`).
   * OUTPUT: RESPOND ONLY with the exact `FINAL_ANSWER: ...` text from the 'Outcome'.
   ---
   OTHERWISE, PROCEED TO PLANNING RULES:
   ```

This makes the LLM planner “aware” of similar past answers. Even if the semantic cache did not trigger (for example, similarity is high but below threshold), the LLM still sees those examples and is strongly nudged to **reuse the existing `FINAL_ANSWER`** instead of planning a new tool call.

---

## 8. End-to-end flow example

### First time question

User:
`How much Anmol singh paid for his DLF apartment via Capbridge?`

1. Cache lookup:

   * `find_best_cached_answer` returns `None` (first time, no history).

2. Normal loop:

   * Perception: selects web/doc tools.
   * Planner: creates a `solve()` with tool calls.
   * Action: runs tools, parses SEBI PDF, etc.
   * Final answer:

     ```text
     FINAL_ANSWER: A key transaction flagged by SEBI involved DLF...
     ```

3. Indexing:

   * `update_index_for_session` writes this query + answer into `historical_conversation_store.json`.

### Slightly different wording later

User:
`How much Anmol Singh paid for DLF apartment via Capbridge`

1. Cache lookup:

   * `_normalize_text` makes both queries almost identical keyword sets.
   * Jaccard similarity ~ 1.0.
   * `best_sim >= 0.90` so cache hit.
2. The loop returns instantly:

   ```text
   ⚡ Semantic memory hit — returning cached FINAL_ANSWER.
   ```

   And the user immediately gets the same `FINAL_ANSWER` without any extra tool calls or planning.

---

## 9. Why this is “smart” indexing?

* It is **global** across sessions (`historical_conversation_store.json`).
* It is **selective**:

  * Only successful `FINAL_ANSWER`s are stored.
  * Junk / error answers are skipped.
* It is **semantic**:

  * Uses Jaccard over cleaned keywords to tolerate small variations.
* It is **integrated in two ways**:

  1. **Hard semantic cache** in `loop.py`:

     * Short-circuits the entire agent pipeline and returns cached results.
  2. **Soft prompt memory** in `decision.py`:

     * Passes “Past similar interactions” into the planner with an explicit rule:
       If close paraphrase → reuse `FINAL_ANSWER` directly.

This combination gives multiple benefits:

* Speed: repeated or paraphrased queries are answered instantly.
* Stability: the user gets consistent answers across sessions.
* Lower cost: fewer tool calls and LLM planning steps for repeated questions.


