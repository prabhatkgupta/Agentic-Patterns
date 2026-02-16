# Agentic Patterns in AI Systems

Reusable blueprints for agentic workflows: fixed steps + model-driven behaviour. Implemented with LangGraph and `/chat/completions` from first principles.

| Pattern | Essence |
|--------|--------|
| **1. Reflection** | Generate → critique → refine |
| **2. ReAct Tool-Use** | Reasoning + acting with tools |
| **3. Orchestrator–Worker** | Central orchestrator + specialized workers |
| **4. Multi-Agent** | Specialized agents with handoffs |

---

## 1. Reflection

**Generate → critique → refine.** Answer generator produces a response; reflector gives feedback (tone, correctness, completeness). A decision node either loops back for another pass or ends. State: `query`, `feedback`, `messages`, `answer`, `max_iterations`.

**Use cases:** Code refinement (write → run → use errors as feedback), IR (retrieve → grade → refine).

![Reflection](Reflection/langgraph_reflection.png)

---

## 2. ReAct Tool-Use

**Reasoning + acting.** Chain-of-thought plus tool calls (search, calculator, APIs, optionally MCP). Model reasons, calls tools, gets results, repeats until it can answer or hits a cap. State: `query`, `messages`, `reasoning`, `tool_calls`, `tool_call_responses`, `answer`, `iterations`, `session`.

**Use cases:** External APIs via MCP, sandboxed DB queries.

![ReAct Tool-Use](ReAct%20Tool-Use/langgraph_tool_call_client.png)

---

## 3. Orchestrator–Worker

**Central orchestrator + workers.** Orchestrator parses the query and emits sub-tasks; workers (e.g. Web-Search, Knowledge-Base) run in parallel via LangGraph `Send`; synthesizer aggregates results. State: `query`, `workers`, `messages`, `result`, `session`, `final_answer`.

**Use cases:** Multi-source QA (e.g. weather + calendar in one question).

![Orchestrator-Worker](Orchestrator-Worker/orchestrator-worker.png)

---

## 4. Multi-Agent

**Specialized agents + handoffs.** Multiple agents, each with own prompt and tools; **handoff** is a tool that passes control to another agent. Graph cycles through the active agent; conditional edges route on handoff or end. State: `iterations`, `session`, `messages`, `answer`, `query`, `tool_calls`, `handoff_message`, `current_state`.

**Use cases:** Complex plans (e.g. 3-day Paris itinerary: calendar + travel + booking agents).

![Multi-Agent](Multi-Agent/langgraph_multiagent.png)

---

## LangGraph

LangGraph provides **State** (TypedDict + reducers), **Nodes**, **Edges**, and **Conditional edges** (including fan-out via `Send`). Run with `graph.invoke(input)` or `ainvoke`. Patterns can be combined (e.g. Orchestrator–Worker with tool-using workers, or Multi-Agent with reflection per agent).

**References:** [Phil Schmid – Agentic Pattern](https://www.philschmid.de/agentic-pattern) · [LangChain multi-agent routing](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base) · [Handoffs (customer support)](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support)
