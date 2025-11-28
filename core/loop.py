# modules/loop.py

import asyncio
from core import context
from modules.perception import run_perception
from modules.decision import generate_plan
from modules.action import run_python_sandbox
from modules.model_manager import ModelManager
from core.session import MultiMCP
from core.strategy import select_decision_prompt_path
from core.context import AgentContext
from modules.tools import summarize_tools
import re

try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

class AgentLoop:
    def __init__(self, context: AgentContext):
        self.context = context
        self.mcp = self.context.dispatcher
        self.model = ModelManager()

    async def _finalize_from_content(self, original_question: str, tool_result: str) -> str:
        """
        Turn raw tool output + user question into a concise final answer.

        This is called when we've hit the FURTHER_PROCESSING_REQUIRED limit.
        It is fully generic: works for "relationship" questions, factual questions,
        numeric questions, etc., without special branching in Python.
        """

        prompt = f"""
You are a careful but concise assistant.

The user asked:
{original_question}

You have the following context from tools (search results, excerpts, or documents).
It may include URLs, summaries, long snippets, or even messages like "no results found":

{tool_result}

Follow this process strictly:

1. Understand what the user is actually asking for
   (e.g., description of people, relationship between entities, a specific amount,
   a definition, a comparison, etc.).

2. Read the context carefully and look for ANY information that helps answer the question.
   Pay special attention to:
   - How entities are connected (customer/vendor, owner/subsidiary, intermediary,
     related party, fund flow between them, etc.).
   - Numbers (amounts, prices, quantities), dates, names, and clear statements.

3. If the context clearly contains enough information to answer, give a direct,
   specific answer in 1–2 sentences.

4. If the context is partial but still gives clues, synthesize the best possible
   answer. In that case:
   - Make a reasonable inference.
   - Mention that the answer is based on limited information from the context.

5. Only answer exactly "unknown" (or an equivalent like "cannot be determined from the context")
   if BOTH of these are true:
   - The context is truly unrelated to the question OR explicitly says that no results
     or no information were found, AND
   - After reading everything, you find no useful evidence that helps answer the question.

Important:
- Prefer giving a best-effort, concrete answer over saying "unknown" whenever the
  context contains any relevant evidence.
- Do NOT repeat large chunks of the context. Just state the conclusion.
- Your final output must be ONLY the answer text, no bullet points, no explanation
  of your steps.

Now, provide your final answer.
"""

        try:
            text = await self.model.generate_text(prompt)
            return text.strip()
        except Exception as e:
            log("loop", f"⚠️ Finalization failed: {e}")
            # Fallback: at least return some of the raw tool_result
            return tool_result[:2000]
        
    async def run(self):
        max_steps = self.context.agent_profile.strategy.max_steps
        allowed_fpr_uses = max_steps - 1  # e.g., 2 when max_steps = 3
        # track how many times we've used FURTHER_PROCESSING_REQUIRED
        self.further_processing_uses = 0

        for step in range(max_steps):
            print(f"🔁 Step {step+1}/{max_steps} starting...")
            self.context.step = step
            lifelines_left = self.context.agent_profile.strategy.max_lifelines_per_step

            while lifelines_left >= 0:
                # === Perception ===
                user_input_override = getattr(self.context, "user_input_override", None)
                perception = await run_perception(context=self.context, user_input=user_input_override or self.context.user_input)

                print(f"[perception] {perception}")

                selected_servers = perception.selected_servers
                selected_tools = self.mcp.get_tools_from_servers(selected_servers)
                if not selected_tools:
                    log("loop", "⚠️ No tools selected — aborting step.")
                    break
                
                effective_user_input = user_input_override or self.context.user_input
                
                # === Planning ===
                tool_descriptions = summarize_tools(selected_tools)
                prompt_path = select_decision_prompt_path(
                    planning_mode=self.context.agent_profile.strategy.planning_mode,
                    exploration_mode=self.context.agent_profile.strategy.exploration_mode,
                )

                plan = await generate_plan(
                    user_input=effective_user_input,  # ✅ use override
                    perception=perception,
                    memory_items=self.context.memory.get_session_items(),
                    tool_descriptions=tool_descriptions,
                    prompt_path=prompt_path,
                    step_num=step + 1,
                    max_steps=max_steps,
                )
                print(f"[plan] {plan}")

                # === Execution ===
                if re.search(r"^\s*(async\s+)?def\s+solve\s*\(", plan, re.MULTILINE):
                    print("[loop] Detected solve() plan — running sandboxed...")

                    self.context.log_subtask(tool_name="solve_sandbox", status="pending")
                    result = await run_python_sandbox(plan, dispatcher=self.mcp)

                    success = False
                    if isinstance(result, str):
                        result = result.strip()
                        if result.startswith("FINAL_ANSWER:"):
                            success = True
                            self.context.final_answer = result
                            self.context.update_subtask_status("solve_sandbox", "success")
                            self.context.memory.add_tool_output(
                                tool_name="solve_sandbox",
                                tool_args={"plan": plan},
                                tool_result={"result": result},
                                success=True,
                                tags=["sandbox"],
                            )
                            return {"status": "done", "result": self.context.final_answer}
                        

                        elif result.startswith("FURTHER_PROCESSING_REQUIRED:"):
                            content = result.split("FURTHER_PROCESSING_REQUIRED:")[1].strip()
                            self.further_processing_uses += 1

                            if self.further_processing_uses <= allowed_fpr_uses:
                                # ✅ Still allowed to forward
                                self.context.user_input_override = (
                                    f"Original user task: {self.context.user_input}\n\n"
                                    f"Your last tool produced this result:\n\n"
                                    f"{content}\n\n"
                                    f"If this fully answers the task, return:\n"
                                    f"FINAL_ANSWER: your answer\n\n"
                                    f"Otherwise, return the next FUNCTION_CALL."
                                )
                                log("loop", f"📨 Forwarding intermediate result to next step:\n{self.context.user_input_override}\n\n")
                                log("loop", f"🔁 Continuing based on FURTHER_PROCESSING_REQUIRED — Step {step+1} continues...")
                                break  # go to next step

                            else:
                                # ❌ We've exceeded the allowed FPR uses, must finalize now
                                log("loop", "⚠️ FURTHER_PROCESSING_REQUIRED exceeded budget — forcing FINAL_ANSWER via summarization.")
                                final_answer = await self._finalize_from_content(
                                    original_question=self.context.user_input,
                                    tool_result=content,
                                )
                                self.context.final_answer = f"FINAL_ANSWER: {final_answer}"
                                self.context.memory.add_final_answer(self.context.final_answer)
                                return {"status": "done", "result": self.context.final_answer}



                        elif result.startswith("[sandbox error:"):
                            success = False
                            self.context.final_answer = "FINAL_ANSWER: [Execution failed]"
                        else:
                            success = True
                            self.context.final_answer = f"FINAL_ANSWER: {result}"
                    else:
                        self.context.final_answer = f"FINAL_ANSWER: {result}"

                    if success:
                        self.context.update_subtask_status("solve_sandbox", "success")
                    else:
                        self.context.update_subtask_status("solve_sandbox", "failure")

                    self.context.memory.add_tool_output(
                        tool_name="solve_sandbox",
                        tool_args={"plan": plan},
                        tool_result={"result": result},
                        success=success,
                        tags=["sandbox"],
                    )

                    if success and "FURTHER_PROCESSING_REQUIRED:" not in result:
                        return {"status": "done", "result": self.context.final_answer}
                    else:
                        lifelines_left -= 1
                        log("loop", f"🛠 Retrying... Lifelines left: {lifelines_left}")
                        continue
                else:
                    log("loop", f"⚠️ Invalid plan detected — retrying... Lifelines left: {lifelines_left-1}")
                    lifelines_left -= 1
                    continue

        log("loop", "⚠️ Max steps reached without finding final answer.")
        self.context.final_answer = "FINAL_ANSWER: [Max steps reached]"
        return {"status": "done", "result": self.context.final_answer}
