import concurrent
from copy import deepcopy
from optparse import Option
from typing import Union, Dict, Tuple, Optional
import yaml
from concurrent.futures import ThreadPoolExecutor
from gentpool.bench.eval import BaseEvalPipeline
from gentpool.bench.eval.base_eval import EvalPipelineResult
from gentpool.bench.eval.evaluator import *
from gentopia.output.console_output import ConsoleOutput
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import zeno
from zeno import ZenoParameters


class MultiProcessEvalPipeline(BaseEvalPipeline):
    eval_config: Union[Dict, str]
    grader_llm: str = "gpt-4"

    def _parse_config_from_file(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _placeholder_eval_result(self) -> EvalResult:
        # For not yet supported eval tasks.
        return EvalResult(score=0.0, fail_rate=0.0, avg_runtime=0.0, avg_cost=0.0, avg_token_usage=0.0, eval_cost=0.0)

    def _weigtht_avg_eval_results(self, eval_results: Dict[str, EvalResult], total_eval_count: int):
        avg_score = 0.0
        avg_fail_rate = 0.0
        avg_runtime = 0.0
        avg_cost = 0.0
        avg_toekn_usage = 0.0
        total_eval_cost = 0.0

        for eval_task, eval_result in eval_results.items():
            avg_score += eval_result.score * self.eval_config[eval_task.split("/")[0]][
                eval_task.split("/")[1]] / total_eval_count
            avg_fail_rate += eval_result.fail_rate * self.eval_config[eval_task.split("/")[0]][
                eval_task.split("/")[1]] / total_eval_count
            avg_runtime += eval_result.avg_runtime * self.eval_config[eval_task.split("/")[0]][
                eval_task.split("/")[1]] / total_eval_count
            avg_cost += eval_result.avg_cost * self.eval_config[eval_task.split("/")[0]][
                eval_task.split("/")[1]] / total_eval_count
            avg_toekn_usage += eval_result.avg_token_usage * self.eval_config[eval_task.split("/")[0]][
                eval_task.split("/")[1]] / total_eval_count
            total_eval_cost += eval_result.eval_cost

        return EvalPipelineResult(eval_results=eval_results,
                                  avg_score=avg_score,
                                  avg_fail_rate=avg_fail_rate,
                                  avg_runtime=avg_runtime,
                                  avg_cost=avg_cost,
                                  avg_token_usage=avg_toekn_usage,
                                  total_eval_cost=total_eval_cost)

    def run_eval(self, agent: BaseAgent, seed: int = 0, output=ConsoleOutput(), save_dir=None, eval_process: int = 8,
                 grade_process: int = 12) -> Tuple[EvalPipelineResult, List[Dict]]:
        if isinstance(self.eval_config, str):
            self.eval_config = self._parse_config_from_file(self.eval_config)

        if self.eval_config["robustness"].get("consistency", 0) > 0:
            raise NotImplementedError("Consistency eval is not supported yet.")
        if self.eval_config["robustness"].get("resilience", 0) > 0:
            raise NotImplementedError("Resilience eval is not supported yet.")
        if self.eval_config["memory"] == True:
            raise NotImplementedError("Memory eval is not supported yet.")

        verbose = self.eval_config.get("verbose", True)
        private = self.eval_config.get("private", False)

        eval_results = {}
        total_eval_count = 0

        tasks = {
            "knowledge": ["world_knowledge", "domain_specific_knowledge", "web_retrieval"],
            "reasoning": ["math", "coding", "planning", "commonsense"],
            "safety": ["integrity", "harmless"],
            "multilingual": ["translation", "understanding"],
            # "robustness": ["consistency", "resilience"],
        }
        evaluator = {
            "world_knowledge": QAEval,
            "domain_specific_knowledge": QAEval,
            "web_retrieval": QAEval,
            "math": QAEval,
            "coding": CodeEval,
            "planning": QAEval,
            "commonsense": QAEval,
            "integrity": IntegrityEval,
            "harmless": QAEval,
            "translation": QAEval,
            "understanding": QAEval
        }
        LOG = []
        agent.clear()
        evaluate_pool = ThreadPoolExecutor(eval_process)
        grader_pool = ThreadPoolExecutor(grade_process)
        try:
            futures = []
            results = []
            for eval_class, task in tasks.items():
                for eval_subclass in task:
                    eval_results[f"{eval_class}/{eval_subclass}"] = self._placeholder_eval_result()
                    n = self.eval_config.get(eval_class, {}).get(eval_subclass, 0)
                    total_eval_count += n
                    for index in range(n):
                        _agent = deepcopy(agent)
                        cls = evaluator[eval_subclass]
                        grader = None if eval_subclass == "coding" else GateGrader(
                            llm=OpenAIGPTClient(model_name=self.grader_llm))
                        if eval_subclass == "integrity":
                            grader = InstructedGrader(llm=OpenAIGPTClient(model_name=self.grader_llm))
                        _evaluator = cls(eval_class=eval_class, eval_subclass=eval_subclass, grader=grader)
                        futures.append(evaluate_pool.submit(_evaluator.evaluate_single, _agent, index, n, seed))
            l = len(futures)
            while futures:
                done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                output.update_status(f"Evaluated {l - len(not_done)}/{l} tasks.")
                for i in done:
                    item, index, result, response, log = i.result()
                    name = f"{item.eval_class}/{item.eval_subclass}"
                    eval_results[name] += result
                    results.append(grader_pool.submit(item.grade_single, response, index))
                    LOG.append(log)
                futures = list(not_done)
            output.done(_all=True)
            output.update_status(f"Wating for grading ...")
            concurrent.futures.wait(results)
            output.done(_all=True)
            for i in results:
                item, result, log = i.result()
                name = f"{item.eval_class}/{item.eval_subclass}"
                output.print(f"{name} Done")
                eval_results[name] += result
                LOG.append(log)

            for i, j in eval_results.items():
                eval_class, eval_subclass = i.split("/")
                n = self.eval_config.get(eval_class, {}).get(eval_subclass, 0)
                if n:
                    j.score /= n
                    j.fail_rate /= n
                    j.avg_runtime /= n
                    j.avg_cost /= n
                    j.avg_token_usage /= n

        finally:
            evaluate_pool.shutdown()
            grader_pool.shutdown()

        output.update_status("> EVALUATING: robustness/consistency ...")
        eval_results["robustness/consistency"] = self._placeholder_eval_result()
        output.done()

        # robustness/resilience
        output.update_status("> EVALUATING: robustness/resilience ...")
        eval_results["robustness/resilience"] = self._placeholder_eval_result()
        output.done()

        # #memory
        # print("> EVALUATING: memory ...")
        # eval_results["memory"] = self._placeholder_eval_result()

        # weighted average
        final_result = self._weigtht_avg_eval_results(eval_results, total_eval_count)

        # print to console:
        if verbose:
            self._print_result(final_result, output, save_dir)

        return final_result, LOG

    def run_eval_async(self, agent: BaseAgent, seed: int = 0, *args, **kwargs):
        raise NotImplementedError

    def _print_result(self, result: EvalPipelineResult, _output=ConsoleOutput(), save_dir=None):
        output = [
            "\n### FINISHING Agent EVAL PIPELINE ###",
            "--------------Task Specific--------------",
            f"Score of knowledge/world_knowledge: {result.eval_results['knowledge/world_knowledge'].score * 100}",
            f"Score of knowledge/domain_specific_knowledge: {result.eval_results['knowledge/domain_specific_knowledge'].score * 100}",
            f"Score of knowledge/web_retrieval: {result.eval_results['knowledge/web_retrieval'].score * 100}",
            f"Score of reasoning/math: {result.eval_results['reasoning/math'].score * 100}",
            f"Score of reasoning/coding: {result.eval_results['reasoning/coding'].score * 100}",
            f"Score of reasoning/planning: {result.eval_results['reasoning/planning'].score * 100}",
            f"Score of reasoning/commonsense: {result.eval_results['reasoning/commonsense'].score * 100}",
            f"Score of safety/integrity: {result.eval_results['safety/integrity'].score * 100}",
            f"Score of safety/harmless: {result.eval_results['safety/harmless'].score * 100}",
            f"Score of multilingual/translation: {result.eval_results['multilingual/translation'].score * 100}",
            f"Score of multilingual/understanding: {result.eval_results['multilingual/understanding'].score * 100}",
            f"Score of robustness/consistency: {result.eval_results['robustness/consistency'].score * 100}",
            f"Score of robustness/resilience: {result.eval_results['robustness/resilience'].score * 100}",
            # f"Score of memory: {result.eval_results['memory'].score*100}",
            "-----------Overal (Weighted Avg)-----------",
            f"Agent score: {round(result.avg_score * 100, 2)}",
            f"Agent run exception rate: {round(result.avg_fail_rate * 100, 2)}%",
            f"Avg runtime per task: {round(result.avg_runtime, 2)}s",
            f"Avg cost per run: ${round(result.avg_cost, 3)}",
            f"Avg token usage per task: {round(result.avg_token_usage, 1)} tokens",
            f"... And the total cost for evaluation ${round(result.total_eval_cost, 5)}"
        ]
        if result.avg_score >= 0.8:
            info, style = "Excellent Scoring! (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧", "green"
        elif result.avg_score >= 0.5:
            info, style = "Not bad at all! (￣▽￣)ノ", "yellow"
        else:
            info, style = "Try out some specialization tricks (z￣▽￣)z ", "red"

        for line in output:
            _output.panel_print(line + '\n\n', f"[{style}]{info}", True)
            if save_dir:
                with open(os.path.join(save_dir, "eval_result.txt"), "a+") as f:
                    f.write(line + '\n\n')

        _output.panel_print("### FINISHING Agent EVAL PIPELINE ###", f"[{style}]{info}", True)
        _output.clear()

    def _parse_eval_to_markdown(self, eval:dict) -> str:
        prompt = eval.get("prompt", "")
        grade = eval.get("output", "")
        return f"## Grader Eval\n{prompt}\n{grade}"

    def vis(self, log: List[Dict], view: str, name: str = 'Agent'):
        assert view in ["openai-chat-markdown", "openai-chat", "chatbot"], "view must be 'openai-chat' or 'chatbot'"
        try:
            from zeno_build.experiments.experiment_run import ExperimentRun
            from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn
            from zeno_build.reporting.visualize import visualize
            from pandas import DataFrame as df

            if view == "openai-chat":
                pass
            else:
                data_column = [c for c in log if isinstance(c, list)]
                eval_column = [self._parse_eval_to_markdown(c) for c in log if isinstance(c, dict)]
                data = df({'data_column': data_column, 'eval_column': eval_column})
                zeno.zeno(ZenoParameters(metadata=data, data_column="data_column", label_column="eval_column", view=view))
        except Exception as e:
            raise e
            pass
