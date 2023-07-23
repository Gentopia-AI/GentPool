from copy import deepcopy
from typing import Union, Dict, Tuple
import yaml
from concurrent.futures import ThreadPoolExecutor
from gentpool.bench.eval import BaseEvalPipeline
from gentpool.bench.eval.base_eval import EvalPipelineResult
from gentpool.bench.eval.evaluator import *
from gentopia.output.console_output import ConsoleOutput


class EvalPipeline(BaseEvalPipeline):
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

    def _eval(self, agent: BaseAgent, cls, eval_class: str, eval_subclass: str, grader: BaseGrader,
              seed: int, output) -> Tuple[int, EvalResult]:
        # output.update_status(f"> EVALUATING: {eval_class}/{eval_subclass} ...")
        n = self.eval_config.get(eval_class, {}).get(eval_subclass, 0)
        evaluator = cls(eval_class=eval_class, eval_subclass=eval_subclass, grader=grader)
        result = evaluator.evaluate(agent, n, seed, verbose=False)
        # output.done()
        output.print(f"Done: {eval_class}/{eval_subclass}: {result.score}")
        return n, result, eval_class, eval_subclass

    def run_eval_sym(self, agent: BaseAgent, seed: int = 0, output=ConsoleOutput(), save_dir=None) -> EvalPipelineResult:
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
        agent.clear()
        with ThreadPoolExecutor() as pool:
            _tasks = []
            for eval_class, task in tasks.items():
                for eval_subclass in task:
                    grader = None if eval_subclass == "coding" else GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm))
                    if eval_subclass == "integrity":
                        grader = InstructedGrader(llm=OpenAIGPTClient(model_name=self.grader_llm))
                    _agent = deepcopy(agent)
                    _tasks.append(pool.submit(self._eval, _agent, evaluator[eval_subclass], eval_class, eval_subclass,
                                              grader, seed, output))
            for future in _tasks:
                n, result, eval_class, eval_subclass = future.result()
                total_eval_count += n
                eval_results[f"{eval_class}/{eval_subclass}"] = result


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

        return final_result



    def run_eval(self, agent: BaseAgent, seed: int = 0, output=ConsoleOutput(), save_dir=None) -> EvalPipelineResult:

        if isinstance(self.eval_config, str):
            self.eval_config = self._parse_config_from_file(self.eval_config)

        # TODO: Add support for following eval types:
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

        # knowledge/world_knowledge
        # print("> EVALUATING: knowledge/world_knowledge ...")
        output.update_status("> EVALUATING: knowledge/world_knowledge ...")
        n = self.eval_config.get("knowledge", {}).get("world_knowledge", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="knowledge", eval_subclass="world_knowledge",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["knowledge/world_knowledge"] = evaluator.evaluate(agent, n, seed, verbose=False)
        output.done()


        # knowledge/domain_specific_knowledge
        output.update_status("> EVALUATING: knowledge/domain_specific_knowledge ...")
        n = self.eval_config.get("knowledge", {}).get("domain_specific_knowledge", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="knowledge", eval_subclass="domain_specific_knowledge",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["knowledge/domain_specific_knowledge"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # knowledge/web_retrieval
        output.update_status("> EVALUATING: knowledge/web_retrieval ...")
        n = self.eval_config.get("knowledge", {}).get("web_retrieval", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="knowledge", eval_subclass="web_retrieval",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["knowledge/web_retrieval"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # reasoning/math
        output.update_status("> EVALUATING: reasoning/math ...")
        n = self.eval_config.get("reasoning", {}).get("math", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="reasoning", eval_subclass="math",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["reasoning/math"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # reasoning/coding
        output.update_status("> EVALUATING: reasoning/coding ...")
        n = self.eval_config.get("reasoning", {}).get("coding", 0)
        total_eval_count += n
        evaluator = CodeEval(eval_class="reasoning", eval_subclass="coding")
        eval_results["reasoning/coding"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # reasoning/planning
        output.update_status("> EVALUATING: reasoning/planning ...")
        n = self.eval_config.get("reasoning", {}).get("planning", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="reasoning", eval_subclass="planning",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["reasoning/planning"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # reasoning/commonsense
        output.update_status("> EVALUATING: reasoning/commonsense ...")
        n = self.eval_config.get("reasoning", {}).get("commonsense", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="reasoning", eval_subclass="commonsense",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["reasoning/commonsense"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # safety/integrity
        output.update_status("> EVALUATING: safety/integrity ...")
        n = self.eval_config.get("safety", {}).get("integrity", 0)
        total_eval_count += n
        evaluator = IntegrityEval(eval_class="safety", eval_subclass="integrity",
                                  grader=InstructedGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["safety/integrity"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # safety/harmless
        output.update_status("> EVALUATING: safety/harmless ...")
        n = self.eval_config.get("safety", {}).get("harmless", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="reasoning", eval_subclass="commonsense",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["safety/harmless"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # multilingual/translation
        output.update_status("> EVALUATING: multilingual/translation ...")
        n = self.eval_config.get("multilingual", {}).get("translation", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="multilingual", eval_subclass="translation",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["multilingual/translation"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # multilingual/understanding
        output.update_status("> EVALUATING: multilingual/understanding ...")
        n = self.eval_config.get("multilingual", {}).get("understanding", 0)
        total_eval_count += n
        evaluator = QAEval(eval_class="multilingual", eval_subclass="understanding",
                           grader=GateGrader(llm=OpenAIGPTClient(model_name=self.grader_llm)))
        eval_results["multilingual/understanding"] = evaluator.evaluate(agent, n, seed, private, verbose=False)
        output.done()

        # robustness/consistency
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

        return final_result

    def run_eval_async(self, agent: BaseAgent, seed: int = 0, *args, **kwargs):
        raise NotImplementedError

    def _print_result(self, result: EvalPipelineResult, _output = ConsoleOutput(), save_dir=None):
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
            f"Agent run exception rate: {round(result.avg_fail_rate * 100,2)}%",
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

            # print(line, end=' ', flush=True)
            # time.sleep(0.7)
            # print()
        _output.panel_print("### FINISHING Agent EVAL PIPELINE ###", f"[{style}]{info}", True)
        _output.clear()


