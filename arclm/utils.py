"""
Small shared utility helpers.
"""

import json
import time
import re
import torch
from datetime import datetime


def format_duration(total_seconds):
    """Format a duration in seconds as hours, minutes, and seconds."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"




class PreModelBenchmark:

    DEFAULT_PROMPTS = {
        "simple_questions": [
            "What is the capital of France?",
            "What is the largest planet in our solar system?",
            "Who wrote 'Romeo and Juliet'?"
            "wat is 1+1?"
        ]
    }

    def __init__(
        self,
        model,
        model_name: str,
        prompts: dict = None,
        temperatures=None,
        max_new_tokens: int = 100,
    ):
        self.model = model
        self.model_name = model_name
        self.prompts = prompts or self.DEFAULT_PROMPTS
        self.temperatures = temperatures or [0.1, 0.25, 0.5, 1.0]
        self.max_new_tokens = max_new_tokens

        

    def run(self, log:bool=True):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results = {
            "model": self.model_name,
            "timestamp": timestamp,
            "max_new_tokens": self.max_new_tokens,
            "temperatures": self.temperatures,
            "results": []
        }

        total_tests = 0
        total_success = 0

        for level_name, prompts in self.prompts.items():

            print("\n" + "=" * 100)
            print(f"LEVEL: {level_name}")
            print("=" * 100)

            for prompt in prompts:

                prompt_data = {
                    "level": level_name,
                    "prompt": prompt,
                    "tests": []
                }
                if log:
                    print(f"\nPrompt: {prompt}")

                for temperature in self.temperatures:
                    
                    if log:
                        print(f"\nTemperature: {temperature}")

                    start_time = time.time()

                    try:

                        output = self.model.predict(
                            prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=temperature
                        )

                        generation_time = time.time() - start_time

                        test_result = {
                            "temperature": temperature,
                            "success": True,
                            "generation_time_seconds": round(generation_time, 4),
                            "output_length": len(output),
                            "output": output
                        }

                        total_success += 1
                        if log:
                            print(output)

                    except Exception as e:

                        generation_time = time.time() - start_time

                        test_result = {
                            "temperature": temperature,
                            "success": False,
                            "generation_time_seconds": round(generation_time, 4),
                            "error": str(e)
                        }
                        
                        print(f"Generation failed: {e}")

                    total_tests += 1
                    prompt_data["tests"].append(test_result)

                results["results"].append(prompt_data)

        results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": total_success,
            "failed_tests": total_tests - total_success,
            "success_rate": round(
                (total_success / total_tests) * 100,
                2
            )
        }

        return results

    def save_json(self, results, filename=None):

        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"benchmark_{self.model_name}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                indent=4,
                ensure_ascii=False
            )

        return filename

    def save_txt(self, results, filename=None):

        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"benchmark_{self.model_name}_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:

            f.write(f"Model: {results['model']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write("=" * 100 + "\n\n")

            for item in results["results"]:

                f.write(
                    f"\nLEVEL: {item['level']}\n"
                    f"{'-' * 80}\n"
                )

                f.write(f"Prompt: {item['prompt']}\n")

                for test in item["tests"]:

                    f.write(
                        f"\nTemperature: {test['temperature']}\n"
                    )

                    if test["success"]:

                        f.write(
                            f"Generation Time: "
                            f"{test['generation_time_seconds']} sec\n"
                        )

                        f.write(
                            f"Output Length: "
                            f"{test['output_length']} chars\n"
                        )

                        f.write(test["output"])
                        f.write("\n")

                    else:

                        f.write(
                            f"ERROR: {test['error']}\n"
                        )

                f.write("\n" + "=" * 100 + "\n")

        return filename

    def benchmark(self):

        results = self.run()

        self.save_json(results)
        self.save_txt(results)

        return results


class BenchmarkMetrics:
    """
    Utility class for computing benchmark metrics
    on generated text.

    If `tokenizer` is None it will uses as default the `Word-Tokenizer`. 
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def prompt_tokens(self, prompt: str) -> int:
        """
        Count prompt tokens.
        """

        if self.tokenizer is not None:
            return len(self.tokenizer.encode(prompt))

        return len(prompt.split())

    def generated_tokens(self, output: str) -> int:
        """
        Count generated output tokens.
        """

        if self.tokenizer is not None:
            return len(self.tokenizer.encode(output))

        return len(output.split())

    def tokens_per_second(
        self,
        output: str,
        generation_time: float
    ) -> float:
        """
        Calculate generated tokens per second.
        """

        if generation_time <= 0:
            return 0.0

        n_tokens = self.generated_tokens(output)

        return round(
            n_tokens / generation_time,
            2
        )

    def peak_gpu_memory_mb(self) -> float:
        """
        Return peak GPU memory usage in MB.
        """

        if not torch.cuda.is_available():
            return 0.0

        peak_bytes = torch.cuda.max_memory_allocated()

        return round(
            peak_bytes / (1024 ** 2),
            2
        )

    def repetition_score(
        self,
        text: str,
        ngram_size: int = 3
    ) -> float:
        """
        Measure text repetition.

        Returns:
            0.0 = no repetition
            1.0 = highly repetitive
        """

        words = text.lower().split()

        if len(words) < ngram_size:
            return 0.0

        ngrams = [
            tuple(words[i:i + ngram_size])
            for i in range(
                len(words) - ngram_size + 1
            )
        ]

        unique = len(set(ngrams))
        total = len(ngrams)

        return round(
            1 - (unique / total),
            4
        )

    def average_sentence_length(
        self,
        text: str
    ) -> float:
        """
        Average words per sentence.
        """

        sentences = re.split(
            r"[.!?]+",
            text
        )

        sentences = [
            s.strip()
            for s in sentences
            if s.strip()
        ]

        if len(sentences) == 0:
            return 0.0

        total_words = sum(
            len(sentence.split())
            for sentence in sentences
        )

        return round(
            total_words / len(sentences),
            2
        )

    def reset_gpu_memory_stats(self):
        """
        Reset CUDA memory counters before generation.
        """

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def compute(
        self,
        prompt: str,
        output: str,
        generation_time: float
    ) -> dict:
        """
        Compute all metrics.
        """

        return {
            "prompt_tokens":
                self.prompt_tokens(prompt),

            "generated_tokens":
                self.generated_tokens(output),

            "tokens_per_second":
                self.tokens_per_second(
                    output,
                    generation_time
                ),

            "peak_gpu_memory_mb":
                self.peak_gpu_memory_mb(),

            "repetition_score":
                self.repetition_score(output),

            "average_sentence_length":
                self.average_sentence_length(
                    output
                ),
        }
