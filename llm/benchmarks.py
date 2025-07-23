"""Logic for extracting questions and answers from HuggingFace benchmarks."""

import abc
import datasets
import re


class Benchmark(abc.ABC):
    def __init__(self, name: str, subset: str, split: str):
        self.name = name
        self.subset = subset
        self.split = split

    def get_benchmark_questions(self):
        """Get the questions as they appear in the benchmark results."""
        return [
            self.format_question(instance) for instance in self.get_data_instances()
        ]

    def format_question(self, instance):
        """Format the question to match the Open LLM Leaderboard results."""
        return instance["question"]["text"]

    def get_data_instances(self):
        """Get the data instances for this benchmark."""
        return [self.extract_data_instance(row) for row in self.load().to_dicts()]

    @staticmethod
    def format_references(choices, correct):
        """Format references for multiple-choice questions."""
        return [
            {
                "text": choice,
                "tags": ["correct"] if choice == correct else ["incorrect"],
            }
            for choice in choices
        ]

    def load(self):
        """Load the dataset for this benchmark."""
        if isinstance(self.split, str):
            return datasets.load_dataset(
                self.name,
                self.subset,
                split=self.split,
                trust_remote_code=True,
            ).to_polars()
        elif isinstance(self.split, list):
            splits = [
                datasets.load_dataset(
                    self.name,
                    self.subset,
                    split=split,
                    trust_remote_code=True,
                )
                for split in self.split
            ]
            return datasets.concatenate_datasets(splits).to_polars()

    @abc.abstractmethod
    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        pass


class MMLU(Benchmark):
    def __init__(self, subset: str = "all"):
        super().__init__("cais/mmlu", subset, ["test", "validation"])

    def format_question(self, instance):
        if self.subset in [
            "abstract_algebra",
            "college_chemistry",
            "computer_security",
            "econometrics",
            "us_foreign_policy",
        ]:
            return instance["question"]["text"]

        question = instance["question"]["text"]
        labels = ["A", "B", "C", "D"]
        references = [reference["text"] for reference in instance["references"]]
        answers = "\n".join(
            [f"{label}. {reference}" for label, reference in zip(labels, references)]
        )
        return f"{question}\n{answers}"

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""

        return {
            "question": {"text": row["question"]},
            "references": [
                {
                    "text": text,
                    "tags": ["correct"] if i == row["answer"] else [],
                }
                for i, text in enumerate(row["choices"])
            ],
            "metadata": {"subject": row["subject"], "source": self.name},
        }


class ARC(Benchmark):
    def __init__(self):
        super().__init__("allenai/ai2_arc", "ARC-Challenge", "test")

    def format_question(self, instance):
        """Format the question to match the Open LLM Leaderboard results."""
        return f"Question: {instance['question']['text']}"

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        return {
            "question": {"text": row["question"]},
            "references": [
                {
                    "text": text,
                    "tags": ["correct"] if label == row["answerKey"] else [],
                }
                for text, label in zip(row["choices"]["text"], row["choices"]["label"])
            ],
            "metadata": {"source": self.name},
        }


class Winogrande(Benchmark):
    def __init__(self):
        super().__init__("allenai/winogrande", "winogrande_xl", "validation")

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""

        return {
            "question": {"text": row["sentence"]},
            "references": [
                {
                    "text": row["option1"],
                    "tags": ["correct"] if row["answer"] == 1 else [],
                },
                {
                    "text": row["option2"],
                    "tags": ["correct"] if row["answer"] == 2 else [],
                },
            ],
            "metadata": {"source": self.name},
        }


class Hellaswag(Benchmark):
    def __init__(self):
        super().__init__("Rowan/hellaswag", "default", "validation")

    def format_question(self, instance):
        subject = instance["metadata"]["subject"]
        question = Hellaswag.strip_tags(instance["question"]["text"])
        return f"{subject}: {question}"

    @staticmethod
    def strip_tags(text):
        """Remove wikihow tags from Hellaswag questions."""
        pattern = r"\[header\]|\[step\]|\[title\]|\[substeps\]"
        text = re.sub(pattern, "", text)
        return " ".join(text.split())

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""

        return {
            "question": {"text": row["ctx"]},
            "references": [
                {
                    "text": text,
                    "tags": ["correct"] if i == int(row["label"]) else [],
                }
                for i, text in enumerate(row["endings"])
            ],
            "metadata": {"subject": row["activity_label"], "source": self.name},
        }


class GSM8K(Benchmark):
    def __init__(self):
        super().__init__("gsm8k", "main", "test")

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        answer = row["answer"]
        reference_short_match = re.search(r"####\s*(.*)", answer)
        reference_short = (
            reference_short_match.group(1).strip() if reference_short_match else ""
        )
        reference_annotated = re.sub(r"####\s*.*", "", answer).strip()
        reference = re.sub(r"<<.*?>>", "", reference_annotated).strip()

        return {
            "question": {"text": row["question"]},
            "references": [
                {
                    "text": reference,
                    "tags": ["correct"],
                },
                {
                    "text": reference_annotated,
                    "tags": ["correct", "annotated"],
                },
                {
                    "text": reference_short,
                    "tags": ["correct", "short"],
                },
            ],
            "metadata": {"source": self.name},
        }


class TruthfulQA(Benchmark):
    def __init__(self):
        super().__init__("truthful_qa", "generation", "validation")

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        references = [
            {"text": row["best_answer"], "tags": ["correct", "best"]},
        ]
        references.extend(
            [
                {"text": text, "tags": ["correct"]}
                for text in row["correct_answers"]
                if text != row["best_answer"]
            ]
        )
        references.extend(
            [{"text": text, "tags": ["incorrect"]} for text in row["incorrect_answers"]]
        )

        return {
            "question": {"text": row["question"]},
            "references": references,
            "metadata": {
                "subject": row["category"],
                "type": row["type"],
                "url": row["source"],
                "source": self.name,
            },
        }


class NarrativeQA(Benchmark):
    def __init__(self):
        super().__init__("deepmind/narrativeqa", "default", ["test", "validation"])

    def load(self):
        """Load the dataset for this benchmark."""
        if isinstance(self.split, str):
            return (
                datasets.load_dataset(
                    self.name,
                    self.subset,
                    split=self.split,
                    trust_remote_code=True,
                )
                .map(NarrativeQA.strip_full_document, load_from_cache_file=False)
                .to_polars()
            )

        elif isinstance(self.split, list):
            splits = [
                datasets.load_dataset(
                    self.name,
                    self.subset,
                    split=split,
                    trust_remote_code=True,
                )
                for split in self.split
            ]
            return (
                datasets.concatenate_datasets(splits)
                .map(NarrativeQA.strip_full_document, load_from_cache_file=False)
                .to_polars()
            )

    @staticmethod
    def strip_full_document(row):
        row["document"] = {
            "id": row["document"]["id"],
            "kind": row["document"]["kind"],
            "url": row["document"]["url"],
            "summary": row["document"]["summary"]["text"],
        }
        return row

    def format_question(self, instance):
        """Format the question to match the Open LLM Leaderboard results."""
        question = instance["question"]["text"]
        summary = instance["question"]["document"]["summary"]
        return f"{summary}\n{question}"

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        references = [
            {"text": reference["text"], "tags": ["correct"]}
            for reference in row["answers"]
        ]

        return {
            "question": {
                "text": row["question"]["text"],
                "document": {
                    "id": row["document"]["id"],
                    "url": row["document"]["url"],
                    "kind": row["document"]["kind"],
                    "summary": row["document"]["summary"],
                },
            },
            "references": references,
            "metadata": {
                "type": row["document"]["kind"],
                "source": self.name,
            },
        }


class NaturalQA(Benchmark):
    def __init__(self, variant):
        self.variant = variant
        super().__init__("lighteval/natural_questions_helm", "default", "validation")

    def format_question(self, instance):
        """Format the question to match the Open LLM Leaderboard results."""
        question = instance["question"]["text"]
        if self.variant == "closedbook":
            return question
        elif self.variant == "openbook_longans":
            passage = instance["question"]["document"]["text"]
            return f"Passage: {passage}\n\nQuestion: {question}"
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        references = [
            {"text": answer, "tags": ["correct", "short"]}
            for answer in set(row["short_answers"])
        ]
        long_references = [
            {"text": answer, "tags": ["correct", "long"]}
            for answer in set(row["long_answers"])
        ]

        if self.variant == "openbook_longans":
            data_instances = [
                {
                    "question": {
                        "text": row["question"],
                        "document": {"text": long_reference["text"]},
                    },
                    "references": references + long_references,
                    "metadata": {"source": self.name, "topic": row["title"]},
                }
                for long_reference in long_references
            ]
        elif self.variant == "closedbook":
            data_instances = [
                {
                    "question": {"text": row["question"]},
                    "references": references + long_references,
                    "metadata": {"source": self.name, "topic": row["title"]},
                }
            ]
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        return data_instances

    def get_data_instances(self):
        """Get the data instances for this benchmark."""
        return [
            instance
            for row in self.load().to_dicts()
            for instance in self.extract_data_instance(row)
        ]


class OpenBookQA(Benchmark):
    def __init__(self):
        super().__init__("allenai/openbookqa", "additional", "test")

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        references = [
            {
                "text": reference,
                "tags": ["correct"] if label == row["answerKey"] else ["incorrect"],
            }
            for reference, label in zip(row["choices"]["text"], row["choices"]["label"])
        ]
        return {
            "question": {
                "text": row["question_stem"],
                "document": {"text": row["fact1"]},
            },
            "references": references,
            "metadata": {
                "dataset_id": row["id"],
                "human_score": row["humanScore"],
                "turk_id_anonymized": row["turkIdAnonymized"],
                "clarity": row["clarity"],
                "source": self.name,
            },
        }


class LegalBench(Benchmark):
    def __init__(self, subset):
        super().__init__("nguha/legalbench", subset, "test")

    def format_question(self, instance):
        """Format the question to match the Open LLM Leaderboard results."""
        if self.subset == "abercrombie":
            question = instance["question"]["text"]
            return f"Description: {question}"
        else:
            return instance["question"]["text"]

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        if self.subset == "abercrombie":
            choices = ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]
            return {
                "question": {
                    "text": row["text"],
                },
                "references": self.format_references(choices, row["answer"]),
                "metadata": {"source": self.name},
            }
        if self.subset == "corporate_lobbying":
            choices = ["Yes", "No"]
            return {
                "question": {
                    "text": (
                        f"Official title of bill: {row['bill_title']}\n"
                        f"Official summary of bill: {row['bill_summary']}\n"
                        f"Company name: {row['company_name']}\n"
                        f"Company business description: {row['company_description']}"
                    ),
                },
                "references": self.format_references(choices, row["answer"]),
                "metadata": {"source": self.name},
            }
        if self.subset == "function_of_decision_section":
            choices = [
                "Facts",
                "Procedural History",
                "Issue",
                "Rule",
                "Analysis",
                "Conclusion",
                "Decree",
            ]
            return {
                "question": {
                    "text": f"Text: {row['Paragraph']}",
                },
                "references": self.format_references(choices, row["answer"]),
                "metadata": {"source": self.name, "citation": row["Citation"]},
            }
        if self.subset == "international_citizenship_questions":
            choices = ["Yes", "No"]
            return {
                "question": {
                    "text": row["question"],
                },
                "references": self.format_references(choices, row["answer"]),
                "metadata": {"source": self.name},
            }
        if self.subset == "proa":
            choices = ["Yes", "No"]
            return {
                "question": {
                    "text": row["text"],
                },
                "references": self.format_references(choices, row["answer"]),
                "metadata": {"source": self.name},
            }


class Math(Benchmark):
    def __init__(self):
        super().__init__("lighteval/MATH", "all", "test")

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        return {
            "question": {
                "text": row["problem"],
            },
            "references": [
                {
                    "text": row["solution"],
                    "tags": ["correct"],
                },
            ],
            "metadata": {
                "level": row["level"],
                "subject": row["type"],
                "source": self.name,
            },
        }


class MedQA(Benchmark):
    def __init__(self):
        super().__init__("truehealth/medqa", "default", ["validation", "test"])

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        references = [
            {
                "text": text,
                "tags": ["correct"] if text == row["answer"] else [],
            }
            for label, text in dict(row["options"]).items()
        ]
        return {
            "question": {
                "text": row["question"],
            },
            "references": references,
            "metadata": {
                "meta_info": row["meta_info"],
                "source": self.name,
            },
        }


class WMT14(Benchmark):
    def __init__(self, subset: str):
        self.subset = subset

        super().__init__("wmt/wmt14", subset, ["test", "validation"])

    def extract_data_instance(self, row):
        """Extract the question and answers from a row of the dataset."""
        source, target = self.subset.split("-")
        return {
            "question": {
                "text": row["translation"][source],
            },
            "references": [{"text": row["translation"][target], "tags": ["correct"]}],
            "metadata": {
                "source": self.name,
            },
        }
