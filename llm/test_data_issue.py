import datasets
import transformers


def _main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    org, model = model_name.split("/")

    data = datasets.load_dataset(
        f"open-llm-leaderboard/details_{org}__{model}",
        name="harness_winogrande_5",
        split="latest",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    df = data.to_pandas()
    first_row = df.iloc[0]

    data = {
        "example": first_row["example"],
        "full_prompt": first_row["full_prompt"],
        "input_tokens_a": first_row["input_tokens"][0],
        "input_tokens_b": first_row["input_tokens"][1],
        "cont_tokens_a": first_row["cont_tokens"][0],
        "cont_tokens_b": first_row["cont_tokens"][1],
    }
    for key, value in data.items():
        if "tokens" in key:
            value = tokenizer.decode(value)
        print(f"{key}:\n {value}\n\n\n")


if __name__ == "__main__":
    _main()
