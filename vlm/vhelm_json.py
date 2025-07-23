import requests
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Saving JSON files for VLMs from the vHELM website")
    parser.add_argument(
        "--save_root", default="data/vhelm",
        help="root directory to save json files", type=str)

    parser.add_argument("--dataset",
                        default='mmmu',
                        help="Which dataset to use. Refer to the README for more", type=str)

    args = parser.parse_args()

    if args.dataset == "mmmu":
        categories = ["Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
                      "Chemistry", "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
                      "Electronics", "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials",
                      "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology"]

    else:
        categories = [""]  # No category required in the url

    # List of valid models
    models = ["anthropic_claude-3-sonnet-20240229", "anthropic_claude-3-opus-20240229", "google_gemini-1.0-pro-vision-001",
              "google_gemini-1.5-pro-preview-0409", "HuggingFaceM4_idefics2-8b", "openai_gpt-4-vision-preview"]

    save_dir = os.path.join(args.save_root, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for category in categories:
        for model in models:
            prompt_dir = os.path.join(save_dir, model, 'prompt')
            score_dir = os.path.join(save_dir, model, 'score')
            [os.makedirs(directory, exist_ok=True) for directory in [prompt_dir, score_dir]]

            if args.dataset == "mmmu":
                print(f"Saving prompts and score for {category} in {args.dataset} and {model}")
                prompt_file = os.path.join(prompt_dir, category.lower() + '.json')
                score_file = os.path.join(score_dir, category.lower() + '.json')

                prompt_url = f"https://storage.googleapis.com/crfm-helm-public/vhelm/benchmark_output/runs/v1.0.0/{args.dataset}:subject={category},question_type=multiple-choice,model={model}/instances.json"
                score_url = f"https://storage.googleapis.com/crfm-helm-public/vhelm/benchmark_output/runs/v1.0.0/{args.dataset}:subject={category},question_type=multiple-choice,model={model}/per_instance_stats.json"
            else:
                print(f"Saving prompts and score for {args.dataset} and {model}")
                prompt_file = os.path.join(prompt_dir, 'promptout.json')
                score_file = os.path.join(score_dir, 'scoreout.json')

                if args.dataset == 'viz_wiz':
                    prompt_url = f"https://storage.googleapis.com/crfm-helm-public/vhelm/benchmark_output/runs/v1.0.0/{args.dataset}:model={model}/instances.json"
                    score_url = f"https://storage.googleapis.com/crfm-helm-public/vhelm/benchmark_output/runs/v1.0.0/{args.dataset}:model={model}/per_instance_stats.json"

                elif args.dataset == 'vqa':
                    prompt_url = f"https://storage.googleapis.com/crfm-helm-public/vhelm/benchmark_output/runs/v1.0.0/{args.dataset}:model={model},groups=vqa_base/instances.json"
                    score_url = f"https://storage.googleapis.com/crfm-helm-public/vhelm/benchmark_output/runs/v1.0.0/{args.dataset}:model={model},groups=vqa_base/per_instance_stats.json"

            if os.path.exists(prompt_file) and os.path.exists(score_file):
                print(f"Prompt and score files already exist for {model}")
                continue
            # Send a GET request to the URL to download the JSON file

            response_prompt = requests.get(prompt_url)
            response_prompt.raise_for_status()  # Raise an exception for invalid HTTP responses

            # Send a GET request to the URL to download the JSON file
            response_score = requests.get(score_url)
            response_score.raise_for_status()  # Raise an exception for invalid HTTP responses


            with open(prompt_file, "wb") as f:
                f.write(response_prompt.content)

            with open(score_file, "wb") as f:
                f.write(response_score.content)

if __name__ == '__main__':
    main()