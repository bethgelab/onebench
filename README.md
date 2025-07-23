# ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities

An ever-evolving benchmark for LLMs and LMMs.

## Installation

(Recommended) Create a new virtual environment and activate it. Some packages require Python>=3.11, therefore we suggest using the following:

```bash
conda create -n onebench python=3.11 -y
conda activate onebench
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Install ONEBench in editable mode:

```bash
python -m pip install -e .
```

Test the installation:

```bash
python -c "import onebench"
```

## Downloading the data

### LLM

#### HELM

[Optional] Upgrade the Google Cloud SDK:

```bash
brew install python@3.11
export CLOUDSDK_PYTHON=$(which python3.11)
gcloud components update
```

Authenticate to Google Cloud:

```bash
gcloud init
```

Download the HELM data:

```bash
python llm/download_helm.py
```

#### Open LLM Leaderboard

Download the Open LLM Leaderboard data:

```bash
python llm/download_open_llm_leaderboard.py
```

#### Chatbot Arena

Download the LMSYS Chatbot Arena data:

```bash
python llm/download_chatbot_arena.py
```

### VLM

The VLM results are in the `data/vlm/{dataset}` directory, where dataset corresponds to `vhelm` and `lmms-eval`. The individual dataset a-matrices are located in `data/vlm/{dataset}/binary` and  `data/vlm/{dataset}/numeric`. The results from Prometheus2 are located in `data/vlm/{dataset}/pairwise_num`. 

[TODO]: Add instructions for json downloads, a matrix creation, prometheus scripts and capability querying.


## 📚Citation
### If you find our work helpful, please use the following citation:
```
@inprocessings{ghosh2025onebench,
  title={ONEBench to test them all: Sample-level benchmarking over open-ended capabilities},
  author={Ghosh, Adhiraj and Dziadzio, Sebastian and Prabhu, Ameya and Udandarao, Vishaal and Albanie, Samuel and Bethge, Matthias},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics },
  year={2025}
}
```

## 🪪 License <a name="license"></a>
Code: MIT. Check `LICENSE`.