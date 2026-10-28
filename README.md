# SWT-Project

This is our project for the Semantic Web Technology course. We use stories from the Nederlandse Verhalenbank to answer questions with a local language model.

The project has two approaches:

- An RDF agent that uses SPARQL queries to find information in a graph built from the stories.
- A RAG pipeline that retrieves stories using BM25 and embeddings, reranks them, and gives them to the model as context.

Part of the RAG code comes from an earlier Language Technology Practical project from my Bachelor AI.

## Setup

Use Python 3.11 and run the commands from the project root.

Install Ollama and make sure it is running. The model is set in `source/config.py` as `GENERATION_MODEL`. The current setting is `gemma4:12b`:

```powershell
ollama pull gemma4:12b
```

If you change the model, also update `GENERATION_MODEL`. The RDF agent uses tool calling and enables thinking, so the model needs to support both. The embedding, reranking, and BERTScore models are downloaded automatically when first used.

## Data

The story data is not included in Git. Place the 6 json files from the Verhalenbank dataset in the data subfolder.

To create the subset, run:

```powershell
python source/create_subset.py
```

The script selects up to 1,000 public Dutch stories with a title and 100–500 words, cleans the text and removes duplicate texts.
The evaluation questions and reference answers are in `resources/test.json`. Keep the stories used by these questions in the subset (seed =64).

## Run

To evaluate the system on the questions in `resources/test.json`:

```powershell
python source/main.py
```
`config.py` contains the configuration, models and what mode you are evaluating all live in here.

`main.py` always runs evaluation.

## Main files

- `source/config.py`: paths, models, and settings.
- `source/create_subset.py`: prepares the story subset.
- `source/RDF_Agent.py`: creates the RDF graph and handles SPARQL tool calls.
- `source/data_processing.py`: loads stories and manages the vector index.
- `source/retrieval.py`: retrieves and reranks stories for RAG.
- `source/generation.py`: generates RAG or LLM-only baseline answers.
- `source/evaluation.py`: evaluates answers and writes the results.
- `source/main.py`: starts evaluation for the selected approach.
