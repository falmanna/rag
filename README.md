
## Installation

### Install vector store
```
docker compose up -d
```
### Install Ollama using docker
NOTE: installing from docker compose might break compatibility with gpu.
```
docker run -e OLLAMA_FLASH_ATTENTION=1 -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
## Environemnt setup
Copy example `.env` and edit it as you per your preferences.
Default settings are optimized for running locally with good performance.
```
cp .env.example .env
```
Create `.venv`
```
python -m venv .venv
source ./venv/bin/activate
```
Install dependencies
```
poetry lock
poetry install
```
## Ingestion
Run the ingestion script to chunk and index the dataset into the vector store.
```
python ingestion.py
```
> Ingesting the dataset takes about 4 hours depending on the machine and configs.
## Running the graph

### Run the graph
To test the graph without a UI, add a question to `main.py` and run the script.

Running the graph will generate a `graph.png` that shows the overall graph.

### Run the UI
To run the UI, run the following command
```
chainlit run app.py
```
This will spin up a chainlit app that you can use to configure and run the graph from the UI.

## Configs
### LLM
After testing multiple models, I choose LLAMA3.1 as it has produced the best results with the most successful rate.

Current implementation requires a model that supports native tool calling in ollama, other models were tested using a less accurate prompt instruction techniques.

> Most local models will struggle with returning structured output when the machine has limited resources and when receiving large context, which is the case in this graph.

It is also possible to test the graph with openAI or even extended to any other supported provider.

### Embedding
The selected embedding for ingestion and runtime must has the same dimension.

### Graph
In the UI next to the chatbox, clicking on the settings button will open an advanced configs menu for configuring some advanced but experimental features in the graph.

Enabling any feature will increase latency and might crash the graph when running locally with a small model. But they help produce better results when running with a large strong model.