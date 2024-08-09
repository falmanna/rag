
## Installation
This demo supports multiple Vectorstores, LLMs providers, Embedding providers, and ranking providers.

The default configs in `.env.example` are the recommended ones based on the performance, and the task requirements.

If you face any problems with setting up any dependency, please switch to a different one and report the problem.

> The system was tested usnig ubuntu

### Environemnt setup
Copy example `.env.example` and edit it as per your preferences.
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
> Optionally setup a free langsmith account to debug the agent in details.
### Install dependencies
Based on the selected system configs, run the required dependencies using docker compose profiles.
```
docker compose --profile xxx --profile xxx up -d --build
```
> If you face GPU compatibility issues with any dependency please head to the official installation guide of that dependency and install it on your system.

> Infinity will take some time to download the embeddings on startup.

## Ingestion
Run the ingestion script to chunk and index the dataset into the vector store.
```
python ingestion.py
```
> Ingesting the dataset takes about 4 hours (laptop CPU i7, GPU 3070ti), performance may vary depending on the machine and configs.
## Running the Agent

### Run the Agent
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
After testing multiple local models, I choose `LLAMA3.1` as it has produced the best results with the most successful rate.

Current implementation requires a model that supports **native tool calling** in ollama, other models were tested using a less accurate prompt instruction techniques.

> Most local models will struggle with returning structured output when the machine has limited resources and when receiving large context, which is the case in this graph.

It is also possible to test the graph with cloud providers for better performance and results.

### Embedding
The selected embedding for ingestion and runtime must has the same dimension.

### Graph Settings
In the UI next to the chatbox, clicking on the settings button will open an advanced configs menu for configuring some advanced but experimental features in the graph.

Enabling any feature will increase latency and might crash the graph when running locally with a small model. But they help produce better results when running with a large strong model.