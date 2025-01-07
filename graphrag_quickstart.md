## Quickstart

### 1. Prepare LLM and Embedding Model

Install ollama software. In a terminal window, download the LLM and embedding model using the following commands:

- For **Windows users**:

  ```
  :: LLM
  ollama pull llama3.2
  :: Embedding model
  ollama pull nomic-embed-text
  ```

### 2. Setup Python Environment for GraphRAG

To run the LLM and embedding model on a local machine, we utilize the [`graphrag-local-ollama`](https://github.com/TheAiSingularity/graphrag-local-ollama) repository:

```shell
git clone https://github.com/TheAiSingularity/graphrag-local-ollama.git
cd graphrag-local-ollama

# conda create -n graphrag-local-ollama python=3.10
# conda activate graphrag-local-ollama

python -m venv .venv
.venv/Scripts/activate

pip install -e .
pip install future ollama plotly
```

### 3. Index GraphRAG

The environment is now ready for GraphRAG with local LLMs and embedding models running on Intel GPUs. Before querying GraphRAG, it is necessary to first index GraphRAG, which could be a resource-intensive operation.

> [!TIP]
> Refer to [here](https://microsoft.github.io/graphrag/) for more details in GraphRAG process explanation.

#### Prepare Input Corpus

Perpare the input corpus, and then initialize the workspace:
For the csv file, I have already combined all other columns into text columns. 
Reference can be found under .\input sample files

- For **Windows users**:

  Please run the following command :

  ```cmd
  :: define inputs corpus
  mkdir ragtest && cd ragtest && mkdir input && cd .. 
  <!-- copy input\* .\ragtest\input -->

  :: initialize ragtest folder
  python -m graphrag.index --init --root .\ragtest

  :: prepare settings.yml, please make sure the initialized settings.yml in ragtest folder is replaced by settings.yml in graphrag-ollama-local folder
  copy settings.yaml .\ragtest /y
  ```

#### Update `settings.yml`

You are required to update the `settings.yml` in `ragtest` folder accordingly:

```yml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: llama3.2 # change it accordingly if using another LLM
  model_supports_json: true
  request_timeout: 1800.0 # add this configuration; you could also increase the request_timeout
  api_base: http://localhost:11434/v1

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: nomic_embed_text # change it accordingly if using another embedding model
    api_base: http://localhost:11434/api
   
input:
  type: file # or blob
  file_type: csv # or csv
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.csv$"

```

#### Conduct GraphRAG indexing

Finally, conduct GraphRAG indexing, which may take a while:

```shell
python -m graphrag.index --root ragtest
```

You will got message `🚀 All workflows completed successfully.` after the GraphRAG indexing is successfully finished.

#### (Optional) Visualize Knowledge Graph

For a clearer view of the knowledge graph, you can visualize it by specifying the path to the `.graphml` file in the `visualize-graphml.py` script, like below:

- For **Windows users**:

  ```python
  graph = nx.read_graphml('ragtest\\output\\20240715-151518\\artifacts\\summarized_graph.graphml') 
  ```

and run the following command to interactively visualize the knowledge graph:

```shell
python visualize-graphml.py
```

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_1.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_1.png"/></a></td>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_2.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_2.png"/></a></td>
</tr>
</table>

### 5. Query GraphRAG

After the GraphRAG has been successfully indexed, you could conduct query based on the knowledge graph through:

```shell
python -m graphrag.query --root ragtest --method global "What is Transformer?"
```

> [!NOTE]
> Only the `global` query method is supported for now.

The sample output looks like:

```log
INFO: Reading settings from ragtest\settings.yaml
creating llm client with {'api_key': 'REDACTED,len=9', 'type': "openai_chat", 'model': 'mistral', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'request_timeout': 1800.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}

SUCCESS: Global Search Response:  The Transformer is a type of neural network architecture primarily used for sequence-to-sequence tasks, such as machine translation and text summarization [Data: Reports (1, 2, 34, 46, 64, +more)]. It was introduced in the paper 'Attention is All You Need' by Vaswani et al. The Transformer model uses self-attention mechanisms to focus on relevant parts of the input sequence when generating an output.

The Transformer architecture was designed to overcome some limitations of Recurrent Neural Networks (RNNs), such as the vanishing gradient problem and the need for long sequences to be processed sequentially [Data: Reports (1, 2)]. The Transformer model processes input sequences in parallel, making it more efficient for handling long sequences.

The Transformer model has been very successful in various natural language processing tasks, such as machine translation and text summarization [Data: Reports (1, 34, 46, 64, +more)]. It has also been applied to other domains, such as computer vision and speech recognition. The key components of the Transformer architecture include self-attention layers, position-wise feedforward networks, and multi-head attention mechanisms [Data: Reports (1, 2, 3)].

Since its initial introduction, the Transformer model has been further developed and improved upon. Variants of the Transformer architecture, such as BERT (Bidirectional Encoder Representations from Transformers) and RoBERTa (Robustly Optimized BERT Pretraining Approach), have achieved state-of-the-art performance on a wide range of natural language processing tasks [Data: Reports (1, 2, 34, 46, 64, +more)].
```