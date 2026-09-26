# WineGPT 🍷

A simple AI-powered wine recommendation assistant built with FastAPI, ChromaDB, embeddings, and an LLM.

## Features

- Wine recommendations
- Food pairing suggestions
- Wine comparisons
- Semantic search with vector embeddings
- Retrieval-Augmented Generation (RAG)
- Gradient Lab: interactive gradient descent visualizer + a from-scratch numpy classifier

## Tech Stack

- Python
- FastAPI
- ChromaDB
- LangChain
- Sentence Transformers
- OpenAI API

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Build the vector database:

```bash
python rag/vectorstore.py
```

Run the API:

```bash
uvicorn app:app --reload
```

Visit:

```text
http://127.0.0.1:8000/docs
```

## Gradient Lab 📉

An interactive look at how models learn. Open `lab/index.html` directly in a browser (no server needed), or visit `/lab` while the API is running.

- **Optimizer race**: SGD, Momentum, RMSProp and Adam race down classic loss surfaces (Rosenbrock, Himmelblau, a saddle point, a bumpy bowl full of local minima). Click to pick a start point, crank the learning rate until things diverge, or switch to the 3D view and drag to orbit.
- **Training on the cellar**: a softmax regression learns to classify the 40 wines in `data/wines.csv` as red, white, rosé or sparkling. Watch the decision boundaries bend and the loss fall in real time, and hover any bottle to see the model's confidence.

The same math in numpy (no ML frameworks, gradients derived by hand):

```bash
python -m ml.train --race                              # compare all four optimizers
python -m ml.train --optimizer momentum --lr 0.1 --epochs 500
python -m ml.visualize --surface bumpy                  # animated matplotlib dashboard
python -m ml.visualize --save lab.gif                   # export it as a GIF
```

## Example Endpoints

```http
GET /recommend?query=wine+for+steak

GET /pairing?food=lobster

GET /compare?wine_a=Pinot+Noir&wine_b=Cabernet+Sauvignon
```

## Disclaimer

This is a demonstration project created for portfolio and social media purposes only. It is not a real production application and does not represent any proprietary, client, employer, or business-critical systems. The implementation is intentionally simplified to showcase AI engineering concepts.