## TravoAI (E-Commerce Chatbot)

An AI-powered conversational chatbot for e-commerce queries built with **Streamlit**, **Groq LLM**, **ChromaDB**, and **Semantic Router**. The bot handles two types of queries — frequently asked questions (FAQ) and product search queries via natural language to SQL conversion.

🔗 **Live Demo:** https://travoai.streamlit.app/

---

 ## App Preview

<img width="927" height="707" alt="image" src="https://github.com/user-attachments/assets/edb61d95-aa1b-496c-acd4-8e03b2788bca" />


> The chat input lets you ask anything — from store policies to product searches. FAQ queries are answered instantly from the knowledge base, while product queries are executed against the database and displayed as natural language responses. make it short
---

## Features

- **Semantic Routing** — Automatically classifies user queries into FAQ or product search using `semantic-router` with a HuggingFace encoder.
- **FAQ Retrieval (RAG)** — Uses ChromaDB with sentence-transformer embeddings to retrieve relevant Q&A pairs and generate grounded answers.
- **Natural Language to SQL** — Converts product-related questions into SQL queries and fetches results from a local SQLite database.
- **Conversational UI** — Clean Streamlit-based chat interface with message history.

---

## Architecture
<img width="564" height="675" alt="image" src="https://github.com/user-attachments/assets/f36305bc-bc55-4193-8cf5-8c07b455b811" />

---

## Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Groq ( `llama3-8b-8192` ) |
| **Embeddings** | HuggingFace ( `all-MiniLM-L6-v2` ) |
| **Vector Store** | ChromaDB |
| **Routing** | Semantic Router |
| **Database** | SQLite |
| **Frontend** | Streamlit |

---

## Database Schema

The SQLite database contains a `product` table with the following fields:

| Column | Type | Description |
|---|---|---|
| `product_link` | string | Hyperlink to the product |
| `title` | string | Name of the product |
| `brand` | string | Brand of the product |
| `price` | integer | Price in Indian Rupees (₹) |
| `discount` | float | Discount (e.g., `0.1` = 10%) |
| `avg_rating` | float | Average rating (0–5) |
| `total_ratings` | integer | Total number of ratings |
---

## Setup & Installation
## 1. Clone the repository
```bash
git clone https://github.com/Gari1428/E-Commerce-Chatbot
cd ecommerce-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
```

### 4. Run the app
```bash
streamlit run main.py
```

---

##  Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Chat UI |
| `groq` | LLM inference |
| `chromadb` | Vector store for FAQ RAG |
| `sentence-transformers` | Text embeddings |
| `semantic-router` | Query routing |
| `pandas` | Data handling |
| `sqlite3` | Product database |
| `python-dotenv` | Environment config |

---

##  Example Queries

**FAQ Queries**
- *"What is your return policy?"*

**Product Search Queries**
- *"Show me Nike shoes under ₹3000"*

---

##  How It Works

### Routing
The `SemanticRouter` in `router.py` uses sentence-transformer embeddings to match the user's query against predefined utterances for each route (`faq` or `sql`).

### FAQ Chain (`faq.py`)
1. On first run, ingests `faqs.csv` into a ChromaDB collection using `all-MiniLM-L6-v2` embeddings.
2. At query time, retrieves the top 2 most similar FAQ entries.
3. Passes retrieved answers as context to the Groq LLM for a grounded response.

### SQL Chain (`sql.py`)
1. Sends the user's question along with the database schema to the Groq LLM.
2. Extracts the generated SQL query from `<SQL></SQL>` tags.
3. Executes the query against the SQLite database.
4. Passes the results back to the Groq LLM for a natural language response.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Groq](https://groq.com/) for ultra-fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for the local vector store
- [Sentence Transformers](https://www.sbert.net/) for the embedding model
- [Semantic Router](https://github.com/aurelio-labs/semantic-router) for the query routing framework
- [HuggingFace](https://huggingface.co/) for hosting pre-trained transformer models
- [Streamlit](https://streamlit.io/) for the rapid UI framework

---
