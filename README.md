# albcovis-project

ensure keys in .env

to create the env
```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

to install albcovis package as editable install:
```
pip install -e .
```

for docker:
```
docker build -t cover-api .
docker run -p 8000:8000 cover-api
```


# albcovis-project

**Album Cover Visual Features API**  

This project provides tools and an API to extract metadata and visual features from album cover art.  
It integrates **MusicBrainz**, **Discogs**, and the **Cover Art Archive** with computer vision techniques (color extraction, texture descriptors, face and text detection).  

---

## 🚀 Setup

### 1. Environment variables
Make sure to create a `.env` file in the project root with the required API keys and settings:
```bash
DISCOGS_TOKEN=your_discogs_token
USER_AGENT="albcovis-project/0.1 (contact@example.com)"
````

### 2. Local development

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

Now you can use the modules (e.g. `from albcovis.services.cover_data_aggregator import CoverDataAggregator`)

---

### 3. Running with Docker

Build the Docker image:

```bash
docker build -t cover-api .
```

Run the container:

```bash
docker run -p 8000:8000 cover-api
```

The API will be available at:
👉 [http://localhost:8000](http://localhost:8000)

OpenAPI docs:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Repository Structure

```
.
├── data
│   ├── processed                # computed feature data (color palettes, face/text detections, etc.)
│   └── source                   # Raw datasets and images
├── notebooks                    # Jupyter notebooks for exploration & experiments
├── scripts                      # Utility scripts to build and analyze datasets
├── src/albcovis
│   ├── api                      # FastAPI app entrypoints and routers
│   ├── models                   # Pydantic data models
│   ├── services                 # API clients, cover data aggregator, feature extractors
│   ├── utils                    # Helper functions (HTTP, image utils, rate limiter, etc.)
│   └── settings.py              # Configuration and environment handling
├── pyproject.toml               # Package metadata
├── requirements.txt             # Dependencies
├── requirements_dev.txt         # Dev dependencies
└── README.md
```

---

## ⚡ Features

* **Metadata aggregation** from MusicBrainz, Discogs, and Cover Art Archive
* **Visual features extraction**:

  * Color palettes
  * Texture descriptors
  * Face detection
  * Text detection
* **REST API** with FastAPI + Docker
* Rate limiting to respect external APIs
* Dataset builder scripts (`analysis_set`, `dev_set`)

---

## 🛠️ Endpoints

* `GET /cover/metadata?mbid=...` → returns structured metadata
* `GET /cover/features?mbid=...` → returns metadata **+ visual features**
