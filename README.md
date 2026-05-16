# Talking to Air

**Talking to Air** is an AI-powered conversational agent for querying, visualizing, and analyzing atmospheric data from NASA satellite missions. Users can ask natural language questions about air quality and receive maps, trend plots, and statistical summaries derived from real satellite observations.

The system features persistent conversation history, intelligent agent routing, multi-variable comparisons, and caching for fast subsequent queries.

---

## Overview

The system combines a LangChain-based agentic backend with a React frontend to provide an interactive interface for exploring atmospheric column data across multiple NASA datasets:

| Dataset | Sensor | Variable | Temporal Resolution | Coverage |
|---|---|---|---|---|
| `OMI_NO2` | OMI/Aura | NO₂ | Daily | Global |
| `TROPOMI_NO2` | Sentinel-5P | NO₂ | Monthly | Global |
| `TEMPO_NO2` | TEMPO | NO₂ | Hourly | North America |
| `TEMPO_O3TOT` | TEMPO | O₃ | Hourly | North America |
| `OMI_O3` | OMI/Aura | O₃ | Daily | Global |
| `TEMPO_HCHO` | TEMPO | HCHO | Hourly | North America |
| `OMI_HCHO` | OMI/Aura | HCHO | Daily | Global |

Data is fetched on demand from NASA Harmony and cached locally in Zarr format to avoid redundant downloads. Conversation state is persisted in PostgreSQL for multi-turn interactions.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Google AI Studio API key](https://ai.google.dev/)
- [NASA Earthdata account](https://urs.earthdata.nasa.gov/)

---

## Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/talking-to-air.git
   cd talking-to-air
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Fill in `.env` with your API keys (see `.env.example` for instructions)

3. **Start with Docker:**
   ```bash
   docker compose up --build
   ```

4. **Access the app:**
   - Frontend: http://localhost:5173
   - API docs: http://localhost:8000/docs

---

## Usage Guide

### Basic Queries

The agent understands natural language queries about atmospheric data. Here are examples:

**Single location queries:**
```
Plot NO2 levels in Texas on April 8, 2024
What was the mean NO2 in Los Angeles in March 2024?
Show the NO2 trend over Paris for the last 18 months
Where was NO2 highest in California today?
```

**Comparative queries:**
```
Compare NO2 between California and New York
Show formaldehyde levels in London vs Tokyo
How does O3 in Texas compare to Florida?
```

**Time-series queries:**
```
Show the NO2 trend over Greece for the last 18 months
Plot HCHO levels over Florida over the past year
How has ozone changed in New York since January?
```

**Peak value queries:**
```
Where was the highest NO2 reading in Texas yesterday?
Find the peak formaldehyde location in Europe last month
```

### How to Use the Interface

1. **Chat Window:** Type your question in the text input at the bottom
2. **Images:** Generated maps and plots are displayed as clickable lightbox images
3. **Sessions:** Use the left sidebar to create new conversations or switch between previous ones
4. **Tool Calls:** Yellow badges show which tools the agent used to answer your question
5. **Conversation History:** All messages and results are saved and persistent

### Important Constraints

- **TEMPO datasets** (hourly data) only cover **North America**. For other regions, use OMI or TROPOMI data
- **TROPOMI_NO2** has **monthly resolution only** — cannot fetch single-day data
- **Date parsing:** Use natural dates like "March 2024" or "last 18 months" or ISO format "2024-03-15"
- **Large regions:** Queries over very large areas may take longer
