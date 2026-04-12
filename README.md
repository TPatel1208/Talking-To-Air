# Talking to Air

**Talking to Air** is an AI-powered conversational agent for querying, visualizing, and analyzing atmospheric data from NASA satellite missions. Users can ask natural language questions about air quality and receive maps, trend plots, and statistical summaries derived from real satellite observations.

---

## Overview

The system combines a LangChain-based agentic backend with a React frontend to provide an interactive interface for exploring atmospheric column data across multiple NASA datasets:

| Dataset | Sensor | Variable | Temporal Resolution | Coverage |
|---|---|---|---|---|
| `OMI_NO2` | OMI/Aura | NO₂ | Daily | Global |
| `TROPOMI_NO2` | Sentinel-5P | NO₂ | Monthly | Global |
| `TEMPO_NO2` | TEMPO | NO₂ | Hourly | North America |
| `TEMPO_HCHO` | TEMPO | HCHO | Hourly | North America |
| `OMI_HCHO` | OMI/Aura | HCHO | Daily | Global |

Data is fetched on demand from NASA Harmony and cached locally in Zarr format to avoid redundant downloads.

---

## Architecture

```
Talking to Air/
├── Backend/
│   ├── api.py                  # FastAPI server
│   ├── GemeniAgent.py          # LangChain agent (Gemini LLM)
│   ├── tools/                  # Agent tools
│   │   ├── harmony_api.py      # NASA Harmony fetch + geocoding
│   │   ├── plot_tools.py       # Map and comparison plotting
│   │   ├── stat_tools.py       # Statistics, trends, peak detection
│   │   └── date_tools.py       # Date parsing utilities
│   ├── preprocessing/
│   │   └── data_loader.py      # Harmony client + Zarr caching
│   └── utils/
│       ├── plotting.py         # Cartopy map rendering
│       └── data_utils.py       # DataArray loading and normalization
└── Frontend/
    └── src/
        ├── components/         # Chat, Dashboard, ImageViewer
        └── hooks/useChat.js    # API state management
```

---

## Prerequisites

- Python 3.10+ (via Anaconda/Miniconda)
- Node.js 22+
- [Google AI Studio API key](https://ai.google.dev/)
- [NASA Earthdata account](https://urs.earthdata.nasa.gov/)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/talking-to-air.git
cd talking-to-air
```

### 2. Backend

```bash
cd Backend
pip install -r requirements.txt
```

Create a `.env` file in `Backend/`:

```
GOOGLE_API_KEY=your_google_api_key
EDL_USERNAME=your_nasa_earthdata_username
EDL_PASSWORD=your_nasa_earthdata_password
```

### 3. Frontend

```bash
cd Frontend
npm install
```

---

## Running

From the project root, run:

```bash
start.bat
```

This opens two terminal windows (Backend and Frontend) and launches the UI at `http://localhost:5173`.

Alternatively, start each server manually:

```bash
# Backend
cd Backend
uvicorn api:app --reload --port 8000

# Frontend
cd Frontend
npm run dev
```

---

## Example Queries

```
Plot NO2 levels in Texas on April 8, 2024
Compare NO2 between California and New York
Show the NO2 trend over Greece for the last 18 months
What was the mean NO2 in Los Angeles in March 2024?
Where was NO2 highest in Texas today?
Plot HCHO over Florida this morning
Compare formaldehyde levels between Texas and California
```

---

## Agent Tools

| Tool | Description |
|---|---|
| `geocode_location` | Converts place names to bounding boxes via OpenStreetMap |
| `fetch_environmental_data` | Downloads satellite data from NASA Harmony |
| `plot_singular` | Renders a regional map for one variable |
| `plot_multiple` | Side-by-side regional comparison maps |
| `compute_statistic_tool` | Computes mean, median, max, min, std over a region |
| `conduct_temporal_statistic` | Plots a time series trend for a region |
| `find_daily_peak` | Locates the peak value and its lat/lon within a region |

---

## Data Sources

**Nitrogen Dioxide (NO₂)**
- [OMI MINDS NO2 Daily (OMI_MINDS_NO2d)](https://doi.org/10.5067/MEASURES/MINDS/DATA304) — NASA GES DISC
- [TROPOMI NO2 Monthly (HAQ_TROPOMI_NO2_GLOBAL_M_L3)](https://disc.gsfc.nasa.gov/) — NASA GES DISC
- [TEMPO NO2 L3 V04 (TEMPO_NO2_L3)](https://asdc.larc.nasa.gov/project/TEMPO) — NASA ASDC

**Ozone (O₃)**
- [TEMPO Total Ozone L3 V04 (TEMPO_O3TOT_L3)](https://asdc.larc.nasa.gov/project/TEMPO) — NASA ASDC
- [OMI Total Ozone Daily L3 (OMDOAO3e)](https://doi.org/10.5067/Aura/OMI/DATA3009) — NASA GES DISC


**Formaldehyde (HCHO)**
- [TEMPO HCHO L3 V04 (TEMPO_HCHO_L3)](https://asdc.larc.nasa.gov/project/TEMPO) — NASA ASDC
- [OMI HCHO Daily L3 (OMHCHOd)](https://doi.org/10.5067/AURA/OMI/DATA3010) — NASA GES DISC
