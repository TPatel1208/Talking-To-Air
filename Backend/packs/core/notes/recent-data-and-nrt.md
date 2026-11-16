---
id: recent-data-and-nrt
title: Recent days, product latency and near-real-time products
datasets:
  - MODIS_AOD_TERRA
  - MODIS_AOD_AQUA
  - VIIRS_AOD_SNPP
  - VIIRS_AOD_NOAA20
  - AERDA_AOD_NRT
  - AERDA_AOD_NRT_3H
  - TEMPO_NO2_NRT
  - TEMPO_HCHO_NRT
---
Standard L3 collections, including the daily MODIS/VIIRS AOD grids, are processed with a multi-day latency, so the most recent few days legitimately have no granules yet. That is expected product latency, not evidence the data does not exist. Near Real-Time (NRT) products carry a short rolling window, often only the last day or two, so a multi-day recent range may fill only partly. Say which days returned data ("data returned for N of your M requested days"); never present the covered subset as the whole range.
