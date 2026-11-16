---
id: ozone-profile-layers
title: Reading the TEMPO ozone profile's layers and levels
datasets:
  - TEMPO_O3PROF
quantities:
  - o3_profile
---
The profile has 24 layers of partial columns. The `layer` dimension is a bare 0-23 index with no coordinate values: the physical altitude and pressure live in per-pixel companion variables, and layer 0 is the top of the atmosphere. A layer index is not a height: layer 26 is not 26 km. A height or pressure the researcher names (with its units) resolves to the nearest available layer, and the result says which layer and how far it sits from what was asked; "nearest available layer to 300 hPa" is a different claim from "a 300 hPa map". A number without units is ambiguous (500 hPa is mid-troposphere, 500 km is outer space); the unit is the only thing saying which axis was meant, so do not supply one.
