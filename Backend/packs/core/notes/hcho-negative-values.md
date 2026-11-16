---
id: hcho-negative-values
title: Negative formaldehyde columns are retrieval noise
quantities:
  - hcho_total_column
---
Retrieval noise scatters HCHO columns below zero over clean scenes: about 31% of finite TEMPO pixels and 25% of finite OMI pixels in a 2026-08-02 check. These are good-quality observations and are kept, so a negative value is noise, not an error. Clipping them at zero would censor the low tail and bias the mean high.
