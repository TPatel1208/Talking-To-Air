# tta-test

A conversational assistant for NASA Earth-observation data: researchers ask questions in natural
language, and the system retrieves satellite granules through the earthdata MCP server, reduces
them, and answers with charts, statistics and ground-sensor comparisons.

Because every answer is a scientific claim, the language below is weighted toward *what a number
describes* and *what was done to the data to produce it*. Terms are added here as they are
resolved, not speculatively.

## Language

### Scope of a result

**Analyzed region**:
The extent a result actually describes — an AOI polygon for a map or regional timeseries, a single
grid cell for a monitor comparison. Every statistic and disclosure attached to a result is scoped
to its analyzed region, never to the granule it was read from.
_Avoid_: AOI (only one kind of analyzed region), bbox, area

**Narrowing**:
Reducing data to its analyzed region — a geometry crop for a region, nearest-cell selection for a
point. Always performed *before* masking, so that everything counted during masking describes the
region the answer is about. Masking first and narrowing afterwards produces numbers that describe
data the researcher never asked about.
_Avoid_: cropping (that's one kind of narrowing), subsetting, clipping

**Counted extent**:
The dimensions and sizes the quality counters were actually reduced over, read off the data rather
than declared by a caller. It is what makes a pixel count interpretable — "4 checked pixels" reads
identically whether it covered a continent or one cell until the counted extent says which.

### Data quality

**Collection identity**:
The marker on an opened granule naming which registered collection it belongs to, spelled either
`short_name` or the CF/ACDD `ShortName`. It is a *dataset-level* fact and is read from the Dataset
before the variable: xarray does not carry a Dataset's global attributes onto a variable taken out
of it, and real products carry the marker only on the Dataset. Resolving it is what makes a pinned
masking rule reachable at all — the tool layer has no collection id, and a science variable's name
is not a registry key — so failing to resolve it does not raise, it silently masks nothing.
_Avoid_: short name (only one of its two spellings), collection id (a different identifier)

**Quality flag variable**:
The sibling variable in a granule that records per-pixel retrieval quality. Identified from a
pinned registry rule, the CF `ancillary_variables` attribute, or a single unambiguous sibling —
never guessed between competing candidates.
_Avoid_: QA band, quality layer, mask variable

**Masking**:
Excluding pixels that are not valid observations — fill sentinels, values outside the valid range,
and pixels the quality flag rejects. Distinct from **narrowing**, which selects *where* to look;
masking decides *which observations there are trustworthy*.

**Masking provenance**:
The disclosure travelling with every result that records what masking actually happened — which
tier decided the rule, whether the quality mask ran at all, and the realized pass rate and counted
extent. Never claims more than was done: a quality mask that could not run reports "not applied"
rather than an optimistic status.
_Avoid_: mask metadata, QA info

**QA pass rate**:
The cos(latitude)-weighted fraction of pixels in the analyzed region that had a retrievable value
and passed the quality flag. Counted from the same boolean condition that performs the mask, so it
cannot disagree with the data actually plotted. Distinct from *valid values %*, which answers the
different question "was there any data here at all".
