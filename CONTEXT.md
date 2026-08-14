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

**Retained axis**:
The one dimension a reduction keeps. A timeseries retains time, a vertical profile retains the
vertical axis, and both reduce latitude and longitude away through the same weighting. Naming the
axis rather than the chart is what lets one reduction serve both: a profile is not a new kind of
math, it is the existing regional reduction pointed at a different survivor.
_Avoid_: kept dimension, plot axis (the frontend's y-axis is a rendering choice; this is not)

**Vertical axis**:
The physical coordinate — pressure in hPa or altitude in km — a layered product's values sit on,
identified by CF metadata rather than variable name. Distinct from the *layer index*, which merely
counts. The distinction is load-bearing: TEMPO_O3PROF stores layer 0 at the top of the atmosphere,
so anything plotted against the index alone is upside down while looking entirely plausible.

**Physical level**:
A vertical position named in the units the atmosphere has — `500 hPa`, `26 km` — as opposed to a
*layer index*, which counts. The units are not decoration: a product may publish both pressure and
altitude, and the unit is the only thing that says which axis was meant, so a bare number is
refused rather than defaulted. Distinct from a *layer*, which is what a physical level resolves to.
_Avoid_: level (ambiguous between the request and the layer), altitude (only one of the two kinds)

**Dominance**:
The cos(latitude)-weighted fraction of the analyzed region whose own vertical coordinate resolves a
requested physical level to the same layer the regional mean chose. Weighted by area, like every
other regional fraction here, so it describes the same field the regional mean does. Reported with
its runner-up layer: "83.1% resolve to layer 19, 16.9% to layer 20" is a fact a reader can act on
in a way a bare percentage is not. Measured per request — it depends on the region *and* the time
window, not on the product.

**Level error**:
How far the layer a request resolved to actually sits from the level asked for, in the axis's own
units. Independent of dominance, and both must be disclosed: an 850 hPa request over New Jersey
lands 46 hPa away at 100% dominance, while a 300 hPa request lands 40 hPa away at 83%. A result
reporting only agreement would call the first one perfect.

**Layer order**:
Which end of a layered array is the top of the atmosphere, measured from the vertical axis rather
than assumed. Disclosure for the reader and for export; never an input to rendering, which plots
against the physical axis and is therefore right either way.

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

**Maturity**:
Where a product sits in its provider's validation lifecycle — beta, provisional, validated — with
that provider's own caveat attached. Scientific provenance, not descriptive metadata: a Beta
product's user guide can say publication is "not recommended and highly discouraged", and that
sentence has to reach the artifacts someone pulls while writing a paper. `unknown` means nobody
checked, which is deliberately not the same claim as a checked `validated`.

**QA pass rate**:
The cos(latitude)-weighted fraction of pixels in the analyzed region that had a retrievable value
and passed the quality flag. Counted from the same boolean condition that performs the mask, so it
cannot disagree with the data actually plotted. Distinct from *valid values %*, which answers the
different question "was there any data here at all".
