/**
 * T59 Phase 2, heap arm — what a frame stack costs in a JS runtime.
 *
 * Plain node, on purpose. The frontend test harness has no jsdom, and a
 * simulated DOM would measure the simulation rather than the typed array.
 * Nothing here needs a document: the question is what `Float32Array` over a
 * decompressed blob costs, and that is a runtime fact, not a DOM one.
 *
 * The measurement that matters and is easy to get wrong: a typed array's
 * storage does NOT live in `heapUsed`. It lives in the ArrayBuffer, which node
 * reports under `external`/`arrayBuffers`. Reading only `heapUsed` would report
 * a 60-frame stack as costing a few hundred KB and conclude frames are free.
 * So all four counters are printed and `arrayBuffers` is the one the budget is
 * set from.
 *
 * Also measures the two framings of the store the PRD's Phase 6 chooses
 * between:
 *   flat   — one Float32Array for the whole stack, frames addressed by
 *            `subarray` views (zero copy)
 *   sliced — one Float32Array per frame, copied out
 * The delta is what "reused-buffer path" actually buys.
 *
 * Usage: node bench_t59_frame_heap.mjs <blob-dir> [--json out.json]
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join } from "node:path";

const MB = (b) => (b / 1e6).toFixed(2);

function mem(label) {
  // A full GC between samples so the numbers are live data, not uncollected
  // garbage from the previous case. Requires --expose-gc; without it the
  // readings drift upward and every later case looks more expensive than it is.
  if (global.gc) {
    global.gc();
    global.gc();
  }
  const m = process.memoryUsage();
  return {
    label,
    rss: m.rss,
    heapUsed: m.heapUsed,
    external: m.external,
    arrayBuffers: m.arrayBuffers,
  };
}

function delta(a, b) {
  return {
    rss: b.rss - a.rss,
    heapUsed: b.heapUsed - a.heapUsed,
    external: b.external - a.external,
    arrayBuffers: b.arrayBuffers - a.arrayBuffers,
  };
}

function run(dir, name) {
  const gzPath = join(dir, name);
  const gz = readFileSync(gzPath);

  const base = mem("baseline");

  let t0 = process.hrtime.bigint();
  const raw = gunzipSync(gz);
  const gunzipMs = Number(process.hrtime.bigint() - t0) / 1e6;

  // The real path: a Float32Array VIEW over the decompressed bytes. No copy,
  // so the buffer node already allocated for gunzip is the storage. Copying
  // into a fresh Float32Array instead would double the peak for no benefit.
  t0 = process.hrtime.bigint();
  const flat = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const viewMs = Number(process.hrtime.bigint() - t0) / 1e6;

  const afterFlat = mem("flat");

  // Touch every value so nothing is deferred and any lazy allocation lands
  // before the measurement. Also gives the NaN count, which is the sanity
  // check that these are the same bytes Python wrote.
  t0 = process.hrtime.bigint();
  let finite = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < flat.length; i++) {
    const v = flat[i];
    if (Number.isFinite(v)) {
      finite++;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const scanMs = Number(process.hrtime.bigint() - t0) / 1e6;

  const m = name.match(/_(\d+)f_(\d+)c/);
  const cells = m ? Number(m[2]) : null;
  // Frame count comes from the BUFFER, not the filename. The filename records
  // the requested N, and a pool short of its target wrote fewer frames than it
  // asked for -- trusting the name there would slice past the end of the
  // buffer, and `subarray` clamps silently rather than throwing, so the views
  // measurement would quietly be of empty arrays.
  const nFrames = Math.floor(raw.byteLength / 4 / cells);
  const namedFrames = m ? Number(m[1]) : null;
  const shortNote = nFrames !== namedFrames ? ` (named ${namedFrames}f, pool was short)` : "";

  // Per-frame views (what the scrubber hands the canvas each tick).
  const views = [];
  t0 = process.hrtime.bigint();
  for (let f = 0; f < nFrames; f++) views.push(flat.subarray(f * cells, (f + 1) * cells));
  const subarrayMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const afterViews = mem("views");

  // The alternative: a copy per frame.
  const copies = [];
  t0 = process.hrtime.bigint();
  for (let f = 0; f < nFrames; f++) copies.push(flat.slice(f * cells, (f + 1) * cells));
  const sliceMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const afterCopies = mem("copies");

  const dFlat = delta(base, afterFlat);
  const dViews = delta(afterFlat, afterViews);
  const dCopies = delta(afterViews, afterCopies);

  console.log(
    `  ${name.padEnd(34)} N=${String(nFrames).padStart(3)} cells=${String(cells).padStart(6)}  ` +
      `gz ${MB(gz.length).padStart(6)} MB -> raw ${MB(raw.length).padStart(6)} MB${shortNote}`
  );
  console.log(
    `      gunzip ${gunzipMs.toFixed(1).padStart(6)} ms   view ${viewMs.toFixed(3)} ms   ` +
      `scan ${scanMs.toFixed(1)} ms   subarray x${nFrames} ${subarrayMs.toFixed(2)} ms   ` +
      `slice-copy x${nFrames} ${sliceMs.toFixed(1)} ms`
  );
  console.log(
    `      flat load: arrayBuffers +${MB(dFlat.arrayBuffers)} MB  heapUsed +${MB(dFlat.heapUsed)} MB  ` +
      `rss +${MB(dFlat.rss)} MB`
  );
  console.log(
    `      views:     arrayBuffers +${MB(dViews.arrayBuffers)} MB  heapUsed +${MB(dViews.heapUsed)} MB` +
      `      copies: arrayBuffers +${MB(dCopies.arrayBuffers)} MB (the cost of NOT reusing the buffer)`
  );
  console.log(
    `      finite ${finite}/${flat.length} (${((100 * finite) / flat.length).toFixed(1)}%)  ` +
      `range ${min.toExponential(3)} .. ${max.toExponential(3)}`
  );

  return {
    file: name,
    n_frames: nFrames,
    cells_per_frame: cells,
    gz_bytes: gz.length,
    raw_bytes: raw.length,
    gunzip_ms: Number(gunzipMs.toFixed(3)),
    view_ms: Number(viewMs.toFixed(4)),
    scan_ms: Number(scanMs.toFixed(3)),
    subarray_ms: Number(subarrayMs.toFixed(4)),
    slice_copy_ms: Number(sliceMs.toFixed(3)),
    flat_delta: dFlat,
    views_delta: dViews,
    copies_delta: dCopies,
    finite,
    total: flat.length,
  };
}

const dir = process.argv[2];
if (!dir) {
  console.error("usage: node --expose-gc bench_t59_frame_heap.mjs <blob-dir> [--json out]");
  process.exit(2);
}
if (!global.gc) {
  console.log("  ! running without --expose-gc; heap deltas include uncollected garbage");
}
console.log(`node ${process.version}\n`);

const files = readdirSync(dir).filter((f) => f.endsWith(".bin.gz")).sort();
const out = [];
for (const f of files) out.push(run(dir, f));

const jsonIdx = process.argv.indexOf("--json");
if (jsonIdx > -1 && process.argv[jsonIdx + 1]) {
  writeFileSync(process.argv[jsonIdx + 1], JSON.stringify(out, null, 2));
  console.log(`\n  wrote ${process.argv[jsonIdx + 1]}`);
}
