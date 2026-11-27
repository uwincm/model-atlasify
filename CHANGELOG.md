# Changelog

## Unreleased

### Maps
- 2 m temperature composites of each `--plot-sub-areas` area get their own
  colour range, by the same rule as the whole domain's (5 C shifts fitting
  the d01 points inside the area at the run's first time).
- Per-item maps are 500 px wide 8-bit palette PNG previews (were 1210 px
  RGBA PNGs): about a sixth of the size. They use a compact layout (same
  fonts on a half-width figure, fewer colour bar labels, a coarser arrow
  lattice); long titles shrink to fit on every map.
- Composite maps (`model-images/`) are AVIF (quality 75) instead of PNG:
  about a third of the size, visually the same. The animation and summary
  pages use the AVIF map of a time, or a PNG left by an older version where
  there is none. Per-item maps stay PNG. New dependency:
  `pillow-avif-plugin` (AVIF support for Pillow builds without it).
- Significant wave height maps draw the dominant wave direction as white
  arrows, like the dominant wave period maps.
- 2 m temperature maps adapt their colour range to each run: the -10 to
  35 C range (every 1.5 C) shifts in 5 C steps to leave the fewest d01 grid
  points outside it at the run's first output time. All the run's `t2m` maps
  use that range.
- OLR/precipitation maps are drawn over satellite imagery (Esri World
  Imagery) instead of the pale land/sea background, so the grey OLR shading
  stands out. `--basemap google_hybrid` uses Google's hybrid imagery (with
  roads and place names) instead, and `--basemap none` the old background.
  Tiles are cached; without network access the maps fall back to the plain
  background.
- Dominant wave period maps use the cmocean "tempo" colour map reversed
  (long-period swell light, without tempo's white end) and a 0-18 s colour bar
  in 2 s steps (was viridis, 0-12 s in 1 s steps), so long-period swell is
  resolved. Mean wave period maps use the same colour map, keeping their
  0-12 s colour bar in 1 s steps.
- Dominant and mean wave period maps draw the dominant and mean wave
  direction as white same-length arrows, between the 10 m wind arrows.

### Products
- GeoTIFFs are LERC-compressed with a maximum error per variable (e.g.
  0.1 degC for t2m, 1 % for RH, 0.01 mm for precipitation; tag `max_error`):
  3-6x smaller. `spray_mass_flux` stays lossless.
- Dominant and mean wave direction (`dominant_wave_direction`, alias `dwd`;
  `mean_wave_direction`, alias `mwd`) from DWM `dwd`/`mwd`: GeoTIFFs in
  degrees clockwise from north that the waves come from (CF convention).
  Part of `--variables waves`.

## 0.2.0 (2026-11-12)

Basic functionality and the full set of maps are in place. Next versions
focus on efficiency and user-friendliness.

### Models
- DWM (Donelan Wave Model) wave output, alongside WRF or on its own,
  including non-Mercator DWM grids.
- HYCOM ocean output (`archv`/`archs` a/b files), read with code adapted from
  pyhycom (no external dependency); used when the UWIN-CM `uwin.nml` has the
  ocean on.
- WRF moving nests: every product is placed where the nest was at that time.

### Products
- Surface winds: earth-relative `u10`/`v10` GeoTIFFs, and wind vectors on
  every surface-field map (scaled with the map's vector spacing).
- 2 m temperature (`t2m`) with the NCL `t2m_29lev` colour table.
- Interval precipitation from `PREC_ACC_C`/`PREC_ACC_NC` when available.
- OLR GeoTIFFs (`olr`) and OLR maps with interval precipitation on top
  (`olr_precipitation`).
- Sea spray mass, latent and sensible heat fluxes (runs with spray on).
- Waves: significant wave height (white 0–0.5 m band), mean and dominant
  wave period.
- Ocean: SST and surface salinity (cmocean colours), surface current speed,
  `surface_u`/`surface_v` GeoTIFFs, and currents drawn as curly vectors.
- Pressure levels (WRF): RH, geopotential height and wind at 700 and
  500 hPa; RH maps with height contours and wind vectors, height maps with
  their own contours.

### Maps
- Composite maps of the outermost domain with its nests drawn on top.
- Lon/lat sub-area maps (`--plot-sub-areas`), `--no-full-plots`.
- Looping HTML animations of each set of composite maps, and a
  `model-images.html` summary table.

### Input/output and processing
- Output straight to S3 (`-o s3://bucket/prefix`).
- The output mirrors the input path (`--cut-dirs` to shorten it).
- `catalog.json` keeps earlier runs instead of being overwritten.
- The exact command is recorded with each run.
- Valid times processed in parallel (`-np`), with progress bars.

## 0.1.0

First working version: WRF output to Cloud-Optimized GeoTIFFs, PNG maps and
a STAC catalog.
