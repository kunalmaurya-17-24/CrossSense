# Data sources and access record

Source documentation was checked on 2026-11-25. The core pipeline accepts local
prepared rasters and requires no account or network connection. Synthetic data
are pipeline fixtures; they are never SEN12MS or Copernicus observations.

## SEN12MS: primary development source

The [authors' dataset repository](https://github.com/zhu-xlab/SEN12MS) and
[toolbox](https://github.com/schmitt-muc/SEN12MS) direct users to the official
[mediaTUM record](https://mediatum.ub.tum.de/1474000), persistent dataset DOI
[10.14459/2019mp1474000](https://doi.org/10.14459/2019mp1474000).
No unrelated mirror is used. SEN12MS-CR and SEN12MS-CR-TS are different datasets
and are not silently substituted.

Citation: Schmitt, M., Hughes, L. H., Qiu, C., and Zhu, X. X. (2019),
*SEN12MS – A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery
for Deep Learning and Data Fusion*, ISPRS Annals IV-2/W7, 153–160,
[doi:10.5194/isprs-annals-IV-2-W7-153-2019](https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019).
The [authors' paper, section 4.2](https://elib.dlr.de/133280/1/SEN12MS_Preprint.pdf)
identifies the dataset license as **CC-BY** and requests citation. The accessible
paper does not establish a versioned CC-BY grant here; inspect the record's
current terms before redistribution. The project's code license does not
relicense any satellite data.

Expected organization: four seasonal groups, `ROIs1158_spring`,
`ROIs1868_summer`, `ROIs1970_fall`, `ROIs2017_winter`, with `s1`, `s2`, and `lc`
subsets. Matching filenames identify ROI and patch (`pXXX`). GeoTIFFs contain
2 SAR channels (VV/VH dB), 13 optical channels in native Sentinel-2 band order,
and four MODIS land-cover layers. Patches are 256×256 on a 10 m grid. The
published collection occupies about 421.3 GiB unpacked and includes overlapping
patches; downloading the full collection is neither necessary nor automatic.
[Dataset structure and storage description](https://elib.dlr.de/133280/1/SEN12MS_Preprint.pdf).

The author documents FTP and rsync access in
[toolbox issue #4](https://github.com/schmitt-muc/SEN12MS/issues/4). The public
dataset share uses a published share login; this is not a personal TUM account.
Our code uses that documented login internally and strips credentials from all
audit URLs. No personal registration is added or bypassed. The author reported
that the historical Nextcloud download endpoint was not a stable HTTPS API;
we preserve certificate checks and do not work around TLS errors.

### Bounded attempt

```powershell
crosssense download-sen12ms --output outputs/data_access/sen12ms --max-mib 50 --time-limit 90
```

The attempted subset is the official winter SAR archive
`ROIs2017_winter_s1.tar.gz` and its optical partner
`ROIs2017_winter_s2.tar.gz` on `dataserv.ub.tum.de`. These are the documented
archive units, not a promise that either fits the cap. The code requests FTP
`SIZE` before either body and checks the combined size and free space. If no
small subset is exposed, a seasonal archive is not partially downloaded and
represented as a usable subset. Unknown size, oversized archive, permission,
missing-file, and authentication failures stop the attempt. Temporary network
failures alone receive at most two retries, with 1 and 2 second delays, inside
the total time budget. The byte cap counts retry traffic too.

`outputs/data_access/sen12ms/sen12ms_download_audit.json` records source,
attempted files, sizes when known, free disk space, UTC times, byte counts,
retry count, status, error and manual next steps. The sticky audit prevents
automatic repeated attempts. Downloaded archives are checked for safe members
and expansion size before extraction; readable SAR/optical files, band counts,
CRS, bounds, resolution, checksums and actual byte sizes are recorded in
`verified_files.json` on successful verification. A downloaded file alone is
not considered a verified dataset.

**Actual development outcome:** the single bounded FTP attempt ran from
2026-11-25 04:46:58 to 04:47:04 UTC. The server reported **14,379,439,087 bytes**
for the winter SAR archive, exceeding the **52,428,800 byte** cap. It stopped
before fetching a body or querying the optical archive's size: **0 raster bytes,
0 retries**, no downloaded or verified pairs. Free disk space was
431,207,309,312 bytes. The sanitized durable audit is
`data/manifests/sen12ms_attempt.json`; the original remains under outputs.
Browser research of the official mediaTUM page also returned HTTP 429 and its
share page timed out; these discovery failures are separate from the FTP size
limit. All development training/evaluation therefore uses synthetic fixtures.

### Manual local preparation

1. Open the official mediaTUM record and review the current terms. Use its
   browser or author-documented FTP/rsync route to obtain a small matched
   S1/S2 subset. Check size first. A larger transfer needs a deliberate choice.
2. Keep the downloaded archive and original product/ROI metadata outside Git.
   Preserve acquisition timestamps rather than assigning seasonal midpoint
   dates: SEN12MS filenames alone do not provide reliable exact times.
3. Verify local files with
   `python -c "from crosssense.data.download import verify_sen12ms_directory; print(verify_sen12ms_directory('data/raw/sen12ms'))"`.
4. Run `crosssense import-sen12ms --root data/raw/sen12ms` to index the author
   patch keys, then complete its scene CSV following `data/README.md`. Supply
   actual timezone-aware times, `source=sen12ms`, band names, and cloud/validity
   provenance. If dates cannot be recovered, default pairing rejects them.
   An explicit `data.allow_author_pairs_without_time: true` opt-in permits only
   identical author keys and records `time_verified=false`, `time_gap_days=-1`.
   Unknown cloud masks likewise remain unknown; this relaxed route cannot
   support a strict time-window/cloud-filtered experiment claim.
5. Build pairs from that CSV. Existing geocoded/calibrated SAR dB must not be
   log-transformed a second time. SEN12MS optical values are not interchangeable
   with modern Sentinel-2 L2A SAFE values or Earth Engine harmonized SR.

## India extension: official Copernicus Data Space Ecosystem

Selected AOI: **Pune, Maharashtra**, WGS84 bounds `[73.80,18.45,73.90,18.55]`,
1–15 January 2024 inclusive. Target local projected grid: **EPSG:32643**,
10 m. This small AOI is an intended extension, not evidence of real-data model
evaluation. Its specification is in `data/manifests/india_aoi.json`.

Use the official [Explore Data](https://dataspace.copernicus.eu/explore-data)
or [Copernicus Browser](https://browser.dataspace.copernicus.eu/). The implemented
metadata route is the documented
[OData Products API](https://documentation.dataspace.copernicus.eu/APIs/OData.html)
at `https://catalogue.dataspace.copernicus.eu/odata/v1/Products`.

```powershell
crosssense catalog-india --output data/manifests/india_catalogue.json
```

The catalogue search does not send authentication. It saves at most four
Sentinel-1 IW GRDH dual VV/VH candidates and four Sentinel-2 L2A candidates with
scene cloud cover ≤20%. It selects by geographic intersection and the fixed
date interval. A candidate list is not a patch pairing; apply the configurable
patch overlap, time-gap, cloud and validity rules after preprocessing. Catalogue
errors and empty results are saved without repeated calls.

**Actual catalogue outcome:** one metadata-only query per sensor succeeded at
2026-11-25 04:47:09 UTC and saved four product records in
`data/manifests/india_catalogue.json`. The two SAR records are original/COG
forms of the **same** 2024-01-08 descending relative-orbit-136 acquisition,
not independent observations. The optical candidates are tile **43QCA** on
2024-01-04 and 2024-01-14, with catalogue scene cloud percentages **3.754334**
and **0.051579**. Both optical dates are within the default seven-day window
of the SAR observation, but patch-level overlap/cloud/validity and prepared
pixel content have not been verified. No candidate raster was downloaded.

Each actual candidate record retains product UUID/name, acquisition start/end,
orbit direction and numbers, S2 tile, processing level/baseline, cloud percentage,
polarizations, geographic footprint, file size, available checksums, and the
original attributes. Unknown fields remain null. Footprint coordinates are
EPSG:4326; **raster projection remains unknown until the local product is
inspected**. The target EPSG code is a preparation choice. The catalogue JSON
also records the exact query and query time, whether any raster was downloaded,
and attribution/terms links.

Full product download requires CDSE registration/login and a valid OAuth token.
Follow [official token generation](https://documentation.dataspace.copernicus.eu/APIs/Token.html)
and store only the short-lived `CDSE_ACCESS_TOKEN` in the process environment.
Use `download_cdse_product(product_id, destination, expected_bytes=..., ...)`
from `crosssense.data.download`; the same size/time/audit limits apply. Most
complete products exceed 50 MiB; stop and choose a bounded browser AOI export
or a smaller prepared asset instead of increasing the cap silently. No paid
service or processing-unit purchase is used by this code.

### Source-specific local preparation

Raw CDSE GRD measurement TIFFs are **not calibrated terrain-corrected dB
rasters**. Prepare VV and VH using an appropriate Sentinel-1 processor such as
ESA SNAP: apply orbit information where available, border/thermal noise
handling as applicable, radiometric calibration, and terrain correction with
a documented DEM and pixel spacing. See the official
[Sentinel-1 processing description](https://sentiwiki.copernicus.eu/web/s1-processing).
Export a projected two-band GeoTIFF with
physical sigma0 dB or explicitly declared linear power. Save the SNAP graph,
software version, orbit file/DEM IDs, and preprocessing decisions with the
scene manifest. CrossSense does not implement SNAP processing itself.

For CDSE Sentinel-2 **L2A**, read quantification and per-band additive offsets
from the product metadata; apply `(DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE`
to valid pixels before export. Do not blindly divide all modern SAFE DN values
by 10000. The equation and DN=0 nodata rule are specified in the official
[Sentinel-2 products documentation](https://sentiwiki.copernicus.eu/web/s2-products).
Export B2/B3/B4/B8 in a known order, preserve nodata, and derive a
separate binary cloud/shadow mask from SCL (document excluded classes, e.g.
0,1,3,8,9,10,11). Reproject continuous reflectance/backscatter with bilinear
resampling and categorical masks with nearest-neighbor. CrossSense then aligns
these prepared rasters to a shared grid and extracts patches. Four-band prepared
optical exports require optical band indices `[1,2,3,4]` and scale `1.0` in a
source-specific config, rather than the full-13-band SEN12MS defaults.
Use `source=copernicus_prepared`. SAR scene rows must explicitly declare
`calibrated=true`, `terrain_corrected=true`, and `sar_units=db` (or `linear`).
Optical rows must declare `processing_level=L2A`, `reflectance_scale=1.0`, and
`reflectance_offset=0.0` for an already normalized export. These declarations
are provenance assertions, not a substitute for actually performing the
required calibration and offset handling.

Access and use of Sentinel data are governed by the
[CDSE terms](https://dataspace.copernicus.eu/terms-and-conditions) and linked
[Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
Preserve “Contains Copernicus Sentinel data (2024)” and identify modifications.
Recheck current terms when distributing derived artifacts.

## Optional Earth Engine route

The core pipeline does not import Earth Engine. If it is used for preparation,
follow the official [Sentinel catalogue](https://developers.google.com/earth-engine/datasets/catalog/sentinel/):

* [`COPERNICUS/S1_GRD`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
  provides calibrated, terrain-corrected dB backscatter. Select a homogeneous
  IW VV/VH subset; do not repeat the logarithm.
* [`COPERNICUS/S2_SR_HARMONIZED`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
  provides harmonized surface reflectance scaled by 10000. It has 12 spectral
  bands, not the 13 bands of SEN12MS. Select explicit band names and masks.

Earth Engine needs a Google account, an enabled/registered Cloud project,
appropriate permissions and compliance with its
[access terms](https://developers.google.com/earth-engine/guides/access).
No Earth Engine request, authentication, export, or paid service was run during
development. Export prepared GeoTIFFs plus provenance, then use the same local
scene CSV. Keep any EE scripts outside the core pipeline. Account-dependent
data preparation is not required after suitable rasters are available locally.
