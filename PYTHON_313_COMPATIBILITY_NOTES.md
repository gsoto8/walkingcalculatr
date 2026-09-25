# walkingcalculatr Python 3.13 Compatibility and Debugging Notes

## Purpose

This document records issues encountered while testing the `CDCgov/walkingcalculatr` Python workflow with Python 3.13 and current Python 3.13-compatible dependencies.

The original repository requirements were published for an older package environment. Because Python 3.13 is required on the current workstation, compatibility issues and any necessary changes are being documented as they are identified.

The goals are to:

- Confirm that the walkingcalculatr workflow can run under Python 3.13.
- Identify compatibility changes required for newer Python package versions.
- Document differences in input-data formats that require changes to the existing code.
- Distinguish analytical-code issues from workstation, network, and external-service issues.
- Preserve a record of changes and their validation.
- Document workstation-specific setup requirements needed to reproduce the workflow.

---

# Environment

- **Python:** 3.13.7
- **Operating system:** Windows
- **Virtual environment:** `wc8r_env`
- **Working repository:** `walkingcalculatr-debug-project-prep`
- **Jupyter kernel:** `Python (wc8r_env)`

The August 2024 pinned `requirements.txt` could not be installed exactly under Python 3.13.

For example, the pinned version of `contourpy` did not provide a compatible Python 3.13 wheel and attempted to build from source. The build failed because the required Visual Studio C++ build tools were not available on the workstation.

The repository's alternative installation approach was therefore used to install current Python 3.13-compatible versions of the required packages.

Major package versions in the current environment include:

- OSMnx: 2.1.1
- requests: 2.34.2
- urllib3: 2.8.0
- Polars: 1.44.2
- pandas: 3.0.6
- GeoPandas: 1.1.4
- NumPy: 2.5.3
- OpenSSL: 3.0.16

---

# Current Debugging Findings

## Issue 1 — `CleanData()` Timestamp Handling / Polars

### Background

The input datasets can represent Unix timestamps differently:

- Spectus data use **10-digit Unix timestamps in seconds**.
- MITRE synthetic data use **13-digit Unix timestamps in milliseconds**.

The original `CleanData()` implementation divided timestamps by 1000 unconditionally in some input branches.

For an already-valid 10-digit timestamp, dividing by 1000 removes timestamp precision.

The subsequent duplicate-removal operation:

```python
dfclean = dfclean.unique(
    subset=["userId", "time"],
    maintain_order=True
)
```

can then remove observations that have been collapsed onto the same timestamp.

### Previous attempted adjustment

An earlier conditional implementation attempted to check whether the timestamp contained 10 or 13 digits before converting it.

That implementation used a pattern similar to:

```python
pl.when(
    dfclean.with_columns(...)
)
```

Under the current version of Polars, this produced:

```text
ComputeError: cannot cast categorical types to Boolean
```

The error occurred because `dfclean.with_columns(...)` returns a DataFrame, while `pl.when()` expects a Boolean expression.

### Current change

Timestamp handling was changed so that:

- 10-digit Unix timestamps remain in seconds.
- 13-digit Unix timestamps are converted from milliseconds to seconds.
- Other timestamp lengths are left unchanged after conversion to an integer representation.

The current implementation is:

```python
time_int = pl.col("time").cast(pl.Int64)

dfclean = dfclean.with_columns(
    pl.when(
        time_int.cast(pl.String).str.len_chars() == 13
    )
    .then(
        time_int // 1000
    )
    .otherwise(
        time_int
    )
    .alias("time")
)
```

The existing duplicate-removal operation remains unchanged:

```python
dfclean = dfclean.unique(
    subset=["userId", "time"],
    maintain_order=True
)
```

### Validation

`CleanData()` was tested independently after the change.

Observed results:

- **Input rows:** 604,512
- **Clean rows:** 604,512
- **Unique users:** 3,162
- **Unique timestamps:** 355,556

The previous unintended collapse of records was no longer observed.

The complete Quickstart workflow was subsequently run successfully with this version of `CleanData()`.

### Interpretation

This issue has two components:

1. **Input-data compatibility:** Spectus and MITRE synthetic data use different Unix timestamp units.
2. **Polars compatibility:** The earlier conditional implementation used a DataFrame inside `pl.when()`, which is incompatible with the current Polars API.

### Status

**Resolved / validated**

---

## Issue 2 — Managed-Network SSL Certificate Verification

### Error

Running `get_walkable_areas()` initially produced an SSL certificate verification error when OSMnx attempted to communicate with the OpenStreetMap Overpass API:

```text
SSLCertVerificationError:
certificate verify failed:
unable to get local issuer certificate
```

### Cause

The Python environment's default certificate trust configuration could not validate the HTTPS certificate chain on the managed workstation/network.

This is an environment/configuration issue rather than a problem with the walkingcalculatr analytical logic.

### Resolution

The `truststore` package was installed in the walkingcalculatr virtual environment:

```powershell
.\wc8r_env\Scripts\python.exe -m pip install truststore
```

At the beginning of a fresh Jupyter kernel, before importing OSMnx, Requests, or `utils.py`:

```python
import truststore
truststore.inject_into_ssl()
```

The repository code can then be imported:

```python
from utils import *
```

This allows Python to use the Windows certificate trust store while retaining HTTPS certificate verification.

### Validation

After enabling `truststore`, the original:

```text
CERTIFICATE_VERIFY_FAILED
```

error was no longer observed.

Subsequent testing exposed a different error involving fresh OSMnx/Overpass requests. That issue is documented separately below.

### Status

**Certificate-verification issue resolved with `truststore`**

---

## Issue 3 — OSMnx Fresh Overpass Requests / Local Cache

### Error

After resolving the certificate-verification error, fresh OSMnx requests to the OpenStreetMap Overpass API produced:

```text
SSLEOFError: [SSL: UNEXPECTED_EOF_WHILE_READING]
EOF occurred in violation of protocol
```

The error occurred during fresh OSMnx requests to the Overpass API.

### Initial investigation

Small OSMnx requests were tested independently of `get_walkable_areas()` and produced the same `SSLEOFError`.

This demonstrated that the failure was not specific to:

- Ness County,
- the size of the county polygon,
- `CleanData()`,
- the timestamp changes, or
- the analytical logic inside `get_walkable_areas()`.

A previously working Quickstart environment was then compared with the current debugging environment.

### Environment comparison

Both environments used the same versions of the major packages:

- Python 3.13.7
- OSMnx 2.1.1
- requests 2.34.2
- urllib3 2.8.0
- Polars 1.44.2
- pandas 3.0.6
- GeoPandas 1.1.4
- NumPy 2.5.3
- OpenSSL 3.0.16

Both environments also reported:

```text
use_cache: True
cache_folder: ./cache
requests_kwargs: {}
```

This indicated that the Python/package environment itself was not the apparent source of the different behavior.

### OSMnx cache comparison

Because `cache_folder` was specified as `./cache`, each repository used a separate cache directory relative to its working directory.

#### Previously working repository

Cache location:

```text
C:\Users\qxb1\walkingcalculatr\Python\cache
```

Cache contents:

- 23 cached files
- Approximately 23.5 MB

#### Current debugging repository

Cache location:

```text
C:\Users\qxb1\walkingcalculatr-debug-project-prep\Python\cache
```

Cache contents:

- 9 cached files
- Approximately 1.7 MB

The previously working repository therefore contained substantially more cached OSMnx/Overpass responses.

### Diagnostic test

The debugging notebook was temporarily configured to use the cache from the previously working repository:

```python
ox.settings.use_cache = True
ox.settings.cache_folder = r"C:\Users\qxb1\walkingcalculatr\Python\cache"
```

After pointing OSMnx to the previously populated cache, the following Ness County call completed quickly and without the SSL error:

```python
tract_area_lst = get_walkable_areas(
    county_name="Ness",
    county_state="KS",
    year=2023,
    fill_size=200E-6
)
```

### Interpretation

The previously working Quickstart notebook was able to reuse OSMnx/Overpass responses that had already been downloaded and stored in its local cache.

The current debugging repository did not contain all required cached responses. As a result, OSMnx attempted one or more fresh requests to the Overpass API, which encountered the TLS/SSL EOF error.

Pointing the debugging environment to the previously populated cache allowed the Ness County workflow to complete successfully.

This indicates that the observed failure is associated with fresh OSMnx/Overpass network requests rather than the analytical logic in `utils.py`.

The existing cache can be used to continue Ness County Python 3.13 compatibility testing without allowing the external network issue to block debugging.

However, the existing cache is **not a general solution for future processing of additional counties**. Queries that are not already cached will require fresh requests to the Overpass API and may encounter the same network/TLS issue.

### Current debugging workaround

For continued Ness County compatibility testing:

```python
ox.settings.use_cache = True
ox.settings.cache_folder = r"C:\Users\qxb1\walkingcalculatr\Python\cache"
```

This accesses previously retrieved Ness County OSM data while Python 3.13 compatibility testing continues.

### Status

**Analytical-code issue ruled out for this error. External OSMnx/Overpass network issue remains open.**

---

## Issue 4 — OneDrive File Access

### Error

While initially running the repository from a OneDrive-synchronized directory, Python was unable to read `utils.py`.

Even a direct file-read operation produced:

```text
OSError: [Errno 22] Invalid argument
```

The error occurred before the code inside `utils.py` was executed.

### Resolution

A local working copy of the repository was created outside the OneDrive-synchronized directory.

The repository is now being run from a local working directory.

A new Python virtual environment was created for the local working copy.

### Result

Python successfully read `utils.py`, and the repository utility module could be imported.

### Status

**Resolved**

---

# Quickstart End-to-End Validation

## Result

After applying the current compatibility changes and pointing OSMnx to the previously populated Ness County cache, the full Quickstart notebook completed successfully from start to finish.

The workflow successfully completed:

- raw data loading
- `CleanData()`
- `ExtractWalks()`
- `get_walkable_areas()`
- `get_walkable()`
- final walk filtering and summary output

### `CleanData()` validation

```text
Input rows:       604,512
Clean rows:       604,512
Unique users:       3,162
Unique timestamps: 355,556
```

### Interpretation

The current `utils.py` changes are sufficient for the Quickstart workflow to run successfully under the Python 3.13 environment used for testing.

The OSMnx/Overpass fresh-request problem is separate from the analytical code and is currently avoided for Ness County by using previously cached OSM responses.

### Status

**Quickstart workflow: passed end-to-end**

---

# Previous Python 3.13 Compatibility Finding

The following issue was identified during earlier testing of another walkingcalculatr notebook. It predates the current Quickstart debugging work but is retained here because it is relevant to Python 3.13/current-package compatibility.

## `tractWalkPrevalence()` / pandas 3

### Location

**Notebook:** `Walk_Prev_Sidewalk_Density`

**Function:** `tractWalkPrevalence()`

**Utility module:** `Python/utils.py`

### Error

The function failed with:

```text
TypeError:
Invalid value 'No Residents' for dtype 'float64'
```

### Cause

The original `tractWalkPrevalence()` function assigns the string:

```text
"No Residents"
```

to the `walk_prev` column.

Under the installed pandas 3 environment, `walk_prev` was represented as a `float64` column.

pandas 3 does not permit assigning a string to this numeric column without explicitly changing the column to a compatible data type.

### Change

Immediately after the `tractPopulation.merge(...)` operation and before assigning `"No Residents"`, the following line was added:

```python
walkPrevDF["walk_prev"] = walkPrevDF["walk_prev"].astype("object")
```

The existing repository logic can then remain unchanged:

```python
walkPrevDF.loc[
    (walkPrevDF["count_walkable"] > 0) &
    (walkPrevDF["pop_estimate"] == 0),
    "walk_prev"
] = "No Residents"
```

### Important implementation detail

The dtype conversion must occur **after** the merge operation.

Placing:

```python
walkPrevDF["walk_prev"] = walkPrevDF["walk_prev"].astype("object")
```

before the merge did not resolve the issue because the merge recreated the DataFrame and `walk_prev` was again represented as a numeric column.

### Result

After moving the dtype conversion immediately after the merge, `tractWalkPrevalence()` proceeded past the previously failing operation.

### Interpretation

This appears to be a compatibility adjustment required by newer pandas behavior rather than a change to the intended analytical logic.

The original function intentionally allows `walk_prev` to contain both:

- numeric prevalence values, and
- the text label `"No Residents"`.

Explicitly converting the column to `object` makes this intended mixed-type behavior explicit.

### Status

**Resolved / validated during earlier testing**

---

# Compatibility Issue Summary

## Current debugging

| ID | Component | Issue | Status |
|---|---|---|---|
| 001 | Python environment | August 2024 pinned requirements could not be reproduced exactly under Python 3.13 | Addressed |
| 002 | `CleanData()` / input data / Polars | 10-digit vs. 13-digit timestamp handling and newer `pl.when()` behavior | Resolved / validated |
| 003 | OSMnx / HTTPS | Python certificate verification failed on managed workstation/network | Resolved with `truststore` |
| 004 | OSMnx / Overpass cache | Fresh Overpass requests produced `SSLEOFError`; cached Ness responses work | External network issue remains open |
| 005 | File access | OneDrive copy of `utils.py` produced `OSError: [Errno 22]` | Resolved |
| 006 | Quickstart | Full Ness County Quickstart workflow | Passed end-to-end |

## Previous compatibility finding

| Component | Issue | Status |
|---|---|---|
| `tractWalkPrevalence()` / pandas 3 | `"No Residents"` could not be assigned to a `float64` column | Resolved / validated during earlier testing |

---

# Debugging Approach

For each additional issue encountered during Python 3.13 testing:

1. Record the original error before modifying code.
2. Identify the function and line where the error occurs.
3. Determine whether the issue appears to be:
   - a Python/package compatibility issue,
   - an input-data format issue,
   - an environment/configuration issue,
   - an external-service/network issue, or
   - an issue in the existing analytical logic.
4. Make one change at a time.
5. Test the affected function independently where practical.
6. Record the exact change.
7. Record the validation result.
8. Create a local checkpoint after a validated code change.
9. Continue using `Python/utils.py` as the working utility module.

---

# Local Checkpoints

Known-good copies of `utils.py` are stored locally in:

```text
debug_checkpoints/
```

These checkpoint files are for local recovery during debugging and are not intended to be uploaded to the repository.

## Current checkpoints

### `utils_before_CleanData_reset.py`

Snapshot of `utils.py` before resetting and simplifying the `CleanData()` timestamp changes.

### `utils_after_CleanData_timestamp_fix.py`

Known-good version after validating the revised `CleanData()` timestamp handling.

Validation results:

```text
Input rows:       604,512
Clean rows:       604,512
Unique users:       3,162
Unique timestamps: 355,556
```

### `utils_quickstart_full_pass.py`

Known-good version after the complete Quickstart notebook successfully passed end-to-end testing.

---

# Next Steps

- Continue testing the remaining repository notebooks under Python 3.13.
- Document additional compatibility issues only as they occur.
- Keep fresh OSMnx/Overpass network problems separate from analytical-code issues.
- Preserve validated versions of `utils.py` as local checkpoints.
- Investigate fresh Overpass connectivity separately before scaling to uncached counties.
- Review and consolidate validated code changes before transferring them to GitHub.