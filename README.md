## Census Tool

This repo contains a core data pipeline (`census_app_core.py`) and a GUI wrapper
(`app.py`) for downloading ACS data, joining it to Census geographies, and
exporting results to Excel and GeoPackage.

### What it does
- Looks up the county FIPS for a city + state (or county name).
- Pulls ACS 5-year data from the Census API.
- Writes an Excel workbook of ACS tables.
- Merges the ACS data to a shapefile and writes a GeoPackage.

---

## Requirements

Python 3.10+ recommended. Install the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

You also need a **Census API key**:
- Set the environment variable `CENSUS_API_KEY`, or
- Put it in `config.json` as `"CENSUS_API_KEY": "your_key_here"`, or
- Enter it directly in the GUI.

---

## Inputs

Both the GUI and the core script need:
- **County** (or city) name
- **State** (name or abbreviation)
- **ACS year** (2018–2023 supported in the UI)
- **Output folder**
- **A shapefile** with a `GEOID` column (or `FIPS` for block groups)

Shapefile requirements:
- Block groups: 12-digit GEOID (state+county+tract+block group)
- Tracts: 11-digit GEOID (state+county+tract)

---

## Outputs

The app writes two files into the output folder:

### Block groups
- `{County}_ACS_BlockGroups_{year}.xlsx`
- `Merged_{County}_BlockGroups_{year}.gpkg`
  - Layer name: `blockgroups_acs_{year}`

### Tracts
- `{County}_Tracts_DP02_DP03_DP04_DP05_{year}.xlsx`
- `Merged_{County}_Tracts_{year}.gpkg`
  - Layer name: `tracts_acs_{year}`

---

## Use the GUI (`app.py`)

```bash
python app.py
```

1. Enter county and state.
2. Select ACS year.
3. Choose output folder.
4. Choose block group or tract shapefile.
5. (Optional) Enter Census API key.
6. Click **Run**.

---

## Use the core pipeline (`census_app_core.py`)

Example (block groups):

```python
from pathlib import Path
from census_app_core import run_blockgroup_pipeline

result = run_blockgroup_pipeline(
    city="Milwaukee",
    state="WI",
    year=2023,
    output_dir=Path("C:/output"),
    shapefile_path=Path("C:/data/USA Census Block Group Boundaries.shp"),
    api_key="YOUR_API_KEY"
)

print(result)
```

Example (tracts):

```python
from pathlib import Path
from census_app_core import run_tract_pipeline

result = run_tract_pipeline(
    city="Suffolk",
    state="NY",
    year=2023,
    output_dir=Path("C:/output"),
    shapefile_path=Path("C:/data/tracts.shp"),
    api_key="YOUR_API_KEY"
)

print(result)
```

---

## Notes
- The pipeline calls the Census API live; internet access is required.
- Large shapefiles are not included in the repo.
