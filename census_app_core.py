from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

STATE_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}


BLOCKGROUP_TABLES = {
    "B01001": "Age by Sex",
    "B02001": "Race",
    "B03002": "Hispanic Latino by Race",
    "B08301": "Means of Transportation to Work",
    "B08303": "Travel Time to Work",
    "B09002": "Own Children Under 18 by Family Type",
    "B11012": "Households by Type",
    "B15003": "Educational Attainment (25+)",
    "B17021": "Poverty Status by Housing Unit",
    "B19001": "Household Income Brackets",
    "B19013": "Median Household Income",
    "B23025": "Employment Status (16+)",
    "B25001": "Housing Units",
    "B25044": "Tenure by Vehicles Available",
}

PROFILE_GROUPS = ["DP02", "DP03", "DP04", "DP05"]


@dataclass
class RunResult:
    output_xlsx: Path
    output_gpkg: Path
    output_layer: str
    county_name: str
    state_fips: str
    county_fips: str
    year: int
    geography: str


def sanitize_sheet_name(name: str) -> str:
    return re.sub(r"[:\\/*?\[\]]", "", name)[:31]


def sanitize_field_names(labels: dict, max_len: int = 63) -> dict:
    safe = {}
    used = set()
    for key, label in labels.items():
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")
        if not cleaned:
            cleaned = key
        cleaned = cleaned[:max_len]
        base = cleaned
        suffix = 1
        while cleaned in used:
            suffix_str = f"_{suffix}"
            cleaned = f"{base[: max_len - len(suffix_str)]}{suffix_str}"
            suffix += 1
        used.add(cleaned)
        safe[key] = cleaned
    return safe


def sanitize_columns(columns, keep: set, max_len: int = 63) -> dict:
    mapping = {}
    used = set(keep)
    for col in columns:
        if col in keep:
            mapping[col] = col
            continue
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        if not cleaned:
            cleaned = "field"
        cleaned = cleaned[:max_len]
        base = cleaned
        suffix = 1
        while cleaned in used:
            suffix_str = f"_{suffix}"
            cleaned = f"{base[: max_len - len(suffix_str)]}{suffix_str}"
            suffix += 1
        used.add(cleaned)
        mapping[col] = cleaned
    return mapping


def sum_variables(df: pd.DataFrame, variables: list[str]) -> float:
    existing = [v for v in variables if v in df.columns]
    if not existing:
        return 0.0
    values = df[existing].apply(pd.to_numeric, errors="coerce")
    return float(values.sum().sum())


def age_bucket_from_label(label: str) -> Optional[str]:
    label_lower = label.lower()
    if not label_lower.startswith("estimate!!total"):
        return None
    if label_lower.endswith("!!male") or label_lower.endswith("!!female"):
        return None
    if label_lower.endswith("!!total"):
        return None
    under_18 = ["under 5 years", "5 to 9 years", "10 to 14 years", "15 to 17 years"]
    age_18_64 = [
        "18 and 19 years",
        "20 years",
        "21 years",
        "22 to 24 years",
        "25 to 29 years",
        "30 to 34 years",
        "35 to 39 years",
        "40 to 44 years",
        "45 to 49 years",
        "50 to 54 years",
        "55 to 59 years",
        "60 and 61 years",
        "62 to 64 years",
    ]
    age_65_plus = [
        "65 and 66 years",
        "67 to 69 years",
        "70 to 74 years",
        "75 to 79 years",
        "80 to 84 years",
        "85 years and over",
    ]
    if any(a in label_lower for a in under_18):
        return "Under 18"
    if any(a in label_lower for a in age_18_64):
        return "18-64"
    if any(a in label_lower for a in age_65_plus):
        return "65+"
    return None


def clean_category_label(label: str) -> str:
    cleaned = label.replace("Estimate!!Total:!!", "")
    cleaned = cleaned.replace("Estimate!!Total:", "")
    cleaned = cleaned.replace("!!", " - ")
    return cleaned.strip(" -")


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def get_group_variables_and_labels(year: int, group_id: str, profile: bool = False) -> tuple[list, dict]:
    if profile:
        url = f"https://api.census.gov/data/{year}/acs/acs5/profile/groups/{group_id}.json"
    else:
        url = f"https://api.census.gov/data/{year}/acs/acs5/groups/{group_id}.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    group_info = resp.json()
    variables = [v for v in group_info["variables"].keys() if v.endswith("E")]
    labels = {
        var_name: group_info["variables"][var_name]["label"]
        for var_name in variables
    }
    return variables, labels


def fetch_blockgroup_table(
    year: int,
    state_fips: str,
    county_fips: str,
    api_key: str,
    table_id: str,
    variables: list,
    logger: Callable[[str], None],
) -> pd.DataFrame:
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    full_df = None
    chunks = list(chunk_list(variables, 49))

    for i, var_chunk in enumerate(chunks, start=1):
        geo_params = {
            "for": "block group:*",
            "in": f"state:{state_fips} county:{county_fips} tract:*",
        }
        params = {
            "get": ",".join(var_chunk + ["NAME"]),
            "key": api_key,
            **geo_params,
        }

        logger(f"Fetching {table_id} chunk {i}/{len(chunks)}")
        resp = requests.get(base_url, params=params, timeout=60)
        if resp.status_code != 200:
            logger(f"ERROR: {table_id} chunk {i} failed: {resp.status_code} {resp.text[:200]}")
            continue

        data = resp.json()
        if not data or len(data) < 2:
            logger(f"WARN: empty response for {table_id} chunk {i}")
            continue

        chunk_df = pd.DataFrame(data[1:], columns=data[0])
        if chunk_df.columns.duplicated().any():
            chunk_df = chunk_df.loc[:, ~chunk_df.columns.duplicated(keep="first")]

        numeric_cols = [c for c in var_chunk if c in chunk_df.columns and c != "NAME"]
        if numeric_cols:
            chunk_df[numeric_cols] = chunk_df[numeric_cols].apply(
                pd.to_numeric, errors="coerce"
            )

        if full_df is None:
            full_df = chunk_df
        else:
            if "NAME" in chunk_df.columns:
                chunk_df = chunk_df.drop(columns=["NAME"])
            full_df = full_df.merge(
                chunk_df, on=["state", "county", "tract", "block group"], how="inner"
            )

    if full_df is None or full_df.empty:
        return pd.DataFrame()

    full_df.rename(columns={"NAME": "Geographic Area Name"}, inplace=True)
    full_df["state"] = full_df["state"].astype(str).str.zfill(2)
    full_df["county"] = full_df["county"].astype(str).str.zfill(3)
    full_df["tract"] = full_df["tract"].astype(str).str.zfill(6)
    full_df["block group"] = full_df["block group"].astype(str).str.zfill(1)
    full_df["GEOID"] = (
        full_df["state"] + full_df["county"] + full_df["tract"] + full_df["block group"]
    )
    full_df["Year"] = year
    return full_df


def clean_numeric_str(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s in {"", "-", "N", "NA", "N/A", "(X)", "(x)", "*", "null", "None"}:
        return np.nan
    s = s.replace(",", "")
    s = s.replace("%", "")
    s = re.sub(r"\s+", "", s)
    return s


def clean_sheet(df: pd.DataFrame, geoid_len: int, sheet_name: str) -> pd.DataFrame:
    df = df.copy()
    if "GEOID" not in df.columns:
        if geoid_len == 12:
            needed = ["state", "county", "tract", "block group"]
            if all(c in df.columns for c in needed):
                df["state"] = df["state"].astype(str).str.zfill(2)
                df["county"] = df["county"].astype(str).str.zfill(3)
                df["tract"] = df["tract"].astype(str).str.zfill(6)
                df["block group"] = df["block group"].astype(str).str.zfill(1)
                df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
            else:
                raise ValueError(
                    f"Sheet '{sheet_name}' has no GEOID and cannot build from components."
                )
        elif geoid_len == 11:
            needed = ["state", "county", "tract"]
            if all(c in df.columns for c in needed):
                df["state"] = df["state"].astype(str).str.zfill(2)
                df["county"] = df["county"].astype(str).str.zfill(3)
                df["tract"] = df["tract"].astype(str).str.zfill(6)
                df["GEOID"] = df["state"] + df["county"] + df["tract"]
            else:
                raise ValueError(
                    f"Sheet '{sheet_name}' has no GEOID and cannot build from components."
                )
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(geoid_len)
    drop_geo_parts = [c for c in ["state", "county", "tract", "block group"] if c in df.columns]
    if drop_geo_parts:
        df.drop(columns=drop_geo_parts, inplace=True)
    return df


def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def fetch_profile_group(
    year: int,
    state_fips: str,
    county_fips: str,
    api_key: str,
    group_id: str,
    variables: list,
    logger: Callable[[str], None],
) -> pd.DataFrame:
    base_url = f"https://api.census.gov/data/{year}/acs/acs5/profile"
    geo_params = {"for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"}

    full_df = None
    chunks = list(chunk_list(variables, 49))

    for i, var_chunk in enumerate(chunks, start=1):
        logger(f"Fetching {group_id} chunk {i}/{len(chunks)}")
        params = {"get": ",".join(var_chunk + ["NAME"]), "key": api_key, **geo_params}
        resp = requests.get(base_url, params=params, timeout=60)
        if resp.status_code != 200:
            logger(f"ERROR: {group_id} chunk {i} failed: {resp.status_code} {resp.text[:200]}")
            continue
        data = resp.json()
        if len(data) < 2:
            continue

        df = pd.DataFrame(data[1:], columns=data[0])
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        numeric_cols = [c for c in var_chunk if c in df.columns]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        if full_df is None:
            full_df = df
        else:
            full_df = full_df.merge(
                df.drop(columns=["NAME"]), on=["state", "county", "tract"], how="inner"
            )

    if full_df is None or full_df.empty:
        return pd.DataFrame()

    full_df.rename(columns={"NAME": "Geographic Area Name"}, inplace=True)
    full_df["state"] = full_df["state"].astype(str).str.zfill(2)
    full_df["county"] = full_df["county"].astype(str).str.zfill(3)
    full_df["tract"] = full_df["tract"].astype(str).str.zfill(6)
    full_df["GEOID"] = full_df["state"] + full_df["county"] + full_df["tract"]
    full_df["Year"] = year
    return full_df


def geocode_city_state(city: str, state: str) -> tuple[str, str, str]:
    city = city.strip()
    state = state.strip()

    if "," in city and not state:
        parts = [p.strip() for p in city.split(",", 1)]
        if len(parts) == 2:
            city, state = parts

    city_clean = re.sub(r"\bcity\b", "", city, flags=re.IGNORECASE).strip() or city

    candidates = [
        f"{city_clean}, {state}".strip(", "),
        f"City Hall, {city_clean}, {state}".strip(", "),
        f"1 City Hall, {city_clean}, {state}".strip(", "),
        f"1 Main St, {city_clean}, {state}".strip(", "),
    ]

    url = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    for address in candidates:
        params = {
            "address": address,
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if not matches:
            continue
        geos = matches[0].get("geographies", {}).get("Counties", [])
        if not geos:
            continue
        county = geos[0]
        state_fips = county["STATE"]
        county_fips = county["COUNTY"]
        county_name = county["NAME"]
        return state_fips, county_fips, county_name

    # Fallback: if the city name matches a county name in the state, use that.
    state_lookup = state
    if len(state) == 2:
        state_lookup = STATE_ABBR_TO_NAME.get(state.upper(), state)
    try:
        fips_path = Path(__file__).resolve().parent / "US_FIPS_Codes.xlsx"
        fips_df = pd.read_excel(fips_path)
        state_mask = fips_df["State"].str.strip().str.casefold() == state_lookup.strip().casefold()
        state_df = fips_df[state_mask].copy()
        if not state_df.empty:
            target = city_clean.strip().casefold()
            county_norm = (
                state_df["County Name"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+county$", "", regex=True)
                .str.casefold()
            )
            matches = state_df[county_norm == target]
            if not matches.empty:
                row = matches.iloc[0]
                state_fips = str(row["FIPS State"]).zfill(2)
                county_fips = str(row["FIPS County"]).zfill(3)
                county_name = str(row["County Name"])
                return state_fips, county_fips, county_name
    except Exception:
        pass

    raise ValueError(
        "No address match found. Try a different city/state or use the county name."
    )


def run_blockgroup_pipeline(
    city: str,
    state: str,
    year: int,
    output_dir: Path,
    shapefile_path: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")

    tables = tables or BLOCKGROUP_TABLES
    state_fips, county_fips, county_name_raw = geocode_city_state(city, state)
    county_name = normalize_name(county_name_raw.replace(" County", ""))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_xlsx = output_dir / f"{county_name}_ACS_BlockGroups_{year}.xlsx"
    output_gpkg = output_dir / f"Merged_{county_name}_BlockGroups_{year}.gpkg"
    output_layer = f"blockgroups_acs_{year}"

    logger(f"Using county: {county_name_raw} (state {state_fips}, county {county_fips})")
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    summary = {
        "total_population": None,
        "sex": {},
        "age": {},
        "travel_time": [],
        "hispanic_race": [],
    }
    for table_id, sheet_name in tables.items():
        variables, labels = get_group_variables_and_labels(year, table_id)
        if not variables:
            logger(f"WARN: no variables found for {table_id}")
            continue

        df = fetch_blockgroup_table(
            year, state_fips, county_fips, api_key, table_id, variables, logger
        )
        if df.empty:
            logger(f"WARN: no data returned for {table_id}")
            continue

        if table_id == "B01001":
            total_var = [v for v, label in labels.items() if label == "Estimate!!Total"]
            summary["total_population"] = sum_variables(df, total_var)

            male_var = [
                v
                for v, label in labels.items()
                if label.startswith("Estimate!!Total:!!Male") and label.endswith("!!Male")
            ]
            female_var = [
                v
                for v, label in labels.items()
                if label.startswith("Estimate!!Total:!!Female") and label.endswith("!!Female")
            ]
            if male_var:
                summary["sex"]["Male"] = sum_variables(df, male_var)
            if female_var:
                summary["sex"]["Female"] = sum_variables(df, female_var)

            age_totals = {"Under 18": 0.0, "18-64": 0.0, "65+": 0.0}
            for var, label in labels.items():
                bucket = age_bucket_from_label(label)
                if bucket:
                    age_totals[bucket] += sum_variables(df, [var])
            summary["age"] = age_totals

        if table_id == "B08303":
            travel_time = []
            for var, label in labels.items():
                if not label.startswith("Estimate!!Total"):
                    continue
                if label == "Estimate!!Total":
                    continue
                travel_time.append((clean_category_label(label), sum_variables(df, [var])))
            summary["travel_time"] = travel_time

        if table_id == "B03002":
            hispanic = []
            for var, label in labels.items():
                if not label.startswith("Estimate!!Total"):
                    continue
                if label == "Estimate!!Total":
                    continue
                hispanic.append((clean_category_label(label), sum_variables(df, [var])))
            summary["hispanic_race"] = hispanic

        safe_labels = sanitize_field_names(labels)
        df.rename(columns=safe_labels, inplace=True)
        df.to_excel(writer, sheet_name=sanitize_sheet_name(sheet_name), index=False)

    workbook = writer.book
    summary_sheet = "Executive Summary"
    summary_ws = workbook.add_worksheet(summary_sheet)
    writer.sheets[summary_sheet] = summary_ws
    header_format = workbook.add_format({"bold": True})
    number_format = workbook.add_format({"num_format": "#,##0"})

    row = 0
    summary_ws.write(row, 0, "Executive Summary", header_format)
    row += 2

    def write_summary_section(title: str, data: list[tuple[str, float]], chart_type: str = "column"):
        nonlocal row
        if not data:
            return
        summary_ws.write(row, 0, title, header_format)
        row += 1
        df = pd.DataFrame(data, columns=["Category", "Value"])
        df.to_excel(writer, sheet_name=summary_sheet, startrow=row, startcol=0, index=False)
        start_row = row
        end_row = row + len(df)
        summary_ws.set_column(0, 0, 40)
        summary_ws.set_column(1, 1, 16, number_format)
        chart = workbook.add_chart({"type": chart_type})
        chart.add_series(
            {
                "name": title,
                "categories": [summary_sheet, start_row + 1, 0, end_row, 0],
                "values": [summary_sheet, start_row + 1, 1, end_row, 1],
            }
        )
        chart.set_title({"name": title})
        chart.set_legend({"none": True})
        summary_ws.insert_chart(start_row, 3, chart, {"x_scale": 1.1, "y_scale": 1.1})
        row = end_row + 3

    pop_data = []
    if summary["total_population"] is not None:
        pop_data.append(("Total population", summary["total_population"]))
    if summary["sex"]:
        for key in ["Male", "Female"]:
            if key in summary["sex"]:
                pop_data.append((key, summary["sex"][key]))
    write_summary_section("Population and Sex", pop_data)

    age_data = [(k, v) for k, v in summary["age"].items() if v]
    write_summary_section("Age Distribution", age_data)

    write_summary_section("Travel Time to Work", summary["travel_time"])
    write_summary_section("Hispanic or Latino by Race", summary["hispanic_race"])

    writer.close()

    logger("Reading Excel workbook for merge...")
    all_sheets = pd.read_excel(output_xlsx, sheet_name=None, dtype=str)

    sheets_to_use = list(tables.values())
    missing = [s for s in sheets_to_use if s not in all_sheets]
    if missing:
        logger(f"WARN: missing sheets: {missing}")

    sheet_frames = {s: all_sheets[s] for s in sheets_to_use if s in all_sheets}
    cleaned = {name: clean_sheet(df, 12, name) for name, df in sheet_frames.items()}
    sheet_names = list(cleaned.keys())
    if not sheet_names:
        raise ValueError("No valid sheets to merge.")

    merged_table = cleaned[sheet_names[0]].copy()
    keep_geo_name = "Geographic Area Name" in merged_table.columns

    for name in sheet_names[1:]:
        df = cleaned[name].copy()
        if "Geographic Area Name" in df.columns:
            if keep_geo_name:
                df = df.drop(columns=["Geographic Area Name"])
            else:
                keep_geo_name = True
        cols_in_common = set(merged_table.columns).intersection(df.columns) - {"GEOID"}
        if cols_in_common:
            df = df.rename(columns={c: f"{name}_{c}" for c in cols_in_common})
        merged_table = merged_table.merge(df, on="GEOID", how="outer")

    cols = ["GEOID"] + [c for c in merged_table.columns if c != "GEOID"]
    merged_table = merged_table[cols]
    merged_table = merged_table.rename(
        columns=sanitize_columns(merged_table.columns, {"GEOID", "Geographic Area Name"})
    )

    logger(f"Reading geometry: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    if "FIPS" in gdf.columns and "GEOID" not in gdf.columns:
        gdf = gdf.rename(columns={"FIPS": "GEOID"})
    elif "GEOID" not in gdf.columns:
        raise ValueError("Shapefile must contain a FIPS or GEOID column.")

    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(12)
    merged_gdf = gdf.merge(merged_table, on="GEOID", how="inner")

    non_numeric_keep = {"GEOID", "Geographic Area Name", "geometry"}
    for col in merged_gdf.columns:
        if col in non_numeric_keep:
            continue
        s = merged_gdf[col]
        if s.dtype == object:
            cleaned = s.map(clean_numeric_str)
            nums = pd.to_numeric(cleaned, errors="coerce")
            if nums.notna().any():
                merged_gdf[col] = nums.astype("float64")
        elif pd.api.types.is_integer_dtype(s):
            merged_gdf[col] = s.astype("float64")

    merged_gdf = merged_gdf.to_crs("EPSG:3071")

    logger(f"Saving GeoPackage: {output_gpkg}")
    merged_gdf.to_file(output_gpkg, layer=output_layer, driver="GPKG")
    logger("Done.")

    return RunResult(
        output_xlsx=output_xlsx,
        output_gpkg=output_gpkg,
        output_layer=output_layer,
        county_name=county_name_raw,
        state_fips=state_fips,
        county_fips=county_fips,
        year=year,
        geography="block_groups",
    )


def run_tract_pipeline(
    city: str,
    state: str,
    year: int,
    output_dir: Path,
    shapefile_path: Path,
    api_key: Optional[str] = None,
    groups: Optional[list[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")

    groups = groups or PROFILE_GROUPS
    state_fips, county_fips, county_name_raw = geocode_city_state(city, state)
    county_name = normalize_name(county_name_raw.replace(" County", ""))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_xlsx = output_dir / f"{county_name}_Tracts_{'_'.join(groups)}_{year}.xlsx"
    output_gpkg = output_dir / f"Merged_{county_name}_Tracts_{year}.gpkg"
    output_layer = f"tracts_acs_{year}"

    logger(f"Using county: {county_name_raw} (state {state_fips}, county {county_fips})")
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    for group_id in groups:
        variables, labels = get_group_variables_and_labels(year, group_id, profile=True)
        if not variables:
            logger(f"WARN: no variables found for {group_id}")
            continue

        df = fetch_profile_group(
            year, state_fips, county_fips, api_key, group_id, variables, logger
        )
        if df.empty:
            logger(f"WARN: no data returned for {group_id}")
            continue

        safe_labels = sanitize_field_names(labels)
        df.rename(columns=safe_labels, inplace=True)
        df.to_excel(writer, sheet_name=sanitize_sheet_name(group_id), index=False)
    writer.close()

    logger("Reading Excel workbook for merge...")
    all_sheets = pd.read_excel(output_xlsx, sheet_name=None, dtype=str)
    if not all_sheets:
        raise ValueError("No sheets found in the tract workbook.")

    cleaned = {name: clean_sheet(df, 11, name) for name, df in all_sheets.items()}
    sheet_names = list(cleaned.keys())
    merged_table = cleaned[sheet_names[0]].copy()
    keep_geo_name = "Geographic Area Name" in merged_table.columns

    for name in sheet_names[1:]:
        df = cleaned[name].copy()
        if "Geographic Area Name" in df.columns:
            if keep_geo_name:
                df = df.drop(columns=["Geographic Area Name"])
            else:
                keep_geo_name = True
        cols_in_common = set(merged_table.columns).intersection(df.columns) - {"GEOID"}
        if cols_in_common:
            df = df.rename(columns={c: f"{name}_{c}" for c in cols_in_common})
        merged_table = merged_table.merge(df, on="GEOID", how="outer")

    cols = ["GEOID"] + [c for c in merged_table.columns if c != "GEOID"]
    merged_table = merged_table[cols]
    merged_table = merged_table.rename(
        columns=sanitize_columns(merged_table.columns, {"GEOID", "Geographic Area Name"})
    )

    logger(f"Reading geometry: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    if "GEOID" not in gdf.columns:
        possible_cols = [col for col in gdf.columns if "GEOID" in col or "FIPS" in col]
        raise ValueError(f"Shapefile must contain GEOID. Found: {possible_cols}")

    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(11)
    merged_gdf = gdf.merge(merged_table, on="GEOID", how="inner")

    non_numeric_keep = {"GEOID", "Geographic Area Name", "geometry"}
    for col in merged_gdf.columns:
        if col in non_numeric_keep:
            continue
        s = merged_gdf[col]
        if s.dtype == object:
            cleaned = s.map(clean_numeric_str)
            nums = pd.to_numeric(cleaned, errors="coerce")
            if nums.notna().any():
                merged_gdf[col] = nums.astype("float64")
        elif pd.api.types.is_integer_dtype(s):
            merged_gdf[col] = s.astype("float64")

    logger(f"Saving GeoPackage: {output_gpkg}")
    merged_gdf.to_file(output_gpkg, layer=output_layer, driver="GPKG")
    logger("Done.")

    return RunResult(
        output_xlsx=output_xlsx,
        output_gpkg=output_gpkg,
        output_layer=output_layer,
        county_name=county_name_raw,
        state_fips=state_fips,
        county_fips=county_fips,
        year=year,
        geography="tracts",
    )
