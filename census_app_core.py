from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import time
import zipfile
from urllib.parse import urljoin
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
    "B11012": "Households by Type",
    "B15003": "Educational Attainment (25+)",
    "B17021": "Poverty Status by Housing Unit",
    "B19001": "Household Income Brackets",
    "B19013": "Median Household Income",
    "B23025": "Employment Status (16+)",
    "B25001": "Housing Units",
    "B25044": "Tenure by Vehicles Available",
}

PROFILE_GROUPS = ["DP05"]

LIMITED_ENGLISH_TABLE_VARS = {
    "B16002_004E": "Limited English speaking household",
    "B16002_005E": "Not a limited English speaking household",
}

ACS5_TABLE_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "B11012": (2014, 2024),
    "B16002": (2014, 2024),
    "B15003": (2014, 2024),
    "B23025": (2014, 2024),
}

ACS5_TABLE_YEAR_EXCLUSIONS: dict[str, set[int]] = {
    "B11012": {2015, 2016, 2017, 2018},
    "B16002": {2016, 2017, 2018, 2019, 2021},
    "B15003": set(),
    "B23025": set(),
}

ACS5_2010_BLOCKGROUP_SUMMARY_FILE_LAYOUTS: dict[str, tuple[int, int, int]] = {
    "B01001": (10, 7, 55),
    "B03002": (13, 38, 58),
    "B08301": (30, 89, 109),
    "B08303": (30, 125, 137),
    "B11012": (33, 116, 130),
    "B16002": (42, 7, 20),
    "B17021": (51, 41, 75),
    "B19001": (53, 7, 23),
    "B19013": (53, 177, 177),
    "B25001": (95, 7, 7),
    "B25044": (97, 111, 125),
}

ACS5_2010_SUMMARY_FILE_ROOT = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2010/data/5_year_seq_by_state"
)

ACS1_COUNTY_PLACE_TABLES = {
    "DP05": "DP05 Demographics",
    "B01001": "Age by Sex",
    "B11001": "Household Type (Including Living Alone)",
    "B15003": "Educational Attainment",
    "B23025": "Employment Status",
    "B08303": "Travel Time to Work",
    "S1810": "Disability Status",
    "S2501": "Household Number",
    "B19001": "Household Income",
    "B25044": "Vehicle Ownership",
    "B08006": "Means of Transportation",
    "S1602": "Limited English Speaking",
}

ACS1_DP05_SELECTED_VARS_2010_2016 = {
    "DP05_0001E": "Total Population",
    "DP05_0017E": "Median Age",
    "DP05_0018E": "18 Years and Over",
    "DP05_0021E": "65 Years and Over",
    "DP05_0032E": "White alone",
    "DP05_0033E": "Black or African American alone",
    "DP05_0034E": "American Indian and Alaska Native alone",
    "DP05_0039E": "Asian alone",
    "DP05_0047E": "Native Hawaiian and Other Pacific Islander alone",
    "DP05_0052E": "Some other race",
    "DP05_0053E": "Two or More Races",
    "DP05_0066E": "Hispanic or Latino",
    "DP05_0072E": "White (Not H/L)",
    "DP05_0073E": "Black or African American (Not H/L)",
    "DP05_0074E": "American Indian and Alaska Native (Not H/L)",
    "DP05_0075E": "Asian (Not H/L)",
    "DP05_0076E": "Native Hawaiian and Other Pacific Islander (Not H/L)",
    "DP05_0077E": "Some other race (Not H/L)",
    "DP05_0078E": "Two or More Races (Not H/L)",
    "DP05_0081E": "Housing Units",
}

ACS1_DP05_SELECTED_VARS_2017 = {
    "DP05_0001E": "Total Population",
    "DP05_0018E": "Median Age",
    "DP05_0025E": "18 Years and Over",
    "DP05_0029E": "65 Years and Over",
    "DP05_0037E": "White alone",
    "DP05_0038E": "Black or African American alone",
    "DP05_0039E": "American Indian and Alaska Native alone",
    "DP05_0044E": "Asian alone",
    "DP05_0052E": "Native Hawaiian and Other Pacific Islander alone",
    "DP05_0057E": "Some other race",
    "DP05_0035E": "Two or More Races",
    "DP05_0071E": "Hispanic or Latino",
    "DP05_0077E": "White (Not H/L)",
    "DP05_0078E": "Black or African American (Not H/L)",
    "DP05_0079E": "American Indian and Alaska Native (Not H/L)",
    "DP05_0080E": "Asian (Not H/L)",
    "DP05_0081E": "Native Hawaiian and Other Pacific Islander (Not H/L)",
    "DP05_0082E": "Some other race (Not H/L)",
    "DP05_0083E": "Two or More Races (Not H/L)",
    "DP05_0086E": "Housing Units",
}

ACS1_DP05_SELECTED_VARS_2018 = dict(ACS1_DP05_SELECTED_VARS_2017)

ACS1_DP05_SELECTED_VARS_2019 = dict(ACS1_DP05_SELECTED_VARS_2017)

ACS1_DP05_SELECTED_VARS_2021 = dict(ACS1_DP05_SELECTED_VARS_2017)

ACS1_DP05_SELECTED_VARS_2022 = {
    "DP05_0001E": "Total Population",
    "DP05_0018E": "Median Age",
    "DP05_0025E": "18 Years and Over",
    "DP05_0029E": "65 Years and Over",
    "DP05_0037E": "White alone",
    "DP05_0038E": "Black or African American alone",
    "DP05_0039E": "American Indian and Alaska Native alone",
    "DP05_0044E": "Asian alone",
    "DP05_0052E": "Native Hawaiian and Other Pacific Islander alone",
    "DP05_0060E": "Some other race",
    "DP05_0061E": "Two or More Races",
    "DP05_0074E": "Hispanic or Latino",
    "DP05_0079E": "White (Not H/L)",
    "DP05_0080E": "Black or African American (Not H/L)",
    "DP05_0081E": "American Indian and Alaska Native (Not H/L)",
    "DP05_0082E": "Asian (Not H/L)",
    "DP05_0083E": "Native Hawaiian and Other Pacific Islander (Not H/L)",
    "DP05_0084E": "Some other race (Not H/L)",
    "DP05_0085E": "Two or More Races (Not H/L)",
    "DP05_0088E": "Housing Units",
}

ACS1_DP05_SELECTED_VARS_2023 = {
    "DP05_0001E": "Total Population",
    "DP05_0018E": "Median Age",
    "DP05_0025E": "18 Years and Over",
    "DP05_0029E": "65 Years and Over",
    "DP05_0037E": "White alone",
    "DP05_0038E": "Black or African American alone",
    "DP05_0039E": "American Indian and Alaska Native alone",
    "DP05_0047E": "Asian alone",
    "DP05_0055E": "Native Hawaiian and Other Pacific Islander alone",
    "DP05_0060E": "Some other race",
    "DP05_0061E": "Two or More Races",
    "DP05_0077E": "Hispanic or Latino",
    "DP05_0082E": "White (Not H/L)",
    "DP05_0083E": "Black or African American (Not H/L)",
    "DP05_0084E": "American Indian and Alaska Native (Not H/L)",
    "DP05_0085E": "Asian (Not H/L)",
    "DP05_0086E": "Native Hawaiian and Other Pacific Islander (Not H/L)",
    "DP05_0087E": "Some other race (Not H/L)",
    "DP05_0088E": "Two or More Races (Not H/L)",
    "DP05_0091E": "Housing Units",
}

ACS1_DP05_SELECTED_VARS_2024 = {
    "DP05_0001E": "Total Population",
    "DP05_0018E": "Median Age",
    "DP05_0025E": "18 Years and Over",
    "DP05_0029E": "65 Years and Over",
    "DP05_0037E": "White alone",
    "DP05_0045E": "Black or African American alone",
    "DP05_0053E": "American Indian and Alaska Native alone",
    "DP05_0061E": "Asian alone",
    "DP05_0069E": "Native Hawaiian and Other Pacific Islander alone",
    "DP05_0074E": "Some other race",
    "DP05_0075E": "Two or More Races",
    "DP05_0090E": "Hispanic or Latino",
    "DP05_0096E": "White (Not H/L)",
    "DP05_0097E": "Black or African American (Not H/L)",
    "DP05_0098E": "American Indian and Alaska Native (Not H/L)",
    "DP05_0099E": "Asian (Not H/L)",
    "DP05_0100E": "Native Hawaiian and Other Pacific Islander (Not H/L)",
    "DP05_0101E": "Some other race (Not H/L)",
    "DP05_0102E": "Two or More Races (Not H/L)",
    "DP05_0105E": "Housing Units",
}

ACS1_DP05_SELECTED_VARS_BY_YEAR = {
    2010: dict(ACS1_DP05_SELECTED_VARS_2010_2016),
    2011: dict(ACS1_DP05_SELECTED_VARS_2010_2016),
    2012: dict(ACS1_DP05_SELECTED_VARS_2010_2016),
    2013: dict(ACS1_DP05_SELECTED_VARS_2010_2016),
    2014: dict(ACS1_DP05_SELECTED_VARS_2010_2016),
    2015: dict(ACS1_DP05_SELECTED_VARS_2010_2016),
    2016: dict(ACS1_DP05_SELECTED_VARS_2010_2016),
    2017: dict(ACS1_DP05_SELECTED_VARS_2017),
    2018: dict(ACS1_DP05_SELECTED_VARS_2018),
    2019: dict(ACS1_DP05_SELECTED_VARS_2019),
    2021: dict(ACS1_DP05_SELECTED_VARS_2021),
    2022: dict(ACS1_DP05_SELECTED_VARS_2022),
    2023: dict(ACS1_DP05_SELECTED_VARS_2023),
    2024: dict(ACS1_DP05_SELECTED_VARS_2024),
}

ACS1_S2501_SELECTED_VARS_2010 = {
    "S2501_C01_001E": "Total Occupied Housing Units",
    "S2501_C01_002E": "1-person household",
    "S2501_C01_003E": "2-person household",
    "S2501_C01_004E": "3-person household",
    "S2501_C01_005E": "4-or more person household",
    "S1101_C01_002E": "Average household size",
}

ACS1_S2501_SELECTED_VARS_2011_2021 = {
    "S2501_C01_001E": "Total Occupied Housing Units",
    "S2501_C01_002E": "1-person household",
    "S2501_C01_003E": "2-person household",
    "S2501_C01_004E": "3-person household",
    "S2501_C01_005E": "4-or more person household",
    "S1101_C01_002E": "Average household size",
}

ACS1_B19001_SELECTED_VARS = {
    "B19001_001E": "Total Households",
    "B19001_002E": "Less than $10,000",
    "B19001_003E": "$10,000 to $14,999",
    "B19001_004E": "$15,000 to $19,999",
    "B19001_005E": "$20,000 to $24,999",
    "B19001_006E": "$25,000 to $29,999",
    "B19001_007E": "$30,000 to $34,999",
    "B19001_008E": "$35,000 to $39,999",
    "B19001_009E": "$40,000 to $44,999",
    "B19001_010E": "$45,000 to $49,999",
    "B19001_011E": "$50,000 to $59,999",
    "B19001_012E": "$60,000 to $74,999",
    "B19001_013E": "$75,000 to $99,999",
    "B19001_014E": "$100,000 to $124,999",
    "B19001_015E": "$125,000 to $149,999",
    "B19001_016E": "$150,000 to $199,999",
    "B19001_017E": "$200,000 or more",
}

ACS1_B23025_SELECTED_VARS = {
    "B23025_001E": "Total Population 16 Years and Over",
    "B23025_002E": "In Labor Force",
    "B23025_005E": "Employed",
    "B23025_007E": "Unemployed",
    "B23025_010E": "Not in Labor Force",
}

ACS1_B08303_SELECTED_VARS = {
    "B08303_001E": "Total",
    "B08303_002E": "Less than 5 minutes",
    "B08303_003E": "5 to 9 minutes",
    "B08303_004E": "10 to 14 minutes",
    "B08303_005E": "15 to 19 minutes",
    "B08303_006E": "20 to 24 minutes",
    "B08303_007E": "25 to 29 minutes",
    "B08303_008E": "30 to 34 minutes",
    "B08303_009E": "35 to 39 minutes",
    "B08303_010E": "40 to 44 minutes",
    "B08303_011E": "45 to 59 minutes",
    "B08303_012E": "60 to 89 minutes",
    "B08303_013E": "90 or more minutes",
}

ACS1_S1810_SELECTED_VARS = {
    "S1810_C01_001E": "Total Civilian Population",
    "S1810_C02_001E": "With a Disability",
    "S1810_C03_001E": "Percent with a Disability",
}

ACS1_B25044_SELECTED_VARS = {
    "B25044_001E": "Total Occupied Housing Units",
    "B25044_003E": "Owner occupied no vehicle available",
    "B25044_004E": "Owner occupied 1 vehicle available",
    "B25044_005E": "Owner occupied 2 vehicles available",
    "B25044_006E": "Owner occupied 3 vehicles available",
    "B25044_007E": "Owner occupied 4 vehicles available",
    "B25044_008E": "Owner occupied 5 or more vehicles available",
    "B25044_010E": "Renter occupied no vehicle available",
    "B25044_011E": "Renter occupied 1 vehicle available",
    "B25044_012E": "Renter occupied 2 vehicles available",
    "B25044_013E": "Renter occupied 3 vehicles available",
    "B25044_014E": "Renter occupied 4 vehicles available",
    "B25044_015E": "Renter occupied 5 or more vehicles available",
}

ACS1_B08006_SELECTED_VARS = {
    "B08006_001E": "Total",
    "B08006_003E": "Drove alone",
    "B08006_004E": "Carpooled",
    "B08006_008E": "Public transportation",
    "B08006_014E": "Bicycle",
    "B08006_015E": "Walked",
    "B08006_016E": "Taxicab, motorcycle, or other means",
    "B08006_017E": "Worked from home",
}

ACS1_S1602_SELECTED_VARS = {
    "S1602_C01_001E": "Total Households",
    "S1602_C01_002E": "Spanish-speaking households",
    "S1602_C01_003E": "Other Indo-European language households",
    "S1602_C01_004E": "Asian and Pacific Island language households",
    "S1602_C01_005E": "Other language households",
    "S1602_C03_001E": "Limited English-speaking households",
    "S1602_C03_002E": "Limited English-speaking Spanish households",
    "S1602_C03_003E": "Limited English-speaking Other Indo-European households",
    "S1602_C03_004E": "Limited English-speaking Asian and Pacific Island households",
    "S1602_C03_005E": "Limited English-speaking Other language households",
    "S1602_C04_001E": "Limited English-speaking households (%)",
    "S1602_C04_002E": "Limited English-speaking Spanish households (%)",
    "S1602_C04_003E": "Limited English-speaking Other Indo-European households (%)",
    "S1602_C04_004E": "Limited English-speaking Asian and Pacific Island households (%)",
    "S1602_C04_005E": "Limited English-speaking Other language households (%)",
}

ACS1_B01001_SELECTED_VARS = {
    "B01001_001E": "Total Population",
    "B01001_003E": "Male Under 5 Years",
    "B01001_004E": "Male 5 to 9 Years",
    "B01001_005E": "Male 10 to 14 Years",
    "B01001_006E": "Male 15 to 17 Years",
    "B01001_007E": "Male 18 and 19 Years",
    "B01001_008E": "Male 20 Years",
    "B01001_009E": "Male 21 Years",
    "B01001_010E": "Male 22 to 24 Years",
    "B01001_011E": "Male 25 to 29 Years",
    "B01001_012E": "Male 30 to 34 Years",
    "B01001_013E": "Male 35 to 39 Years",
    "B01001_014E": "Male 40 to 44 Years",
    "B01001_015E": "Male 45 to 49 Years",
    "B01001_016E": "Male 50 to 54 Years",
    "B01001_017E": "Male 55 to 59 Years",
    "B01001_018E": "Male 60 and 61 Years",
    "B01001_019E": "Male 62 to 64 Years",
    "B01001_020E": "Male 65 and 66 Years",
    "B01001_021E": "Male 67 to 69 Years",
    "B01001_022E": "Male 70 to 74 Years",
    "B01001_023E": "Male 75 to 79 Years",
    "B01001_024E": "Male 80 to 84 Years",
    "B01001_025E": "Male 85 Years and Over",
    "B01001_027E": "Female Under 5 Years",
    "B01001_028E": "Female 5 to 9 Years",
    "B01001_029E": "Female 10 to 14 Years",
    "B01001_030E": "Female 15 to 17 Years",
    "B01001_031E": "Female 18 and 19 Years",
    "B01001_032E": "Female 20 Years",
    "B01001_033E": "Female 21 Years",
    "B01001_034E": "Female 22 to 24 Years",
    "B01001_035E": "Female 25 to 29 Years",
    "B01001_036E": "Female 30 to 34 Years",
    "B01001_037E": "Female 35 to 39 Years",
    "B01001_038E": "Female 40 to 44 Years",
    "B01001_039E": "Female 45 to 49 Years",
    "B01001_040E": "Female 50 to 54 Years",
    "B01001_041E": "Female 55 to 59 Years",
    "B01001_042E": "Female 60 and 61 Years",
    "B01001_043E": "Female 62 to 64 Years",
    "B01001_044E": "Female 65 and 66 Years",
    "B01001_045E": "Female 67 to 69 Years",
    "B01001_046E": "Female 70 to 74 Years",
    "B01001_047E": "Female 75 to 79 Years",
    "B01001_048E": "Female 80 to 84 Years",
    "B01001_049E": "Female 85 Years and Over",
}




@dataclass
class RunResult:
    output_xlsx: Path
    output_gpkg: Optional[Path]
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


def plain_language_label(label: str) -> str:
    if not label:
        return ""
    text = str(label).strip()
    text = text.replace("Estimate!!", "")
    text = text.replace("Percent Estimate!!", "")
    text = text.replace("!!", " > ")
    text = text.replace(":", " - ")
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def build_data_dictionary_rows(
    table_id: str,
    sheet_name: str,
    labels: dict[str, str],
    safe_labels: dict[str, str],
    geography: str,
    dataset: str,
    year: int,
) -> list[dict]:
    rows = []
    for variable_id, label in labels.items():
        rows.append(
            {
                "Output Field": safe_labels.get(variable_id, variable_id),
                "Output Sheet": sheet_name,
                "Source Table ID": table_id,
                "Source Variable ID": variable_id,
                "Plain Language Definition": plain_language_label(label),
                "Original Census Label": label,
                "Geography": geography,
                "Dataset": dataset,
                "Year": year,
            }
        )
    return rows


def write_data_dictionary_sheet(writer: pd.ExcelWriter, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates().sort_values(
        by=["Year", "Source Table ID", "Source Variable ID", "Output Field"]
    )
    sheet_name = "Data Dictionary"
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 0, 28)
    worksheet.set_column(1, 1, 24)
    worksheet.set_column(2, 3, 18)
    worksheet.set_column(4, 5, 56)
    worksheet.set_column(6, 8, 16)


def sum_variables(df: pd.DataFrame, variables: list[str]) -> float:
    existing = [v for v in variables if v in df.columns]
    if not existing:
        return 0.0
    values = df[existing].apply(pd.to_numeric, errors="coerce")
    return float(values.sum().sum())


def add_acs1_row_total(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    existing = [variable for variable in variables if variable in df.columns]
    if not existing:
        df["total"] = 0.0
        return df
    values = df[existing].apply(pd.to_numeric, errors="coerce")
    df["total"] = values.sum(axis=1)
    return df


def age_bucket_from_label(label: str) -> Optional[str]:
    label_lower = label.lower()
    if not label_lower.startswith("estimate!!total"):
        return None
    if label_lower.endswith("!!male") or label_lower.endswith("!!female"):
        return None
    if label_lower.endswith("!!total"):
        return None

    under_5 = ["under 5 years"]
    age_5_19 = ["5 to 9 years", "10 to 14 years", "15 to 17 years", "18 and 19 years"]
    age_20_34 = ["20 years", "21 years", "22 to 24 years", "25 to 29 years", "30 to 34 years"]
    age_35_49 = ["35 to 39 years", "40 to 44 years", "45 to 49 years"]
    age_50_64 = ["50 to 54 years", "55 to 59 years", "60 and 61 years", "62 to 64 years"]
    age_65_74 = ["65 and 66 years", "67 to 69 years", "70 to 74 years"]
    age_75_84 = ["75 to 79 years", "80 to 84 years"]
    age_85_plus = ["85 years and over"]

    if any(a in label_lower for a in under_5):
        return "Under 5"
    if any(a in label_lower for a in age_5_19):
        return "5 to 19"
    if any(a in label_lower for a in age_20_34):
        return "20 to 34"
    if any(a in label_lower for a in age_35_49):
        return "35 to 49"
    if any(a in label_lower for a in age_50_64):
        return "50 to 64"
    if any(a in label_lower for a in age_65_74):
        return "65 to 74"
    if any(a in label_lower for a in age_75_84):
        return "75 to 84"
    if any(a in label_lower for a in age_85_plus):
        return "85 and Over"
    return None

def clean_category_label(label: str) -> str:
    cleaned = label.replace("Estimate!!Total:!!", "")
    cleaned = cleaned.replace("Estimate!!Total:", "")
    cleaned = cleaned.replace("!!", " - ")
    return cleaned.strip(" -")


def clean_category_column_name(column: str, prefix: str = "Estimate_Total_") -> str:
    label = column
    if label.startswith(prefix):
        label = label[len(prefix):]
    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def sum_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.sum())


def sum_available_columns(df: pd.DataFrame, columns: list[str]) -> float:
    total = 0.0
    for column in columns:
        if column in df.columns:
            total += sum_column(df, column)
    return total


def sum_matching_columns(df: pd.DataFrame, prefixes: list[str]) -> float:
    total = 0.0
    for column in df.columns:
        if any(column.startswith(prefix) for prefix in prefixes):
            total += sum_column(df, column)
    return total


def sum_columns_by_label(
    df: pd.DataFrame,
    label_map: dict[str, str],
    exact_labels: list[str] | None = None,
    prefix_labels: list[str] | None = None,
) -> float:
    exact_labels = exact_labels or []
    prefix_labels = prefix_labels or []
    total = 0.0
    for column_name, label in label_map.items():
        if label in exact_labels or any(label.startswith(prefix) for prefix in prefix_labels):
            total += sum_column(df, column_name)
    return total


def first_available_total(df: pd.DataFrame, candidates: list[str]) -> float:
    for column in candidates:
        if column in df.columns:
            return sum_column(df, column)
    return 0.0


def acs5_table_availability_message(table_id: str, year: int) -> str:
    excluded_years = ACS5_TABLE_YEAR_EXCLUSIONS.get(table_id, set())
    if year in excluded_years:
        return f"WARN: table {table_id} is not published by Census ACS5 for {year}; skipping."
    return f"WARN: table {table_id} is not available for ACS5 {year}; skipping."


def resolve_tract_table_id(table_id: str, year: int) -> str:
    if table_id == "B15003" and year in {2010, 2011, 2012, 2013}:
        return "B15002"
    if table_id == "B08301" and year == 2010:
        return "B08006"
    return table_id


def prepare_trend_output_frame(
    sheet_name: str,
    df: pd.DataFrame,
    year: int,
    *,
    location_value: Optional[str] = None,
) -> pd.DataFrame:
    out_df = df.copy()
    if "Year" in out_df.columns:
        out_df["Year"] = year
    else:
        out_df.insert(0, "Year", year)

    if location_value is not None:
        if "Location" in out_df.columns:
            out_df["Location"] = location_value
        else:
            out_df.insert(0, "Location", location_value)

    if sheet_name == "Median Household Income":
        geo_priority = ["Location", "Year", "GEOID", "Geographic Area Name"]
        value_cols = [
            col
            for col in out_df.columns
            if col not in {"Location", "Year", "GEOID", "Geographic Area Name", "state", "county", "tract", "block group", "place"}
        ]
        if len(value_cols) == 1:
            value_col = value_cols[0]
            out_df = out_df.rename(columns={value_col: "Median Household Income"})
            keep_cols = [col for col in geo_priority if col in out_df.columns] + ["Median Household Income"]
            return out_df[keep_cols]

    ordered_front = [col for col in ["Location", "Year"] if col in out_df.columns]
    ordered_rest = [col for col in out_df.columns if col not in ordered_front]
    return out_df[ordered_front + ordered_rest]


def age_bucket_from_text(text: str) -> Optional[str]:
    normalized = text.lower().replace("_", " ")

    under_5 = ["under 5 years"]
    age_5_19 = ["5 to 9 years", "10 to 14 years", "15 to 17 years", "18 and 19 years"]
    age_20_34 = ["20 years", "21 years", "22 to 24 years", "25 to 29 years", "30 to 34 years"]
    age_35_49 = ["35 to 39 years", "40 to 44 years", "45 to 49 years"]
    age_50_64 = ["50 to 54 years", "55 to 59 years", "60 and 61 years", "62 to 64 years"]
    age_65_74 = ["65 and 66 years", "67 to 69 years", "70 to 74 years"]
    age_75_84 = ["75 to 79 years", "80 to 84 years"]
    age_85_plus = ["85 years and over"]

    if any(a in normalized for a in under_5):
        return "Under 5"
    if any(a in normalized for a in age_5_19):
        return "5 to 19"
    if any(a in normalized for a in age_20_34):
        return "20 to 34"
    if any(a in normalized for a in age_35_49):
        return "35 to 49"
    if any(a in normalized for a in age_50_64):
        return "50 to 64"
    if any(a in normalized for a in age_65_74):
        return "65 to 74"
    if any(a in normalized for a in age_75_84):
        return "75 to 84"
    if any(a in normalized for a in age_85_plus):
        return "85 and Over"
    return None


def build_blockgroup_summary(
    sheet_frames: dict[str, pd.DataFrame], label_maps: dict[str, dict[str, str]]
) -> dict:
    summary = {
        "section_denominators": {},
        "total_population": None,
        "sex": {},
        "age": {},
        "education": [],
        "household_income": [],
        "housing_units": [],
        "housing_occupancy_tenure": [],
        "vehicle_availability": [],
        "means_of_transportation": [],
        "employment_status": [],
        "travel_time": [],
        "hispanic_race": [],
    }

    age_df = sheet_frames.get("Age by Sex")
    if age_df is not None and not age_df.empty:
        summary["total_population"] = sum_column(age_df, "Estimate_Total")
        male_total = sum_column(age_df, "Estimate_Total_Male")
        female_total = sum_column(age_df, "Estimate_Total_Female")
        if male_total:
            summary["sex"]["Male"] = male_total
        if female_total:
            summary["sex"]["Female"] = female_total

        age_totals = {
            "Under 5": 0.0,
            "5 to 19": 0.0,
            "20 to 34": 0.0,
            "35 to 49": 0.0,
            "50 to 64": 0.0,
            "65 to 74": 0.0,
            "75 to 84": 0.0,
            "85 and Over": 0.0,
        }
        for column in age_df.columns:
            if column in {"Estimate_Total", "Estimate_Total_Male", "Estimate_Total_Female"}:
                continue
            suffix = None
            if column.startswith("Estimate_Total_Male_"):
                suffix = column[len("Estimate_Total_Male_") :]
            elif column.startswith("Estimate_Total_Female_"):
                suffix = column[len("Estimate_Total_Female_") :]
            if not suffix:
                continue
            bucket = age_bucket_from_text(suffix)
            if bucket and bucket in age_totals:
                age_totals[bucket] += sum_column(age_df, column)
        summary["age"] = age_totals

    travel_df = sheet_frames.get("Travel Time to Work")
    if travel_df is not None and not travel_df.empty:
        display_labels = label_maps.get("Travel Time to Work", {})
        travel_order = [
            "Less than 5 minutes",
            "5 to 9 minutes",
            "10 to 14 minutes",
            "15 to 19 minutes",
            "20 to 24 minutes",
            "25 to 29 minutes",
            "30 to 34 minutes",
            "35 to 39 minutes",
            "40 to 44 minutes",
            "45 to 59 minutes",
            "60 to 89 minutes",
            "90 or more minutes",
        ]
        travel_lookup = {}
        for column in travel_df.columns:
            if not column.startswith("Estimate_Total_") or column == "Estimate_Total":
                continue
            total = sum_column(travel_df, column)
            if total == 0:
                continue
            label = clean_category_label(
                display_labels.get(column, clean_category_column_name(column))
            )
            travel_lookup[label] = total
        travel_time = [(label, travel_lookup[label]) for label in travel_order if label in travel_lookup]
        summary["travel_time"] = travel_time

    education_df = sheet_frames.get("Educational Attainment (25+)")
    if education_df is not None and not education_df.empty:
        if "Estimate_Total_Male_No_schooling_completed" in education_df.columns:
            education_summary = [
                (
                    "Less than High School Diploma",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Male_No_schooling_completed",
                            "Estimate_Total_Male_Nursery_to_4th_grade",
                            "Estimate_Total_Male_5th_and_6th_grade",
                            "Estimate_Total_Male_7th_and_8th_grade",
                            "Estimate_Total_Male_9th_grade",
                            "Estimate_Total_Male_10th_grade",
                            "Estimate_Total_Male_11th_grade",
                            "Estimate_Total_Male_12th_grade_no_diploma",
                            "Estimate_Total_Female_No_schooling_completed",
                            "Estimate_Total_Female_Nursery_to_4th_grade",
                            "Estimate_Total_Female_5th_and_6th_grade",
                            "Estimate_Total_Female_7th_and_8th_grade",
                            "Estimate_Total_Female_9th_grade",
                            "Estimate_Total_Female_10th_grade",
                            "Estimate_Total_Female_11th_grade",
                            "Estimate_Total_Female_12th_grade_no_diploma",
                        ],
                    ),
                ),
                (
                    "High School Diploma or Equivalent",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Male_High_school_graduate_GED_or_alternative",
                            "Estimate_Total_Female_High_school_graduate_GED_or_alternative",
                        ],
                    ),
                ),
                (
                    "Some College No Degree",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Male_Some_college_less_than_1_year",
                            "Estimate_Total_Male_Some_college_1_or_more_years_no_degree",
                            "Estimate_Total_Female_Some_college_less_than_1_year",
                            "Estimate_Total_Female_Some_college_1_or_more_years_no_degree",
                        ],
                    ),
                ),
                (
                    "Associates Degree",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Male_Associate_s_degree",
                            "Estimate_Total_Female_Associate_s_degree",
                        ],
                    ),
                ),
                (
                    "Bachelors Degree",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Male_Bachelor_s_degree",
                            "Estimate_Total_Female_Bachelor_s_degree",
                        ],
                    ),
                ),
                (
                    "Graduate or Professional Degree",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Male_Master_s_degree",
                            "Estimate_Total_Male_Professional_school_degree",
                            "Estimate_Total_Male_Doctorate_degree",
                            "Estimate_Total_Female_Master_s_degree",
                            "Estimate_Total_Female_Professional_school_degree",
                            "Estimate_Total_Female_Doctorate_degree",
                        ],
                    ),
                ),
            ]
        else:
            education_summary = [
                (
                    "Less than High School Diploma",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_No_schooling_completed",
                            "Estimate_Total_Nursery_school",
                            "Estimate_Total_Kindergarten",
                            "Estimate_Total_1st_grade",
                            "Estimate_Total_2nd_grade",
                            "Estimate_Total_3rd_grade",
                            "Estimate_Total_4th_grade",
                            "Estimate_Total_5th_grade",
                            "Estimate_Total_6th_grade",
                            "Estimate_Total_7th_grade",
                            "Estimate_Total_8th_grade",
                            "Estimate_Total_9th_grade",
                            "Estimate_Total_10th_grade",
                            "Estimate_Total_11th_grade",
                            "Estimate_Total_12th_grade_no_diploma",
                        ],
                    ),
                ),
                (
                    "High School Diploma or Equivalent",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_High_school_graduate_includes_equivalency",
                            "Estimate_Total_Regular_high_school_diploma",
                            "Estimate_Total_GED_or_alternative_credential",
                        ],
                    ),
                ),
                (
                    "Some College No Degree",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Some_college_less_than_1_year",
                            "Estimate_Total_Some_college_1_or_more_years_no_degree",
                        ],
                    ),
                ),
                (
                    "Associates Degree",
                    sum_column(education_df, "Estimate_Total_Associate_s_degree"),
                ),
                (
                    "Bachelors Degree",
                    sum_column(education_df, "Estimate_Total_Bachelor_s_degree"),
                ),
                (
                    "Graduate or Professional Degree",
                    sum_matching_columns(
                        education_df,
                        [
                            "Estimate_Total_Master_s_degree",
                            "Estimate_Total_Professional_school_degree",
                            "Estimate_Total_Doctorate_degree",
                        ],
                    ),
                ),
            ]
        summary["education"] = [(label, total) for label, total in education_summary if total != 0]

    income_df = sheet_frames.get("Household Income Brackets")
    if income_df is not None and not income_df.empty:
        income_summary = [
            (
                "Less than $25,000",
                sum_matching_columns(
                    income_df,
                    [
                        "Estimate_Total_Less_than_10_000",
                        "Estimate_Total_10_000_to_14_999",
                        "Estimate_Total_15_000_to_19_999",
                        "Estimate_Total_20_000_to_24_999",
                    ],
                ),
            ),
            (
                "$25,000 to $49,999",
                sum_matching_columns(
                    income_df,
                    [
                        "Estimate_Total_25_000_to_29_999",
                        "Estimate_Total_30_000_to_34_999",
                        "Estimate_Total_35_000_to_39_999",
                        "Estimate_Total_40_000_to_44_999",
                        "Estimate_Total_45_000_to_49_999",
                    ],
                ),
            ),
            (
                "$50,000 to $74,999",
                sum_matching_columns(
                    income_df,
                    [
                        "Estimate_Total_50_000_to_59_999",
                        "Estimate_Total_60_000_to_74_999",
                    ],
                ),
            ),
            (
                "$75,000 to $99,999",
                sum_column(income_df, "Estimate_Total_75_000_to_99_999"),
            ),
            (
                "$100,000 to $149,999",
                sum_matching_columns(
                    income_df,
                    [
                        "Estimate_Total_100_000_to_124_999",
                        "Estimate_Total_125_000_to_149_999",
                    ],
                ),
            ),
            (
                "$150,000 and Over",
                sum_matching_columns(
                    income_df,
                    [
                        "Estimate_Total_150_000_to_199_999",
                        "Estimate_Total_200_000_or_more",
                    ],
                ),
            ),
        ]
        summary["household_income"] = [
            (label, total) for label, total in income_summary if total != 0
        ]

    housing_units_df = sheet_frames.get("Housing Units")
    if housing_units_df is not None and not housing_units_df.empty:
        housing_units_summary = [
            (
                "Housing Units",
                sum_column(housing_units_df, "Estimate_Total"),
            ),
        ]
        summary["housing_units"] = [
            (label, total) for label, total in housing_units_summary if total != 0
        ]

    tenure_df = sheet_frames.get("Tenure by Vehicles Available")
    if tenure_df is not None and not tenure_df.empty:
        tenure_summary = [
            (
                "Owner-occupied",
                sum_column(tenure_df, "Estimate_Total_Owner_occupied"),
            ),
            (
                "Renter-occupied",
                sum_column(tenure_df, "Estimate_Total_Renter_occupied"),
            ),
        ]
        summary["housing_occupancy_tenure"] = [
            (label, total) for label, total in tenure_summary if total != 0
        ]
        vehicle_summary = [
            (
                "No vehicle available",
                sum_matching_columns(
                    tenure_df,
                    [
                        "Estimate_Total_Owner_occupied_No_vehicle_available",
                        "Estimate_Total_Renter_occupied_No_vehicle_available",
                    ],
                ),
            ),
            (
                "1 vehicle available",
                sum_matching_columns(
                    tenure_df,
                    [
                        "Estimate_Total_Owner_occupied_1_vehicle_available",
                        "Estimate_Total_Renter_occupied_1_vehicle_available",
                    ],
                ),
            ),
            (
                "2 vehicles available",
                sum_matching_columns(
                    tenure_df,
                    [
                        "Estimate_Total_Owner_occupied_2_vehicles_available",
                        "Estimate_Total_Renter_occupied_2_vehicles_available",
                    ],
                ),
            ),
            (
                "3 or more vehicles available",
                sum_matching_columns(
                    tenure_df,
                    [
                        "Estimate_Total_Owner_occupied_3_vehicles_available",
                        "Estimate_Total_Renter_occupied_3_vehicles_available",
                        "Estimate_Total_Owner_occupied_4_vehicles_available",
                        "Estimate_Total_Renter_occupied_4_vehicles_available",
                        "Estimate_Total_Owner_occupied_5_or_more_vehicles_available",
                        "Estimate_Total_Renter_occupied_5_or_more_vehicles_available",
                    ],
                ),
            ),
        ]
        summary["vehicle_availability"] = [
            (label, total) for label, total in vehicle_summary if total != 0
        ]
        occupied_total = first_available_total(
            tenure_df,
            ["Estimate_Total", "Total"],
        )
        if occupied_total:
            summary["section_denominators"]["Tenure by Vehicles Available"] = (
                "Total Occupied Housing Units",
                occupied_total,
            )

    transport_df = sheet_frames.get("Means of Transportation to Work")
    if transport_df is not None and not transport_df.empty:
        total_commuters = first_available_total(
            transport_df,
            ["Estimate_Total", "Total"],
        )

        def transport_total(primary: list[str], fallback: list[str] = []) -> float:
            total = sum_available_columns(transport_df, primary)
            if total:
                return total
            if fallback:
                return sum_available_columns(transport_df, fallback)
            return 0.0

        transportation_summary = [
            (
                "Worked from Home",
                transport_total(
                    [
                        "Estimate_Total_Worked_from_home",
                        "Estimate_Total_Worked_at_home",
                    ],
                    [
                        "Estimate_Total_Male_Worked_at_home",
                        "Estimate_Total_Female_Worked_at_home",
                    ],
                ),
            ),
            (
                "Car/truck or van",
                transport_total(
                    [
                        "Estimate_Total_Car_truck_or_van_Drove_alone",
                        "Estimate_Total_Car_truck_or_van_Carpooled",
                        "Estimate_Total_Drove_alone",
                        "Estimate_Total_Carpooled",
                    ],
                    [
                        "Estimate_Total_Male_Drove_alone",
                        "Estimate_Total_Female_Drove_alone",
                        "Estimate_Total_Male_Carpooled",
                        "Estimate_Total_Female_Carpooled",
                    ],
                ),
            ),
            (
                "Public transportation",
                transport_total(
                    [
                        "Estimate_Total_Public_transportation",
                        "Estimate_Total_Public_transportation_excluding_taxicab",
                        "Estimate_Total_Bus_or_trolley_bus",
                        "Estimate_Total_Streetcar_or_trolley_car",
                        "Estimate_Total_Subway_or_elevated",
                        "Estimate_Total_Railroad",
                        "Estimate_Total_Ferryboat",
                    ],
                    [
                        "Estimate_Total_Male_Public_transportation_excluding_taxicab",
                        "Estimate_Total_Female_Public_transportation_excluding_taxicab",
                    ],
                ),
            ),
            (
                "Walked",
                transport_total(
                    [
                        "Estimate_Total_Walked",
                    ],
                    [
                        "Estimate_Total_Male_Walked",
                        "Estimate_Total_Female_Walked",
                        "Walked",
                    ],
                ),
            ),
            (
                "Bicycle",
                transport_total(
                    [
                        "Estimate_Total_Bicycle",
                    ],
                    [
                        "Estimate_Total_Male_Bicycle",
                        "Estimate_Total_Female_Bicycle",
                        "Bicycle",
                    ],
                ),
            ),
            (
                "Motorcycle",
                transport_total(
                    [
                        "Estimate_Total_Motorcycle",
                    ],
                    [
                        "Estimate_Total_Male_Motorcycle",
                        "Estimate_Total_Female_Motorcycle",
                        "Motorcycle",
                    ],
                ),
            ),
            (
                "Taxi or ride hailing services",
                transport_total(
                    [
                        "Estimate_Total_Taxi_or_ride_hailing_services",
                        "Estimate_Total_Taxicab",
                    ],
                    [
                        "Estimate_Total_Male_Taxi_or_ride_hailing_services",
                        "Estimate_Total_Female_Taxi_or_ride_hailing_services",
                    ],
                ),
            ),
            (
                "Other means",
                transport_total(
                    [
                        "Estimate_Total_Other_means",
                        "Estimate_Total_Taxicab_motorcycle_or_other_means",
                    ],
                    [
                        "Estimate_Total_Male_Taxicab_motorcycle_or_other_means",
                        "Estimate_Total_Female_Taxicab_motorcycle_or_other_means",
                        "Other_means",
                    ],
                ),
            ),
        ]

        transportation_values = {label: float(total or 0.0) for label, total in transportation_summary}
        if total_commuters:
            listed_total = sum(transportation_values.values())
            remainder = float(total_commuters) - listed_total
            if abs(remainder) > 0.5:
                if transportation_values.get("Public transportation", 0.0) == 0.0:
                    transit_fallback = sum_available_columns(
                        transport_df,
                        [
                            "Estimate_Total_Public_transportation_excluding_taxicab",
                            "Public_transportation",
                            "Estimate_Total_Public_transportation",
                            "Estimate_Total_Bus_or_trolley_bus",
                            "Estimate_Total_Streetcar_or_trolley_car",
                            "Estimate_Total_Subway_or_elevated",
                            "Estimate_Total_Railroad",
                            "Estimate_Total_Ferryboat",
                        ],
                    )
                    if transit_fallback:
                        transportation_values["Public transportation"] = float(transit_fallback)
                        listed_total = sum(transportation_values.values())
                        remainder = float(total_commuters) - listed_total
                if abs(remainder) > 0.5:
                    transportation_values["Other means"] = (
                        transportation_values.get("Other means", 0.0) + remainder
                    )
            transportation_summary = list(transportation_values.items())
        summary["means_of_transportation"] = [
            (label, total) for label, total in transportation_summary if total != 0
        ]
        if total_commuters:
            summary["section_denominators"]["Means of Transportation to Work"] = (
                "Total Commuters",
                total_commuters,
            )

    employment_df = sheet_frames.get("Employment Status (16+)")
    if employment_df is not None and not employment_df.empty:
        employment_summary = [
            (
                "In labor force",
                sum_column(employment_df, "Estimate_Total_In_labor_force"),
            ),
            (
                "Not in labor force",
                sum_column(employment_df, "Estimate_Total_Not_in_labor_force"),
            ),
        ]
        summary["employment_status"] = [
            (label, total) for label, total in employment_summary if total != 0
        ]

    race_df = sheet_frames.get("Hispanic Latino by Race")
    if race_df is not None and not race_df.empty:
        label_map = label_maps.get("Hispanic Latino by Race", {})
        race_summary = [
            (
                "White (Non-Hispanic)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Not Hispanic or Latino:!!White alone"],
                ),
            ),
            (
                "Hispanic or Latino (of Any Race)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Hispanic or Latino:"],
                ),
            ),
            (
                "Black (Non-Hispanic)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Not Hispanic or Latino:!!Black or African American alone"],
                ),
            ),
            (
                "Asian (Non-Hispanic)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Not Hispanic or Latino:!!Asian alone"],
                ),
            ),
            (
                "American Indian and Alaska Native (Non-Hispanic)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Not Hispanic or Latino:!!American Indian and Alaska Native alone"],
                ),
            ),
            (
                "Native Hawaiian and Other Pacific Islander (Non-Hispanic)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Not Hispanic or Latino:!!Native Hawaiian and Other Pacific Islander alone"],
                ),
            ),
            (
                "Some Other Race (Non-Hispanic)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Not Hispanic or Latino:!!Some other race alone"],
                ),
            ),
            (
                "Two or More Races (Non-Hispanic)",
                sum_columns_by_label(
                    race_df,
                    label_map,
                    exact_labels=["Estimate!!Total:!!Not Hispanic or Latino:!!Two or more races:"],
                ),
            ),
        ]
        summary["hispanic_race"] = [(label, total) for label, total in race_summary if total != 0]

    return summary


def build_acs1_dp05_summary(sheet_frames: dict[str, pd.DataFrame]) -> dict:
    summary = {
        "section_denominators": {},
        "total_population": None,
        "sex": {},
        "age": {},
        "education": [],
        "household_income": [],
        "limited_english": [],
        "housing_units": [],
        "housing_occupancy_tenure": [],
        "vehicle_availability": [],
        "means_of_transportation": [],
        "employment_status": [],
        "travel_time": [],
        "hispanic_race": [],
    }

    # [FIX-9/10/11] Resolve alternate sheet names used by ACS1 place/county
    # trend workbooks so callers do not need to pre-rename their frames.
    _SHEET_ALIASES: dict[str, str] = {
        "Educational Attainment":  "Educational Attainment (25+)",
        "Household Income":        "Household Income Brackets",
        "Vehicle Ownership":       "Tenure by Vehicles Available",
    }
    sheet_frames = dict(sheet_frames)  # shallow copy – do not mutate caller's dict
    for canonical, alias in _SHEET_ALIASES.items():
        if canonical not in sheet_frames and alias in sheet_frames:
            sheet_frames[canonical] = sheet_frames[alias]

    dp05_df = sheet_frames.get("DP05 Demographics")
    # [FIX-4] Do not bail out when DP05 Demographics is absent.  All other
    # sections (Age, Education, Income, Employment, Travel, etc.) are
    # independent of DP05 and should still be populated when their sheets are
    # present.  We only skip DP05-derived fields (total_population, race) when
    # that specific sheet is missing.
    if dp05_df is not None and not dp05_df.empty:
        summary["total_population"] = sum_column(dp05_df, "Total_Population")

    age_df = sheet_frames.get("Age by Sex")
    if age_df is not None and not age_df.empty:
        age_summary = [
            (
                "Under 5",
                sum_matching_columns(
                    age_df,
                    ["Male_Under_5_Years", "Female_Under_5_Years"],
                ),
            ),
            (
                "5 to 19",
                sum_matching_columns(
                    age_df,
                    [
                        "Male_5_to_9_Years",
                        "Female_5_to_9_Years",
                        "Male_10_to_14_Years",
                        "Female_10_to_14_Years",
                        "Male_15_to_17_Years",
                        "Female_15_to_17_Years",
                        "Male_18_and_19_Years",
                        "Female_18_and_19_Years",
                    ],
                ),
            ),
            (
                "20 to 34",
                sum_matching_columns(
                    age_df,
                    [
                        "Male_20_Years",
                        "Female_20_Years",
                        "Male_21_Years",
                        "Female_21_Years",
                        "Male_22_to_24_Years",
                        "Female_22_to_24_Years",
                        "Male_25_to_29_Years",
                        "Female_25_to_29_Years",
                        "Male_30_to_34_Years",
                        "Female_30_to_34_Years",
                    ],
                ),
            ),
            (
                "35 to 49",
                sum_matching_columns(
                    age_df,
                    [
                        "Male_35_to_39_Years",
                        "Female_35_to_39_Years",
                        "Male_40_to_44_Years",
                        "Female_40_to_44_Years",
                        "Male_45_to_49_Years",
                        "Female_45_to_49_Years",
                    ],
                ),
            ),
            (
                "50 to 64",
                sum_matching_columns(
                    age_df,
                    [
                        "Male_50_to_54_Years",
                        "Female_50_to_54_Years",
                        "Male_55_to_59_Years",
                        "Female_55_to_59_Years",
                        "Male_60_and_61_Years",
                        "Female_60_and_61_Years",
                        "Male_62_to_64_Years",
                        "Female_62_to_64_Years",
                    ],
                ),
            ),
            (
                "65 to 74",
                sum_matching_columns(
                    age_df,
                    [
                        "Male_65_and_66_Years",
                        "Female_65_and_66_Years",
                        "Male_67_to_69_Years",
                        "Female_67_to_69_Years",
                        "Male_70_to_74_Years",
                        "Female_70_to_74_Years",
                    ],
                ),
            ),
            (
                "75 to 84",
                sum_matching_columns(
                    age_df,
                    [
                        "Male_75_to_79_Years",
                        "Female_75_to_79_Years",
                        "Male_80_to_84_Years",
                        "Female_80_to_84_Years",
                    ],
                ),
            ),
            (
                "85 and Over",
                sum_matching_columns(
                    age_df,
                    ["Male_85_Years_and_Over", "Female_85_Years_and_Over"],
                ),
            ),
        ]
        summary["age"] = {label: value for label, value in age_summary if value}

    education_df = sheet_frames.get("Educational Attainment")
    if education_df is not None and not education_df.empty:
        education_summary = [
            (
                "Less than High School Diploma",
                sum_matching_columns(
                    education_df,
                    [
                        "Estimate_Total_No_schooling_completed",
                        "Estimate_Total_Nursery_school",
                        "Estimate_Total_Kindergarten",
                        "Estimate_Total_1st_grade",
                        "Estimate_Total_2nd_grade",
                        "Estimate_Total_3rd_grade",
                        "Estimate_Total_4th_grade",
                        "Estimate_Total_5th_grade",
                        "Estimate_Total_6th_grade",
                        "Estimate_Total_7th_grade",
                        "Estimate_Total_8th_grade",
                        "Estimate_Total_9th_grade",
                        "Estimate_Total_10th_grade",
                        "Estimate_Total_11th_grade",
                        "Estimate_Total_12th_grade_no_diploma",
                    ],
                ),
            ),
            (
                "High School Diploma or Equivalent",
                sum_matching_columns(
                    education_df,
                    [
                        "Estimate_Total_High_school_graduate_includes_equivalency",
                        "Estimate_Total_Regular_high_school_diploma",
                        "Estimate_Total_GED_or_alternative_credential",
                    ],
                ),
            ),
            (
                "Some College No Degree",
                sum_matching_columns(
                    education_df,
                    [
                        "Estimate_Total_Some_college_less_than_1_year",
                        "Estimate_Total_Some_college_1_or_more_years_no_degree",
                    ],
                ),
            ),
            (
                "Associates Degree",
                sum_column(education_df, "Estimate_Total_Associate_s_degree"),
            ),
            (
                "Bachelors Degree",
                sum_column(education_df, "Estimate_Total_Bachelor_s_degree"),
            ),
            (
                "Graduate or Professional Degree",
                sum_matching_columns(
                    education_df,
                    [
                        "Estimate_Total_Master_s_degree",
                        "Estimate_Total_Professional_school_degree",
                        "Estimate_Total_Doctorate_degree",
                    ],
                ),
            ),
        ]
        summary["education"] = [(label, total) for label, total in education_summary if total != 0]

    income_df = sheet_frames.get("Household Income")
    if income_df is not None and not income_df.empty:
        # [FIX-10] Place/county trend workbooks store income columns with an
        # "Estimate_Total_" prefix (e.g. "Estimate_Total_Less_than_10_000").
        # The original code used bare names ("Less_than_10_000").  Both forms
        # are listed so sum_matching_columns finds whichever is present.
        income_summary = [
            (
                "Less than $25,000",
                sum_matching_columns(
                    income_df,
                    [
                        "Less_than_10_000",
                        "Estimate_Total_Less_than_10_000",
                        "10_000_to_14_999",
                        "Estimate_Total_10_000_to_14_999",
                        "15_000_to_19_999",
                        "Estimate_Total_15_000_to_19_999",
                        "20_000_to_24_999",
                        "Estimate_Total_20_000_to_24_999",
                    ],
                ),
            ),
            (
                "$25,000 to $49,999",
                sum_matching_columns(
                    income_df,
                    [
                        "25_000_to_29_999",
                        "Estimate_Total_25_000_to_29_999",
                        "30_000_to_34_999",
                        "Estimate_Total_30_000_to_34_999",
                        "35_000_to_39_999",
                        "Estimate_Total_35_000_to_39_999",
                        "40_000_to_44_999",
                        "Estimate_Total_40_000_to_44_999",
                        "45_000_to_49_999",
                        "Estimate_Total_45_000_to_49_999",
                    ],
                ),
            ),
            (
                "$50,000 to $74,999",
                sum_matching_columns(
                    income_df,
                    [
                        "50_000_to_59_999",
                        "Estimate_Total_50_000_to_59_999",
                        "60_000_to_74_999",
                        "Estimate_Total_60_000_to_74_999",
                    ],
                ),
            ),
            (
                "$75,000 to $99,999",
                sum_matching_columns(
                    income_df,
                    ["75_000_to_99_999", "Estimate_Total_75_000_to_99_999"],
                ),
            ),
            (
                "$100,000 to $149,999",
                sum_matching_columns(
                    income_df,
                    [
                        "100_000_to_124_999",
                        "Estimate_Total_100_000_to_124_999",
                        "125_000_to_149_999",
                        "Estimate_Total_125_000_to_149_999",
                    ],
                ),
            ),
            (
                "$150,000 and Over",
                sum_matching_columns(
                    income_df,
                    [
                        "150_000_to_199_999",
                        "Estimate_Total_150_000_to_199_999",
                        "200_000_or_more",
                        "Estimate_Total_200_000_or_more",
                    ],
                ),
            ),
        ]
        summary["household_income"] = [
            (label, total) for label, total in income_summary if total != 0
        ]

    limited_english_df = sheet_frames.get("Limited English Speaking")
    if limited_english_df is not None and not limited_english_df.empty:
        limited_english_summary = [
            (
                "Limited English-speaking households",
                sum_matching_columns(
                    limited_english_df,
                    [
                        "Limited_English_speaking_households",
                        "Speak_English_less_than_very_well",
                    ],
                ),
            ),
            (
                "Limited English-speaking Spanish households",
                sum_matching_columns(
                    limited_english_df,
                    [
                        "Limited_English_speaking_Spanish_households",
                        "Speak_Spanish_Speak_English_less_than_very_well",
                    ],
                ),
            ),
            (
                "Limited English-speaking Other Indo-European households",
                sum_matching_columns(
                    limited_english_df,
                    [
                        "Limited_English_speaking_Other_Indo_European_households",
                        "Speak_other_Indo_European_languages_Speak_English_less_than_very_well",
                    ],
                ),
            ),
            (
                "Limited English-speaking Asian and Pacific Island households",
                sum_matching_columns(
                    limited_english_df,
                    [
                        "Limited_English_speaking_Asian_and_Pacific_Island_households",
                        "Speak_Asian_and_Pacific_Island_languages_Speak_English_less_than_very_well",
                    ],
                ),
            ),
            (
                "Limited English-speaking Other language households",
                sum_matching_columns(
                    limited_english_df,
                    [
                        "Limited_English_speaking_Other_language_households",
                        "Speak_other_languages_Speak_English_less_than_very_well",
                    ],
                ),
            ),
        ]
        summary["limited_english"] = [
            (label, total) for label, total in limited_english_summary if total != 0
        ]

    vehicle_df = sheet_frames.get("Vehicle Ownership")
    if vehicle_df is not None and not vehicle_df.empty:
        # [FIX-11] Place/county trend workbooks store vehicle columns with an
        # "Estimate_Total_" prefix.  Both bare and prefixed forms are listed.
        vehicle_summary = [
            (
                "No vehicle available",
                sum_matching_columns(
                    vehicle_df,
                    [
                        "Owner_occupied_no_vehicle_available",
                        "Estimate_Total_Owner_occupied_No_vehicle_available",
                        "Renter_occupied_no_vehicle_available",
                        "Estimate_Total_Renter_occupied_No_vehicle_available",
                    ],
                ),
            ),
            (
                "1 vehicle available",
                sum_matching_columns(
                    vehicle_df,
                    [
                        "Owner_occupied_1_vehicle_available",
                        "Estimate_Total_Owner_occupied_1_vehicle_available",
                        "Renter_occupied_1_vehicle_available",
                        "Estimate_Total_Renter_occupied_1_vehicle_available",
                    ],
                ),
            ),
            (
                "2 vehicles available",
                sum_matching_columns(
                    vehicle_df,
                    [
                        "Owner_occupied_2_vehicles_available",
                        "Estimate_Total_Owner_occupied_2_vehicles_available",
                        "Renter_occupied_2_vehicles_available",
                        "Estimate_Total_Renter_occupied_2_vehicles_available",
                    ],
                ),
            ),
            (
                "3 or more vehicles available",
                sum_matching_columns(
                    vehicle_df,
                    [
                        "Owner_occupied_3_vehicles_available",
                        "Estimate_Total_Owner_occupied_3_vehicles_available",
                        "Renter_occupied_3_vehicles_available",
                        "Estimate_Total_Renter_occupied_3_vehicles_available",
                        "Owner_occupied_4_vehicles_available",
                        "Estimate_Total_Owner_occupied_4_vehicles_available",
                        "Renter_occupied_4_vehicles_available",
                        "Estimate_Total_Renter_occupied_4_vehicles_available",
                        "Owner_occupied_5_or_more_vehicles_available",
                        "Estimate_Total_Owner_occupied_5_or_more_vehicles_available",
                        "Renter_occupied_5_or_more_vehicles_available",
                        "Estimate_Total_Renter_occupied_5_or_more_vehicles_available",
                    ],
                ),
            ),
        ]
        summary["vehicle_availability"] = [
            (label, total) for label, total in vehicle_summary if total != 0
        ]

    employment_df = sheet_frames.get("Employment Status (16+)")
    if employment_df is not None and not employment_df.empty:
        # [FIX-3] ACS1 place/county trend workbooks store these columns as
        # "Estimate_Total_In_labor_force" (lowercase) while this builder
        # expects title-cased names ("In_Labor_Force", etc.).  Resolve both
        # forms so either source works without pre-processing.
        _EMP_ALIASES: dict[str, list[str]] = {
            "In_Labor_Force":     ["In_Labor_Force",    "Estimate_Total_In_labor_force"],
            "Not_in_Labor_Force": ["Not_in_Labor_Force", "Estimate_Total_Not_in_labor_force"],
            "Employed":           ["Employed",           "Estimate_Total_In_labor_force_Civilian_labor_force_Employed"],
            "Unemployed":         ["Unemployed",         "Estimate_Total_In_labor_force_Civilian_labor_force_Unemployed"],
        }
        def _emp_col(df: pd.DataFrame, candidates: list[str]) -> float:
            for c in candidates:
                if c in df.columns:
                    return sum_column(df, c)
            return 0.0
        employment_summary = [
            ("In labor force",     _emp_col(employment_df, _EMP_ALIASES["In_Labor_Force"])),
            ("Not in labor force", _emp_col(employment_df, _EMP_ALIASES["Not_in_Labor_Force"])),
            ("Employed",           _emp_col(employment_df, _EMP_ALIASES["Employed"])),
            ("Unemployed",         _emp_col(employment_df, _EMP_ALIASES["Unemployed"])),
        ]
        summary["employment_status"] = [
            (label, total) for label, total in employment_summary if total != 0
        ]

    travel_df = sheet_frames.get("Travel Time to Work")
    if travel_df is not None and not travel_df.empty:
        total_commuters = first_available_total(
            travel_df,
            ["Estimate_Total", "Total"],
        )
        travel_summary = [
            ("Less than 5 minutes", sum_column(travel_df, "Less_than_5_minutes")),
            ("5 to 9 minutes", sum_column(travel_df, "5_to_9_minutes")),
            ("10 to 14 minutes", sum_column(travel_df, "10_to_14_minutes")),
            ("15 to 19 minutes", sum_column(travel_df, "15_to_19_minutes")),
            ("20 to 24 minutes", sum_column(travel_df, "20_to_24_minutes")),
            ("25 to 29 minutes", sum_column(travel_df, "25_to_29_minutes")),
            ("30 to 34 minutes", sum_column(travel_df, "30_to_34_minutes")),
            ("35 to 39 minutes", sum_column(travel_df, "35_to_39_minutes")),
            ("40 to 44 minutes", sum_column(travel_df, "40_to_44_minutes")),
            ("45 to 59 minutes", sum_column(travel_df, "45_to_59_minutes")),
            ("60 to 89 minutes", sum_column(travel_df, "60_to_89_minutes")),
            ("90 or more minutes", sum_column(travel_df, "90_or_more_minutes")),
        ]
        summary["travel_time"] = [
            (label, total) for label, total in travel_summary if total != 0
        ]
        if total_commuters:
            summary["section_denominators"]["Travel Time to Work"] = (
                "Total Commuters",
                total_commuters,
            )

    # [FIX-4] These fields are sourced exclusively from DP05 Demographics;
    # skip them when that sheet is absent rather than crashing.
    if dp05_df is not None and not dp05_df.empty:
        housing_units = sum_column(dp05_df, "Housing_Units")
        if housing_units:
            summary["housing_units"] = [("Housing Units", housing_units)]

        race_summary = [
            ("White (Non-Hispanic)", sum_column(dp05_df, "White_Not_H_L")),
            ("Hispanic or Latino (of Any Race)", sum_column(dp05_df, "Hispanic_or_Latino")),
            ("Black (Non-Hispanic)", sum_column(dp05_df, "Black_or_African_American_Not_H_L")),
            (
                "American Indian and Alaska Native (Non-Hispanic)",
                sum_column(dp05_df, "American_Indian_and_Alaska_Native_Not_H_L"),
            ),
            ("Asian (Non-Hispanic)", sum_column(dp05_df, "Asian_Not_H_L")),
            (
                "Native Hawaiian and Other Pacific Islander (Non-Hispanic)",
                sum_column(dp05_df, "Native_Hawaiian_and_Other_Pacific_Islander_Not_H_L"),
            ),
            ("Some Other Race (Non-Hispanic)", sum_column(dp05_df, "Some_other_race_Not_H_L")),
            ("Two or More Races (Non-Hispanic)", sum_column(dp05_df, "Two_or_More_Races_Not_H_L")),
        ]
        summary["hispanic_race"] = [(label, value) for label, value in race_summary if value]

    return summary


def summary_sections(summary: dict) -> list[tuple[str, list[tuple[str, float]]]]:
    sections = []

    pop_data = []
    if summary.get("total_population") is not None:
        pop_data.append(("Total population", summary["total_population"]))
    for key in ["Male", "Female"]:
        if key in summary.get("sex", {}):
            pop_data.append((key, summary["sex"][key]))
    sections.append(("Population and Sex", pop_data))

    age_data = [(k, v) for k, v in summary.get("age", {}).items() if v]
    sections.append(("Age Distribution", age_data))
    sections.append(("Educational Attainment", summary.get("education", [])))
    sections.append(("Household Income", summary.get("household_income", [])))
    sections.append(("Limited English Speaking", summary.get("limited_english", [])))
    sections.append(("Housing Units", summary.get("housing_units", [])))
    sections.append(("Housing Occupancy and Tenure", summary.get("housing_occupancy_tenure", [])))
    sections.append(("Tenure by Vehicles Available", summary.get("vehicle_availability", [])))
    sections.append(("Means of Transportation to Work", summary.get("means_of_transportation", [])))
    sections.append(("Employment Status", summary.get("employment_status", [])))
    sections.append(("Travel Time to Work", summary.get("travel_time", [])))
    sections.append(("Hispanic or Latino by Race", summary.get("hispanic_race", [])))
    return [(title, data) for title, data in sections if data]


def get_section_denominator_info(
    summary: dict,
    section: str,
    section_values: Optional[dict[str, float]] = None,
) -> tuple[str, Optional[float]]:
    section_denominators = summary.get("section_denominators", {}) or {}
    denominator_info = section_denominators.get(section)
    if denominator_info:
        return denominator_info[0], denominator_info[1]
    if section in {"Age Distribution", "Hispanic or Latino by Race"}:
        return "Total population", summary.get("total_population")
    if section == "Housing Units":
        return "Total", None

    if section_values:
        for total_key in (
            "Total population",
            "Total Households",
            "Total Occupied Housing Units",
            "Total Commuters",
            "Total",
            "Total civilian population",
        ):
            if total_key in section_values:
                return total_key, section_values.get(total_key)

        numeric_values = []
        for value in section_values.values():
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value >= 0:
                numeric_values.append(numeric_value)
        if len(numeric_values) > 1:
            total = sum(numeric_values)
            if total > 0:
                return "Total", total

    return "Total", None


def write_summary_sheet(writer: pd.ExcelWriter, summary: dict, sheet_name: str = "Executive Summary") -> None:
    workbook = writer.book
    summary_ws = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = summary_ws
    header_format = workbook.add_format({"bold": True})
    number_format = workbook.add_format({"num_format": "#,##0"})
    percent_format = workbook.add_format({"num_format": "0.0%"})
    total_format = workbook.add_format({"bold": True, "num_format": "#,##0"})
    total_percent_format = workbook.add_format({"bold": True, "num_format": "0.0%"})

    row = 0
    summary_ws.write(row, 0, sheet_name, header_format)
    row += 2

    for title, data in summary_sections(summary):
        section_gap = 3
        summary_ws.write(row, 0, title, header_format)
        row += 1
        df = pd.DataFrame(data, columns=["Category", "Value"])
        denominator_label, denominator_value = get_section_denominator_info(
            summary,
            title,
            {category: value for category, value in data},
        )
        if denominator_value in (None, 0):
            denominator_value = float(df["Value"].sum())
            denominator_label = "Total"
        if denominator_value:
            df["Percent"] = df["Value"] / float(denominator_value)
        else:
            df["Percent"] = 0.0
        existing_total_mask = (
            df["Category"].astype(str).str.strip().str.casefold() == str(denominator_label).strip().casefold()
        )
        if not existing_total_mask.any():
            total_row = pd.DataFrame(
                [{
                    "Category": denominator_label,
                    "Value": denominator_value,
                    "Percent": 1.0 if denominator_value else 0.0,
                }]
            )
            df = pd.concat([df, total_row], ignore_index=True)
            total_row_added = True
        else:
            total_row_added = False
            # move any existing total rows to the bottom to keep chart ranges clean
            total_rows = df[existing_total_mask]
            df = pd.concat([df[~existing_total_mask], total_rows], ignore_index=True)

        df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=0, index=False)
        summary_ws.set_column(0, 0, 40)
        summary_ws.set_column(1, 1, 16, number_format)
        summary_ws.set_column(2, 2, 12, percent_format)

        start_row = row
        end_row = row + len(df) - 1
        if len(df) > 1:
            chart_end_row = end_row - 1 if total_row_added or existing_total_mask.any() else end_row
            if chart_end_row >= start_row:
                chart = workbook.add_chart({"type": "column"})
                chart.add_series(
                    {
                        "name": title,
                        "categories": [sheet_name, start_row + 1, 0, chart_end_row, 0],
                        "values": [sheet_name, start_row + 1, 2, chart_end_row, 2],
                    }
                )
                chart.set_title({"name": title})
                chart.set_legend({"none": True})
                summary_ws.insert_chart(start_row, 4, chart, {"x_scale": 1.1, "y_scale": 1.1})

        row += len(df) + section_gap


def write_peer_comparison_sheet(writer: pd.ExcelWriter, peer_summaries: dict[str, dict]) -> None:
    if len(peer_summaries) <= 1:
        return

    sheet_name = "Peer Comparison"
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    header_format = workbook.add_format({"bold": True})
    number_format = workbook.add_format({"num_format": "#,##0"})
    decimal_format = workbook.add_format({"num_format": "#,##0.0"})
    percent_format = workbook.add_format({"num_format": "0.0%"})

    worksheet.write(0, 0, sheet_name, header_format)
    worksheet.set_column(0, 0, 40)

    geography_names = list(peer_summaries.keys())
    for col_idx in range(1, len(geography_names) + 1):
        worksheet.set_column(col_idx, col_idx, 18, number_format)

    section_order = []
    category_order: dict[str, list[str]] = {}
    summary_lookup: dict[str, dict[str, dict[str, float]]] = {}

    for geography_name, summary in peer_summaries.items():
        section_map: dict[str, dict[str, float]] = {}
        for section, data in summary_sections(summary):
            if section not in section_order:
                section_order.append(section)
            category_order.setdefault(section, [])
            values = section_map.setdefault(section, {})
            for category, value in data:
                values[category] = value
                if category not in category_order[section]:
                    category_order[section].append(category)
        summary_lookup[geography_name] = section_map

    row = 2
    for section in section_order:
        categories = category_order.get(section, [])
        if not categories:
            continue

        worksheet.write(row, 0, section, header_format)
        row += 1

        df = pd.DataFrame({"Category": categories})
        for geography_name in geography_names:
            df[geography_name] = [
                summary_lookup.get(geography_name, {}).get(section, {}).get(category)
                for category in categories
            ]

        df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=0, index=False)
        start_row = row
        end_row = row + len(df)

        for data_row in range(len(df)):
            excel_row = start_row + 1 + data_row
            for col_idx, column_name in enumerate(df.columns[1:], start=1):
                value = df.iloc[data_row, col_idx]
                if pd.isna(value):
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                cell_format = number_format if float(numeric_value).is_integer() else decimal_format
                worksheet.write_number(excel_row, col_idx, numeric_value, cell_format)

        row = end_row + 2

        excluded_percent_categories = {
            "Total population",
            "Total Households",
            "Total Occupied Housing Units",
            "Total Commuters",
            "Total",
            "Housing Units",
        }
        percent_categories = [
            category for category in categories if section != "Housing Units" and category not in excluded_percent_categories
        ]
        percent_df = pd.DataFrame({"Category": percent_categories})
        if percent_categories:
            for geography_name in geography_names:
                section_values = summary_lookup.get(geography_name, {}).get(section, {})
                denominator = get_trend_section_denominator(
                    peer_summaries.get(geography_name, {}),
                    section,
                    section_values,
                )
                percent_df[geography_name] = [
                    None
                    if denominator in (None, 0) or section_values.get(category) is None
                    else float(section_values.get(category)) / float(denominator)
                    for category in percent_categories
                ]

        chart_start_row = start_row
        chart_end_row = end_row
        chart_title = section

        has_percent_values = False
        if not percent_df.empty:
            for geography_name in geography_names:
                if geography_name not in percent_df.columns:
                    continue
                series = pd.to_numeric(percent_df[geography_name], errors="coerce")
                if series.notna().any():
                    has_percent_values = True
                    break

        if has_percent_values:
            worksheet.write(row, 0, f"{section} % Value", header_format)
            row += 1
            percent_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=0, index=False)
            percent_start_row = row
            percent_end_row = row + len(percent_df)

            for data_row in range(len(percent_df)):
                excel_row = percent_start_row + 1 + data_row
                for col_idx in range(1, len(percent_df.columns)):
                    value = percent_df.iloc[data_row, col_idx]
                    if pd.isna(value):
                        continue
                    try:
                        worksheet.write_number(excel_row, col_idx, float(value), percent_format)
                    except (TypeError, ValueError):
                        continue

            chart_start_row = percent_start_row
            chart_end_row = percent_end_row
            chart_title = f"{section} (%)"
            row = percent_end_row + 2

        chart = workbook.add_chart({"type": "column"})
        for col_idx, geography_name in enumerate(geography_names, start=1):
            chart.add_series(
                {
                    "name": [sheet_name, chart_start_row, col_idx],
                    "categories": [sheet_name, chart_start_row + 1, 0, chart_end_row, 0],
                    "values": [sheet_name, chart_start_row + 1, col_idx, chart_end_row, col_idx],
                }
            )
        chart.set_title({"name": chart_title})
        chart.set_legend({"position": "left"})
        if chart_title.endswith("(%)"):
            chart.set_y_axis({"num_format": "0%"})
        worksheet.insert_chart(chart_start_row, len(geography_names) + 2, chart, {"x_scale": 1.3, "y_scale": 1.15})

        row = max(row + 2, chart_start_row + 20)


def calculate_percent_change(start_value: Optional[float], end_value: Optional[float]) -> Optional[float]:
    if start_value in (None, 0) or end_value is None:
        return None
    return (float(end_value) - float(start_value)) / float(start_value)


def calculate_cagr(start_value: Optional[float], end_value: Optional[float], periods: int) -> Optional[float]:
    if start_value in (None, 0) or end_value is None or periods <= 0:
        return None
    start = float(start_value)
    end = float(end_value)
    if start <= 0 or end < 0:
        return None
    return (end / start) ** (1 / periods) - 1


def get_trend_section_denominator(
    summary: dict,
    section: str,
    section_values: dict[str, float],
) -> Optional[float]:
    _, denominator = get_section_denominator_info(summary, section, section_values)
    return denominator


def get_percentage_categories(
    section: str,
    categories: list[str],
    section_values: dict[str, float],
) -> list[str]:
    if section == "Housing Units":
        return []
    excluded = {
        "Total population",
        "Total Households",
        "Total Occupied Housing Units",
        "Total Commuters",
        "Total",
        "Housing Units",
    }
    return [category for category in categories if category in section_values and category not in excluded]


def add_trend_row_total(section: str, entry: dict, categories: list[str]) -> None:
    if section != "Age Distribution":
        return

    total = 0.0
    has_value = False
    for category in categories:
        value = entry.get(category)
        if value is None or pd.isna(value):
            continue
        try:
            total += float(value)
            has_value = True
        except (TypeError, ValueError):
            continue

    if has_value:
        entry["Total"] = total


def write_trend_sheet(writer: pd.ExcelWriter, yearly_summaries: dict[int, dict], sheet_name: str = "Trend Summary") -> None:
    if len(yearly_summaries) <= 1:
        return

    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    header_format = workbook.add_format({"bold": True})
    number_format = workbook.add_format({"num_format": "#,##0"})
    decimal_format = workbook.add_format({"num_format": "#,##0.0"})
    percent_format = workbook.add_format({"num_format": "0.0%"})

    def write_formatted_block(
        df: pd.DataFrame,
        start_row: int,
        percent_cols: Optional[set[str]] = None,
        text_cols: Optional[set[str]] = None,
    ) -> None:
        percent_cols = percent_cols or set()
        text_cols = text_cols or set()
        for row_offset in range(len(df)):
            excel_row = start_row + 1 + row_offset
            for col_idx, col_name in enumerate(df.columns):
                if col_name in text_cols:
                    continue
                value = df.iloc[row_offset, col_idx]
                if pd.isna(value):
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if col_name in percent_cols:
                    cell_format = percent_format
                elif float(numeric_value).is_integer():
                    cell_format = number_format
                else:
                    cell_format = decimal_format
                worksheet.write_number(excel_row, col_idx, numeric_value, cell_format)

    years = sorted(yearly_summaries)
    section_order = []
    category_order: dict[str, list[str]] = {}
    summary_lookup: dict[int, dict[str, dict[str, float]]] = {}

    for year in years:
        section_map: dict[str, dict[str, float]] = {}
        for section, data in summary_sections(yearly_summaries[year]):
            if section not in section_order:
                section_order.append(section)
            category_order.setdefault(section, [])
            values = section_map.setdefault(section, {})
            for category, value in data:
                values[category] = value
                if category not in category_order[section]:
                    category_order[section].append(category)
        summary_lookup[year] = section_map

    worksheet.write(0, 0, sheet_name, header_format)
    worksheet.set_column(0, 0, 20)
    row = 2

    for section in section_order:
        categories = category_order.get(section, [])
        if not categories:
            continue
        section_years = [
            year
            for year in years
            if summary_lookup.get(year, {}).get(section)
        ]
        if not section_years:
            continue

        worksheet.write(row, 0, section, header_format)
        row += 1

        trend_rows = []
        for year in section_years:
            entry = {"Year": year}
            for category in categories:
                entry[category] = summary_lookup.get(year, {}).get(section, {}).get(category)
            add_trend_row_total(section, entry, categories)
            trend_rows.append(entry)
        trend_df = pd.DataFrame(trend_rows)
        trend_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=0, index=False)
        trend_header_row = row
        trend_first_data_row = row + 1
        trend_last_data_row = row + len(trend_df)
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, max(1, len(trend_df.columns) - 1), 16)
        write_formatted_block(trend_df, trend_header_row, text_cols={"Year"})

        chart = workbook.add_chart({"type": "line"})
        for col_idx, category in enumerate(trend_df.columns[1:], start=1):
            chart.add_series(
                {
                    "name": [sheet_name, trend_header_row, col_idx],
                    "categories": [sheet_name, trend_first_data_row, 0, trend_last_data_row, 0],
                    "values": [sheet_name, trend_first_data_row, col_idx, trend_last_data_row, col_idx],
                }
            )
        chart.set_title({"name": section})
        chart.set_legend({"position": "left"})
        worksheet.insert_chart(trend_header_row, len(trend_df.columns) + 2, chart, {"x_scale": 1.35, "y_scale": 1.15})

        row = max(trend_last_data_row + 3, trend_header_row + 18)

        percent_categories = get_percentage_categories(
            section,
            categories,
            summary_lookup.get(section_years[0], {}).get(section, {}),
        )
        percent_rows = []
        for year in section_years:
            section_values = summary_lookup.get(year, {}).get(section, {})
            denominator = get_trend_section_denominator(yearly_summaries[year], section, section_values)
            percent_entry = {"Year": year}
            for category in percent_categories:
                value = section_values.get(category)
                if denominator in (None, 0) or value is None:
                    percent_entry[category] = None
                else:
                    percent_entry[category] = float(value) / float(denominator)
            if len(percent_entry) > 1:
                percent_rows.append(percent_entry)

        if percent_rows and percent_categories:
            worksheet.write(row, 0, f"{section} % Value", header_format)
            row += 1
            percent_df = pd.DataFrame(percent_rows)
            percent_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=0, index=False)
            percent_header_row = row
            percent_first_data_row = row + 1
            percent_last_data_row = row + len(percent_df)
            worksheet.set_column(0, 0, 20)
            worksheet.set_column(1, max(1, len(percent_df.columns) - 1), 16)
            write_formatted_block(
                percent_df,
                percent_header_row,
                percent_cols=set(percent_df.columns) - {"Year"},
                text_cols={"Year"},
            )

            percent_chart = workbook.add_chart({"type": "line"})
            for col_idx, category in enumerate(percent_df.columns[1:], start=1):
                percent_chart.add_series(
                    {
                        "name": [sheet_name, percent_header_row, col_idx],
                        "categories": [sheet_name, percent_first_data_row, 0, percent_last_data_row, 0],
                        "values": [sheet_name, percent_first_data_row, col_idx, percent_last_data_row, col_idx],
                    }
                )
            percent_chart.set_title({"name": f"{section} % Value"})
            percent_chart.set_legend({"position": "left"})
            percent_chart.set_y_axis({"num_format": "0%"})
            worksheet.insert_chart(
                percent_header_row,
                len(percent_df.columns) + 2,
                percent_chart,
                {"x_scale": 1.35, "y_scale": 1.15},
            )

            row = max(percent_last_data_row + 3, percent_header_row + 18)

        metric_rows = []
        start_year = section_years[0]
        end_year = section_years[-1]
        periods = end_year - start_year
        for category in categories:
            start_value = summary_lookup.get(start_year, {}).get(section, {}).get(category)
            end_value = summary_lookup.get(end_year, {}).get(section, {}).get(category)
            change = None if start_value is None or end_value is None else float(end_value) - float(start_value)
            metric_rows.append(
                {
                    "Category": category,
                    f"{start_year}": start_value,
                    f"{end_year}": end_value,
                    "Absolute Change": change,
                    "Percent Change": calculate_percent_change(start_value, end_value),
                    "CAGR": calculate_cagr(start_value, end_value, periods),
                }
            )
        metrics_df = pd.DataFrame(metric_rows)
        metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=0, index=False)
        worksheet.set_column(0, 0, 40)
        worksheet.set_column(1, max(1, len(metrics_df.columns) - 1), 16)
        if "Percent Change" in metrics_df.columns:
            write_formatted_block(
                metrics_df,
                row,
                percent_cols={"Percent Change", "CAGR"},
                text_cols={"Category"},
            )
        else:
            write_formatted_block(metrics_df, row, text_cols={"Category"})

        row += len(metrics_df) + 4


def dedupe_locations(
    primary: str,
    primary_state: str,
    comparisons: Optional[list[dict]] = None,
) -> list[dict]:
    ordered = []
    seen = set()
    candidates = [{"location": primary, "state": primary_state}] + list(comparisons or [])
    for item in candidates:
        clean_name = str(item.get("location", "")).strip()
        clean_state = str(item.get("state", "")).strip()
        if not clean_name or not clean_state:
            continue
        key = (clean_name.casefold(), clean_state.casefold())
        if key in seen:
            continue
        seen.add(key)
        ordered.append({"location": clean_name, "state": clean_state})
    return ordered


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def census_get(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = 60,
    retries: int = 3,
    logger: Optional[Callable[[str], None]] = None,
    request_label: str = "request",
):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return requests.get(url, params=params, timeout=timeout)
        except requests.exceptions.ReadTimeout as exc:
            last_exc = exc
            if logger:
                logger(f"WARN: {request_label} timed out (attempt {attempt}/{retries}).")
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if logger:
                logger(f"WARN: {request_label} failed (attempt {attempt}/{retries}): {exc}")
        if attempt < retries:
            time.sleep(min(2 * attempt, 5))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed request for {request_label}")


def get_group_variables_and_labels(
    year: int,
    group_id: str,
    profile: bool = False,
    survey: str = "acs5",
    dataset_path: str = "",
    api_key: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> tuple[list, dict]:
    if survey == "acs5" and not profile and not dataset_path:
        year_range = ACS5_TABLE_YEAR_RANGES.get(group_id)
        if year_range:
            year_min, year_max = year_range
            if year < year_min or year > year_max or year in ACS5_TABLE_YEAR_EXCLUSIONS.get(group_id, set()):
                return [], {}
    api_key = api_key or os.getenv("CENSUS_API_KEY")
    params = {"key": api_key} if api_key else None
    if dataset_path:
        url = f"https://api.census.gov/data/{year}/acs/{survey}/{dataset_path}/groups/{group_id}.json"
    elif profile:
        url = f"https://api.census.gov/data/{year}/acs/{survey}/profile/groups/{group_id}.json"
    else:
        url = f"https://api.census.gov/data/{year}/acs/{survey}/groups/{group_id}.json"
    resp = census_get(
        url,
        params=params,
        timeout=60,
        retries=3,
        request_label=f"group metadata {group_id} {survey.upper()} {year}",
    )
    if resp.status_code == 404:
        return [], {}
    resp.raise_for_status()
    try:
        group_info = resp.json()
    except ValueError:
        if logger:
            logger(
                f"ERROR: invalid JSON response for group metadata {group_id} {survey.upper()} {year}: "
                f"status={resp.status_code}, content-type={resp.headers.get('Content-Type')}, "
                f"body={resp.text[:300]!r}"
            )
        return [], {}
    variables = [
        v
        for v in group_info["variables"].keys()
        if v.endswith("E") and not v.endswith("PE")
    ]
    labels = {
        var_name: group_info["variables"][var_name]["label"]
        for var_name in variables
    }
    if group_id == "B16002":
        available_set = set(variables)
        selected_variables = [
            variable_id
            for variable_id in LIMITED_ENGLISH_TABLE_VARS
            if variable_id in available_set
        ]
        selected_labels = {
            variable_id: LIMITED_ENGLISH_TABLE_VARS[variable_id]
            for variable_id in selected_variables
        }
        return selected_variables, selected_labels
    if survey == "acs5" and group_id == "B08006":
        available_set = set(variables)
        selected_variables = [
            variable_id
            for variable_id in ACS1_B08006_SELECTED_VARS
            if variable_id in available_set
        ]
        selected_labels = {
            variable_id: ACS1_B08006_SELECTED_VARS[variable_id]
            for variable_id in selected_variables
        }
        return selected_variables, selected_labels
    return variables, labels


def get_raw_group_variables_and_labels(
    year: int,
    group_id: str,
    survey: str = "acs5",
    dataset_path: str = "",
) -> tuple[list[str], dict[str, str]]:
    return get_group_variables_and_labels(
        year,
        group_id,
        profile=False,
        survey=survey,
        dataset_path=dataset_path,
    ) if group_id not in {"B16002", "B08006"} else _get_raw_group_variables_and_labels_uncached(
        year, group_id, survey=survey, dataset_path=dataset_path
    )


def _get_raw_group_variables_and_labels_uncached(
    year: int,
    group_id: str,
    survey: str = "acs5",
    dataset_path: str = "",
    api_key: Optional[str] = None,
) -> tuple[list[str], dict[str, str]]:
    api_key = api_key or os.getenv("CENSUS_API_KEY")
    params = {"key": api_key} if api_key else None
    if dataset_path:
        url = f"https://api.census.gov/data/{year}/acs/{survey}/{dataset_path}/groups/{group_id}.json"
    else:
        url = f"https://api.census.gov/data/{year}/acs/{survey}/groups/{group_id}.json"
    resp = census_get(
        url,
        params=params,
        timeout=60,
        retries=3,
        request_label=f"group metadata {group_id} {survey.upper()} {year}",
    )
    if resp.status_code == 404:
        return [], {}
    resp.raise_for_status()
    try:
        group_info = resp.json()
    except ValueError:
        return [], {}
    variables = [
        v
        for v in group_info["variables"].keys()
        if v.endswith("E") and not v.endswith("PE")
    ]
    labels = {
        var_name: group_info["variables"][var_name]["label"]
        for var_name in variables
    }
    return variables, labels


def resolve_blockgroup_table_id(table_id: str, year: int) -> str:
    if year == 2010 and table_id == "B08006":
        return "B08301"
    return table_id


def get_2010_blockgroup_variables_and_labels(table_id: str) -> tuple[list[str], dict[str, str]]:
    if table_id == "B16002":
        full_variables, _ = _get_raw_group_variables_and_labels_uncached(2010, table_id, survey="acs5")
        available_set = set(full_variables)
        selected_variables = [
            variable_id for variable_id in LIMITED_ENGLISH_TABLE_VARS if variable_id in available_set
        ]
        selected_labels = {
            variable_id: LIMITED_ENGLISH_TABLE_VARS[variable_id]
            for variable_id in selected_variables
        }
        return selected_variables, selected_labels
    if table_id == "B15003":
        return [], {}
    if table_id == "B23025":
        return [], {}
    return _get_raw_group_variables_and_labels_uncached(2010, table_id, survey="acs5")


def summary_file_state_dir_name(state_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", state_name)
    return cleaned or state_name


def summary_file_cache_dir() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", Path.cwd()))
    return base / "CensusTool" / "acs_summary_file"


def download_to_cache(url: str, local_path: Path, logger: Callable[[str], None]) -> Path:
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger(f"Downloading summary file resource: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return local_path


def fetch_directory_listing(url: str) -> list[str]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return re.findall(r'href="([^"]+)"', resp.text, flags=re.IGNORECASE)


def find_summary_file_name(entries: list[str], prefix: str, suffix: str, contains: str = "") -> str:
    candidates = []
    for entry in entries:
        name = entry.split("/")[-1]
        lower = name.lower()
        if not lower.startswith(prefix.lower()):
            continue
        if not lower.endswith(suffix.lower()):
            continue
        if contains and contains.lower() not in lower:
            continue
        candidates.append(name)
    if not candidates:
        raise ValueError(f"Could not find summary file member matching {prefix}*{contains}*{suffix}.")
    candidates.sort()
    return candidates[0]


def sort_estimate_variables(variables: list[str]) -> list[str]:
    def sort_key(variable_id: str) -> tuple[int, str]:
        match = re.search(r"_(\d+)E$", variable_id)
        if match:
            return (int(match.group(1)), variable_id)
        return (10**9, variable_id)
    return sorted(variables, key=sort_key)


def load_2010_blockgroup_sequence_frame(
    state_name: str,
    state_fips: str,
    sequence_number: int,
    logger: Callable[[str], None],
    usecols: Optional[list[int]] = None,
) -> pd.DataFrame:
    state_dir = summary_file_state_dir_name(state_name)
    listing_url = (
        f"{ACS5_2010_SUMMARY_FILE_ROOT}/{state_dir}/Tracts_Block_Groups_Only/"
    )
    entries = fetch_directory_listing(listing_url)
    stusab = next((abbr.lower() for abbr, name in STATE_ABBR_TO_NAME.items() if name.lower() == state_name.lower()), "")
    if not stusab:
        raise ValueError(f"Unable to resolve state abbreviation for {state_name}.")
    seq_token = f"{sequence_number:04d}"
    file_name = find_summary_file_name(entries, "20105", ".zip", contains=f"{stusab}{seq_token}")
    local_path = summary_file_cache_dir() / "2010" / "5_year" / state_dir / "Tracts_Block_Groups_Only" / file_name
    download_to_cache(urljoin(listing_url, file_name), local_path, logger)
    with zipfile.ZipFile(local_path) as zf:
        member_name = next(
            (
                name
                for name in zf.namelist()
                if name.lower().endswith(".txt") and f"{stusab}{seq_token}" in name.lower()
            ),
            None,
        )
        if member_name is None:
            member_name = next((name for name in zf.namelist() if name.lower().endswith(".txt")), None)
        if member_name is None:
            raise ValueError(f"No sequence text file found inside {file_name}.")
        with zf.open(member_name) as handle:
            return pd.read_csv(handle, header=None, dtype=str, low_memory=False, usecols=usecols)


def load_2010_blockgroup_geography_frame(
    state_name: str,
    logger: Callable[[str], None],
) -> pd.DataFrame:
    state_dir = summary_file_state_dir_name(state_name)
    listing_url = (
        f"{ACS5_2010_SUMMARY_FILE_ROOT}/{state_dir}/Tracts_Block_Groups_Only/"
    )
    entries = fetch_directory_listing(listing_url)
    file_name = find_summary_file_name(entries, "g", ".csv")
    local_path = summary_file_cache_dir() / "2010" / "5_year" / state_dir / "Tracts_Block_Groups_Only" / file_name
    download_to_cache(urljoin(listing_url, file_name), local_path, logger)
    # The 2010 ACS summary-file geography CSV is headerless. Read the needed
    # positions explicitly so downstream code gets canonical column names.
    return pd.read_csv(
        local_path,
        header=None,
        dtype=str,
        low_memory=False,
        usecols=[2, 4, 9, 10, 13, 14, 48, 49],
        names=["SUMLEVEL", "LOGRECNO", "STATE", "COUNTY", "TRACT", "BLKGRP", "GEOID", "NAME"],
    )


def fetch_blockgroup_table_2010_summary_file(
    state_name: str,
    state_fips: str,
    county_fips: str,
    table_id: str,
    variables: list[str],
    logger: Callable[[str], None],
) -> pd.DataFrame:
    layout = ACS5_2010_BLOCKGROUP_SUMMARY_FILE_LAYOUTS.get(table_id)
    if layout is None:
        logger(f"WARN: table {table_id} is not configured for 2010 block group summary-file fallback; skipping.")
        return pd.DataFrame()

    full_variables, _ = _get_raw_group_variables_and_labels_uncached(2010, table_id, survey="acs5")
    full_variable_order = sort_estimate_variables(full_variables)
    sequence_number, start_pos, end_pos = layout
    expected_cells = end_pos - start_pos + 1
    if len(full_variable_order) < expected_cells:
        logger(
            f"WARN: table {table_id} metadata has {len(full_variable_order)} estimate cells but summary file expects {expected_cells}; skipping."
        )
        return pd.DataFrame()
    full_variable_order = full_variable_order[:expected_cells]

    requested_variables = [variable_id for variable_id in variables if variable_id in full_variable_order]
    desired_variables = [variable_id for variable_id in full_variable_order if variable_id in requested_variables]
    desired_indices = [full_variable_order.index(variable_id) for variable_id in desired_variables]
    usecols = [5] + [start_pos - 1 + index for index in desired_indices]
    logger(f"Fetching {table_id} from 2010 ACS summary file sequence {sequence_number:04d}")
    sequence_df = load_2010_blockgroup_sequence_frame(
        state_name,
        state_fips,
        sequence_number,
        logger,
        usecols=usecols,
    )
    geography_df = load_2010_blockgroup_geography_frame(state_name, logger)

    geography_df.columns = [str(col).strip().upper() for col in geography_df.columns]
    required_geo_cols = {"LOGRECNO", "SUMLEVEL", "STATE", "COUNTY", "TRACT", "BLKGRP", "NAME"}
    missing_geo_cols = sorted(required_geo_cols - set(geography_df.columns))
    if missing_geo_cols:
        raise ValueError(
            f"2010 summary geography file is missing required columns: {', '.join(missing_geo_cols)}"
        )

    geo = geography_df[
        (geography_df["SUMLEVEL"].astype(str).str.zfill(3) == "150")
        & (geography_df["STATE"].astype(str).str.zfill(2) == state_fips)
        & (geography_df["COUNTY"].astype(str).str.zfill(3) == county_fips)
    ][["LOGRECNO", "STATE", "COUNTY", "TRACT", "BLKGRP", "NAME"]].copy()
    if geo.empty:
        return pd.DataFrame()

    seq_subset = sequence_df.copy()
    seq_subset.columns = ["LOGRECNO"] + desired_variables

    df = geo.merge(seq_subset, on="LOGRECNO", how="inner")
    for column in desired_variables:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df.rename(
        columns={
            "NAME": "Geographic Area Name",
            "STATE": "state",
            "COUNTY": "county",
            "TRACT": "tract",
            "BLKGRP": "block group",
        },
        inplace=True,
    )
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["tract"] = df["tract"].astype(str).str.zfill(6)
    df["block group"] = df["block group"].astype(str).str.zfill(1)
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    df["Year"] = 2010
    ordered = ["Geographic Area Name", "state", "county", "tract", "block group", "GEOID", "Year"] + requested_variables
    return df[[column for column in ordered if column in df.columns]]


def get_acs1_dp05_selected_variables(
    year: int, logger: Optional[Callable[[str], None]] = None
) -> tuple[list[str], dict[str, str]]:
    selected_map = dict(ACS1_DP05_SELECTED_VARS_BY_YEAR.get(year, {}))
    available_variables, available_labels = get_group_variables_and_labels(
        year, "DP05", profile=True, survey="acs1"
    )
    if not available_variables:
        return [], {}

    available_set = set(available_variables)
    selected_variables = [variable_id for variable_id in selected_map if variable_id in available_set]
    selected_labels = {
        variable_id: selected_map[variable_id]
        for variable_id in selected_variables
    }

    if logger:
        for variable_id in selected_variables:
            logger(
                f"ACS1 DP05 mapping {year}: {selected_labels[variable_id]} -> "
                f"{variable_id} ({available_labels.get(variable_id, '')})"
            )

    return selected_variables, selected_labels


def get_acs1_selected_variables_for_table(
    year: int,
    table_id: str,
    dataset_path: str,
    logger: Optional[Callable[[str], None]] = None,
) -> tuple[list[str], dict[str, str]]:
    if table_id == "DP05":
        return get_acs1_dp05_selected_variables(year, logger=logger)
    if table_id == "B01001":
        available_variables, _ = get_group_variables_and_labels(
            year,
            table_id,
            profile=False,
            survey="acs1",
            dataset_path=dataset_path,
        )
        if not available_variables:
            return [], {}
        available_set = set(available_variables)
        selected_variables = [variable_id for variable_id in ACS1_B01001_SELECTED_VARS if variable_id in available_set]
        selected_labels = {
            variable_id: ACS1_B01001_SELECTED_VARS[variable_id]
            for variable_id in selected_variables
        }
        return selected_variables, selected_labels

    if table_id == "B11001":
        return get_group_variables_and_labels(
            year,
            table_id,
            profile=False,
            survey="acs1",
            dataset_path=dataset_path,
        )

    if table_id == "S2501":
        selected_map = (
            ACS1_S2501_SELECTED_VARS_2010
            if year == 2010
            else ACS1_S2501_SELECTED_VARS_2011_2021
        )
    elif table_id == "B19001":
        selected_map = ACS1_B19001_SELECTED_VARS
    elif table_id == "B23025":
        selected_map = ACS1_B23025_SELECTED_VARS
    elif table_id == "B08303":
        selected_map = ACS1_B08303_SELECTED_VARS
    elif table_id == "S1810":
        selected_map = ACS1_S1810_SELECTED_VARS
    elif table_id == "B25044":
        selected_map = ACS1_B25044_SELECTED_VARS
    elif table_id == "B08006":
        selected_map = ACS1_B08006_SELECTED_VARS
    elif table_id == "S1602":
        selected_map = ACS1_S1602_SELECTED_VARS
    elif table_id == "B15003":
        available_variables, available_labels = get_group_variables_and_labels(
            year,
            table_id,
            profile=False,
            survey="acs1",
            dataset_path=dataset_path,
        )
        return available_variables, available_labels
    else:
        return [], {}

    available_variables, _ = get_group_variables_and_labels(
        year,
        table_id,
        profile=(dataset_path == "profile"),
        survey="acs1",
        dataset_path=dataset_path,
    )
    if not available_variables:
        return [], {}

    available_set = set(available_variables)
    selected_variables = [variable_id for variable_id in selected_map if variable_id in available_set]
    selected_labels = {variable_id: selected_map[variable_id] for variable_id in selected_variables}
    return selected_variables, selected_labels


def get_acs1_county_place_table_configs(year: int, tables: Optional[dict[str, str]] = None) -> list[dict]:
    tables = tables or ACS1_COUNTY_PLACE_TABLES
    configs = [
        {
            "table_id": "DP05",
            "sheet_name": "DP05 Demographics",
            "dataset_path": "profile",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "B01001",
            "sheet_name": "Age by Sex",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "B15003",
            "sheet_name": "Educational Attainment",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "B23025",
            "sheet_name": "Employment Status (16+)",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "B08303",
            "sheet_name": "Travel Time to Work",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "S1810",
            "sheet_name": "Disability Status",
            "dataset_path": "subject",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "S2501",
            "sheet_name": "Household Number",
            "dataset_path": "subject",
            "variables": {},
            "year_min": 2014,
            "year_max": 2021,
        },
        {
            "table_id": "B11001",
            "sheet_name": "Household Type (Including Living Alone)",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "B19001",
            "sheet_name": "Household Income",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "B25044",
            "sheet_name": "Vehicle Ownership",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "B08006",
            "sheet_name": "Means of Transportation",
            "dataset_path": "",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
        {
            "table_id": "S1602",
            "sheet_name": "Limited English Speaking",
            "dataset_path": "subject",
            "variables": {},
            "year_min": 2014,
            "year_max": 2024,
        },
    ]
    return [
        config
        for config in configs
        if config["table_id"] in tables and config["year_min"] <= year <= config["year_max"] and year != 2020
    ]


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
        resp = census_get(
            base_url,
            params=params,
            timeout=60,
            retries=3,
            logger=logger,
            request_label=f"{table_id} chunk {i}/{len(chunks)}",
        )
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


def fetch_tract_table(
    year: int,
    state_fips: str,
    county_fips: str,
    api_key: str,
    table_id: str,
    variables: list,
    logger: Callable[[str], None],
    survey: str = "acs5",
) -> pd.DataFrame:
    base_url = f"https://api.census.gov/data/{year}/acs/{survey}"
    full_df = None
    chunks = list(chunk_list(variables, 49))

    for i, var_chunk in enumerate(chunks, start=1):
        params = {
            "get": ",".join(var_chunk + ["NAME"]),
            "for": "tract:*",
            "in": f"state:{state_fips} county:{county_fips}",
            "key": api_key,
        }

        logger(f"Fetching {table_id} chunk {i}/{len(chunks)}")
        resp = census_get(
            base_url,
            params=params,
            timeout=60,
            retries=3,
            logger=logger,
            request_label=f"{table_id} chunk {i}/{len(chunks)}",
        )
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
            chunk_df[numeric_cols] = chunk_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        if full_df is None:
            full_df = chunk_df
        else:
            if "NAME" in chunk_df.columns:
                chunk_df = chunk_df.drop(columns=["NAME"])
            full_df = full_df.merge(chunk_df, on=["state", "county", "tract"], how="inner")

    if full_df is None or full_df.empty:
        return pd.DataFrame()

    full_df.rename(columns={"NAME": "Geographic Area Name"}, inplace=True)
    full_df["state"] = full_df["state"].astype(str).str.zfill(2)
    full_df["county"] = full_df["county"].astype(str).str.zfill(3)
    full_df["tract"] = full_df["tract"].astype(str).str.zfill(6)
    full_df["GEOID"] = full_df["state"] + full_df["county"] + full_df["tract"]
    full_df["Year"] = year
    return full_df


def fetch_county_table(
    year: int,
    state_fips: str,
    county_fips: str,
    api_key: str,
    table_id: str,
    variables: list,
    logger: Callable[[str], None],
    survey: str = "acs1",
    profile: bool = False,
    dataset_path: str = "",
) -> pd.DataFrame:
    base_url = f"https://api.census.gov/data/{year}/acs/{survey}"
    if dataset_path:
        base_url += f"/{dataset_path}"
    elif profile:
        base_url += "/profile"
    full_df = None
    chunks = list(chunk_list(variables, 49))

    for i, var_chunk in enumerate(chunks, start=1):
        params = {
            "get": ",".join(var_chunk + ["NAME"]),
            "for": f"county:{county_fips}",
            "in": f"state:{state_fips}",
            "key": api_key,
        }

        logger(f"Fetching {table_id} chunk {i}/{len(chunks)}")
        resp = census_get(
            base_url,
            params=params,
            timeout=60,
            retries=3,
            logger=logger,
            request_label=f"{table_id} chunk {i}/{len(chunks)}",
        )
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
            chunk_df[numeric_cols] = chunk_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        if full_df is None:
            full_df = chunk_df
        else:
            if "NAME" in chunk_df.columns:
                chunk_df = chunk_df.drop(columns=["NAME"])
            full_df = full_df.merge(chunk_df, on=["state", "county"], how="inner")

    if full_df is None or full_df.empty:
        return pd.DataFrame()

    full_df.rename(columns={"NAME": "Geographic Area Name"}, inplace=True)
    full_df["state"] = full_df["state"].astype(str).str.zfill(2)
    full_df["county"] = full_df["county"].astype(str).str.zfill(3)
    full_df["GEOID"] = full_df["state"] + full_df["county"]
    if survey == "acs1":
        full_df = add_acs1_row_total(full_df, variables)
    full_df["Year"] = year
    return full_df


def fetch_place_table(
    year: int,
    state_fips: str,
    place_fips: str,
    api_key: str,
    table_id: str,
    variables: list,
    logger: Callable[[str], None],
    survey: str = "acs1",
    profile: bool = False,
    dataset_path: str = "",
) -> pd.DataFrame:
    base_url = f"https://api.census.gov/data/{year}/acs/{survey}"
    if dataset_path:
        base_url += f"/{dataset_path}"
    elif profile:
        base_url += "/profile"
    full_df = None
    chunks = list(chunk_list(variables, 49))

    for i, var_chunk in enumerate(chunks, start=1):
        params = {
            "get": ",".join(var_chunk + ["NAME"]),
            "for": f"place:{place_fips}",
            "in": f"state:{state_fips}",
            "key": api_key,
        }

        logger(f"Fetching {table_id} chunk {i}/{len(chunks)}")
        resp = census_get(
            base_url,
            params=params,
            timeout=60,
            retries=3,
            logger=logger,
            request_label=f"{table_id} chunk {i}/{len(chunks)}",
        )
        if resp.status_code != 200:
            logger(f"ERROR: {table_id} chunk {i} failed: {resp.status_code} {resp.text[:200]}")
            continue

        try:
            data = resp.json()
        except ValueError:
            logger(
                f"ERROR: invalid JSON response for {table_id} chunk {i}/{len(chunks)}: "
                f"status={resp.status_code}, content-type={resp.headers.get('Content-Type')}, "
                f"body={resp.text[:300]!r}"
            )
            continue
        if not data or len(data) < 2:
            logger(f"WARN: empty response for {table_id} chunk {i}")
            continue

        chunk_df = pd.DataFrame(data[1:], columns=data[0])
        if chunk_df.columns.duplicated().any():
            chunk_df = chunk_df.loc[:, ~chunk_df.columns.duplicated(keep="first")]

        numeric_cols = [c for c in var_chunk if c in chunk_df.columns and c != "NAME"]
        if numeric_cols:
            chunk_df[numeric_cols] = chunk_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        if full_df is None:
            full_df = chunk_df
        else:
            if "NAME" in chunk_df.columns:
                chunk_df = chunk_df.drop(columns=["NAME"])
            full_df = full_df.merge(chunk_df, on=["state", "place"], how="inner")

    if full_df is None or full_df.empty:
        return pd.DataFrame()

    full_df.rename(columns={"NAME": "Geographic Area Name"}, inplace=True)
    full_df["state"] = full_df["state"].astype(str).str.zfill(2)
    full_df["place"] = full_df["place"].astype(str).str.zfill(5)
    full_df["GEOID"] = full_df["state"] + full_df["place"]
    if survey == "acs1":
        full_df = add_acs1_row_total(full_df, variables)
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
        elif geoid_len == 7:
            needed = ["state", "place"]
            if all(c in df.columns for c in needed):
                df["state"] = df["state"].astype(str).str.zfill(2)
                df["place"] = df["place"].astype(str).str.zfill(5)
                df["GEOID"] = df["state"] + df["place"]
            else:
                raise ValueError(
                    f"Sheet '{sheet_name}' has no GEOID and cannot build from components."
                )
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(geoid_len)
    drop_geo_parts = [c for c in ["state", "county", "tract", "block group", "place"] if c in df.columns]
    if drop_geo_parts:
        df.drop(columns=drop_geo_parts, inplace=True)
    return df


def prepare_geometry_geoid(gdf: gpd.GeoDataFrame, geoid_len: int, source_label: str) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if "GEOID" not in gdf.columns:
        candidate_cols = [
            col for col in gdf.columns if "GEOID" in str(col).upper() or "FIPS" in str(col).upper()
        ]
        if not candidate_cols:
            raise ValueError(
                f"{source_label} must contain a GEOID or FIPS column. Found: {list(gdf.columns)}"
            )
        preferred = next(
            (
                col
                for col in candidate_cols
                if str(col).upper() in {"GEOID", "GEOID10", "GEOID20", "FIPS", "FIPS10", "FIPS20"}
            ),
            candidate_cols[0],
        )
        gdf = gdf.rename(columns={preferred: "GEOID"})
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(geoid_len)
    return gdf


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
        resp = census_get(
            base_url,
            params=params,
            timeout=60,
            retries=3,
            logger=logger,
            request_label=f"{group_id} chunk {i}/{len(chunks)}",
        )
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
    # [FIX] Transient server errors (502/503/504) were previously re-raised
    # immediately via raise_for_status(), aborting all remaining candidates and
    # the FIPS fallback.  Now each candidate is retried up to 3 times with
    # exponential back-off; only non-retryable HTTP errors stop that candidate.
    _RETRYABLE = {502, 503, 504}
    _MAX_RETRIES = 3

    for address in candidates:
        params = {
            "address": address,
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "format": "json",
        }
        last_status = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=60)
            except requests.exceptions.RequestException:
                # Network-level failure – try next candidate immediately.
                break
            last_status = resp.status_code
            if resp.status_code in _RETRYABLE:
                # Transient gateway error – wait and retry the same address.
                if attempt < _MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                else:
                    # Exhausted retries for this candidate; try the next one.
                    break
            if resp.status_code != 200:
                # Non-retryable HTTP error for this candidate; try the next.
                break
            # Successful response – parse and return if a match is found.
            try:
                data = resp.json()
            except Exception:
                break
            matches = data.get("result", {}).get("addressMatches", [])
            if not matches:
                break  # No match for this candidate address; try the next.
            geos = matches[0].get("geographies", {}).get("Counties", [])
            if not geos:
                break
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


def lookup_place_state(place: str, state: str) -> tuple[str, str, str]:
    place = place.strip()
    state_lookup = state.strip()
    if len(state_lookup) == 2:
        state_lookup = STATE_ABBR_TO_NAME.get(state_lookup.upper(), state_lookup)

    places_path = Path(__file__).resolve().parent / "Places_data.xlsx"
    if not places_path.exists():
        raise ValueError("Places_data.xlsx not found.")

    df = pd.read_excel(places_path)
    df = df[df["PLACE"].fillna(0).astype(int) > 0].copy()
    df["STNAME"] = df["STNAME"].astype(str).str.strip()
    state_df = df[df["STNAME"].str.casefold() == state_lookup.casefold()].copy()
    if state_df.empty:
        raise ValueError(f"No places found for state '{state}'.")

    state_df["NAME"] = state_df["NAME"].astype(str).str.strip()
    matches = state_df[state_df["NAME"].str.casefold() == place.casefold()]
    if matches.empty:
        target = re.sub(r"\s+", " ", place).strip().casefold()
        matches = state_df[
            state_df["NAME"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold()
            == target
        ]
    if matches.empty:
        raise ValueError(f"No place match found for '{place}, {state_lookup}'.")

    row = matches.iloc[0]
    state_fips = str(row["STATE"]).zfill(2)
    place_fips = str(row["PLACE"]).zfill(5)
    place_name = str(row["NAME"]).strip()
    return state_fips, place_fips, place_name


def run_blockgroup_pipeline(
    city: str,
    state: str,
    year: int,
    output_dir: Path,
    shapefile_path: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    geoids: Optional[list[str]] = None,
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
    summary_frames: dict[str, pd.DataFrame] = {}
    summary_label_maps: dict[str, dict[str, str]] = {}
    data_dictionary_rows: list[dict] = []

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
        if geoids:
            df = df[df["GEOID"].isin(geoids)]
            if df.empty:
                logger(f"WARN: no data returned for {table_id} after GEOID filtering")
                continue

        safe_labels = sanitize_field_names(labels)
        data_dictionary_rows.extend(
            build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "block_groups", "acs5", year)
        )
        summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
        df.rename(columns=safe_labels, inplace=True)
        summary_frames[sheet_name] = df.copy()
        df.to_excel(writer, sheet_name=sanitize_sheet_name(sheet_name), index=False)

    write_summary_sheet(writer, build_blockgroup_summary(summary_frames, summary_label_maps))
    write_data_dictionary_sheet(writer, data_dictionary_rows)
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


def run_blockgroup_trend_pipeline(
    city: str,
    state: str,
    years: list[int],
    output_dir: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    geoids: Optional[list[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")
    if not years or len(set(years)) < 2:
        raise ValueError("Trend mode requires at least two years.")

    tables = tables or BLOCKGROUP_TABLES
    sorted_years = sorted({int(year) for year in years})
    state_fips, county_fips, county_name_raw = geocode_city_state(city, state)
    county_name = normalize_name(county_name_raw.replace(" County", ""))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = output_dir / f"{county_name}_ACS_BlockGroups_Trend_{sorted_years[0]}_{sorted_years[-1]}.xlsx"

    logger(f"Using county block group trend: {county_name_raw} (state {state_fips}, county {county_fips})")
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    combined_frames_by_sheet: dict[str, list[pd.DataFrame]] = {}
    yearly_summaries: dict[int, dict] = {}
    data_dictionary_rows: list[dict] = []

    for year in sorted_years:
        logger(f"Building block group trend data for {year}")
        summary_frames: dict[str, pd.DataFrame] = {}
        summary_label_maps: dict[str, dict[str, str]] = {}
        for table_id, sheet_name in tables.items():
            effective_table_id = resolve_blockgroup_table_id(table_id, year)
            if year == 2010:
                variables, labels = get_2010_blockgroup_variables_and_labels(effective_table_id)
            else:
                variables, labels = get_group_variables_and_labels(year, effective_table_id)
            if not variables:
                logger(acs5_table_availability_message(effective_table_id, year))
                continue

            if year == 2010:
                df = fetch_blockgroup_table_2010_summary_file(
                    state,
                    state_fips,
                    county_fips,
                    effective_table_id,
                    variables,
                    logger,
                )
            else:
                df = fetch_blockgroup_table(
                    year, state_fips, county_fips, api_key, effective_table_id, variables, logger
                )
            if df.empty:
                logger(f"WARN: no data returned for {effective_table_id} ({year})")
                continue
            if geoids:
                df = df[df["GEOID"].isin(geoids)]
                if df.empty:
                    logger(f"WARN: no data returned for {effective_table_id} ({year}) after GEOID filtering")
                    continue

            safe_labels = sanitize_field_names(labels)
            data_dictionary_rows.extend(
                build_data_dictionary_rows(
                    effective_table_id,
                    sheet_name,
                    labels,
                    safe_labels,
                    "block_groups",
                    "acs5",
                    year,
                )
            )
            summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
            df.rename(columns=safe_labels, inplace=True)
            summary_frames[sheet_name] = df.copy()

            out_df = prepare_trend_output_frame(sheet_name, df, year)
            combined_frames_by_sheet.setdefault(sheet_name, []).append(out_df)

        yearly_summaries[year] = build_blockgroup_summary(summary_frames, summary_label_maps)

    for sheet_name, frames in combined_frames_by_sheet.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(
                writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
            )

    if yearly_summaries:
        write_summary_sheet(writer, yearly_summaries[sorted_years[-1]])
        write_trend_sheet(writer, yearly_summaries)
    write_data_dictionary_sheet(writer, data_dictionary_rows)
    writer.close()
    logger("Done.")

    return RunResult(
        output_xlsx=output_xlsx,
        output_gpkg=None,
        output_layer="",
        county_name=county_name_raw,
        state_fips=state_fips,
        county_fips=county_fips,
        year=sorted_years[-1],
        geography="block_groups_trend",
    )


def run_county_pipeline(
    city: str,
    state: str,
    year: int,
    output_dir: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    comparison_locations: Optional[list[dict]] = None,
    survey: str = "acs1",
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")

    tables = tables or BLOCKGROUP_TABLES
    targets = []
    for item in dedupe_locations(city, state, comparison_locations):
        location = item["location"]
        target_state = item["state"]
        state_fips, county_fips, county_name_raw = geocode_city_state(location, target_state)
        targets.append(
            {
                "query_name": location,
                "query_state": target_state,
                "state_fips": state_fips,
                "county_fips": county_fips,
                "name_raw": county_name_raw,
                "display_name": f"{county_name_raw}, {target_state}",
                "safe_name": normalize_name(county_name_raw.replace(" County", "")),
            }
        )
    primary_target = targets[0]
    county_name = primary_target["safe_name"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_xlsx = output_dir / f"{county_name}_ACS{survey.upper()}_County_{year}.xlsx"

    logger(
        f"Using county: {primary_target['name_raw']} "
        f"in {primary_target['query_state']} "
        f"(state {primary_target['state_fips']}, county {primary_target['county_fips']})"
    )
    if len(targets) > 1:
        logger("Peer comparison counties: " + ", ".join(t["display_name"] for t in targets[1:]))
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    target_frames: dict[str, dict[str, pd.DataFrame]] = {t["display_name"]: {} for t in targets}
    summary_label_maps: dict[str, dict[str, str]] = {}
    data_dictionary_rows: list[dict] = []
    if survey == "acs1":
        table_configs = get_acs1_county_place_table_configs(year, tables)
        for config in table_configs:
            table_id = config["table_id"]
            sheet_name = config["sheet_name"]
            variables, labels = get_acs1_selected_variables_for_table(
                year, table_id, config["dataset_path"], logger=logger
            )
            if not variables:
                logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                continue
            safe_labels = sanitize_field_names(labels)
            data_dictionary_rows.extend(
                build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "county", survey, year)
            )
            summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
            combined_frames = []
            for target in targets:
                df = fetch_county_table(
                    year,
                    target["state_fips"],
                    target["county_fips"],
                    api_key,
                    table_id,
                    variables,
                    logger,
                    survey=survey,
                    dataset_path=config["dataset_path"],
                )
                if df.empty:
                    logger(f"WARN: no data returned for {table_id} ({target['display_name']})")
                    continue
                df.rename(columns=safe_labels, inplace=True)
                target_frames[target["display_name"]][sheet_name] = df.copy()
                combined_frames.append(df)
            if combined_frames:
                pd.concat(combined_frames, ignore_index=True).to_excel(
                    writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
                )
    else:
        for table_id, sheet_name in tables.items():
            variables, labels = get_group_variables_and_labels(year, table_id, survey=survey)
            if not variables:
                logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                continue

            safe_labels = sanitize_field_names(labels)
            data_dictionary_rows.extend(
                build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "county", survey, year)
            )
            summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
            combined_frames = []
            for target in targets:
                df = fetch_county_table(
                    year,
                    target["state_fips"],
                    target["county_fips"],
                    api_key,
                    table_id,
                    variables,
                    logger,
                    survey=survey,
                )
                if df.empty:
                    logger(f"WARN: no data returned for {table_id} ({target['display_name']})")
                    continue
                df.rename(columns=safe_labels, inplace=True)
                target_frames[target["display_name"]][sheet_name] = df.copy()
                combined_frames.append(df)
            if combined_frames:
                pd.concat(combined_frames, ignore_index=True).to_excel(
                    writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
                )

    peer_summaries = {
        target["display_name"]: (
            build_acs1_dp05_summary(target_frames[target["display_name"]])
            if survey == "acs1"
            else build_blockgroup_summary(target_frames[target["display_name"]], summary_label_maps)
        )
        for target in targets
    }
    write_summary_sheet(writer, peer_summaries[primary_target["display_name"]])
    write_peer_comparison_sheet(writer, peer_summaries)
    write_data_dictionary_sheet(writer, data_dictionary_rows)

    writer.close()
    logger("Done.")

    return RunResult(
        output_xlsx=output_xlsx,
        output_gpkg=None,
        output_layer="",
        county_name=primary_target["display_name"],
        state_fips=primary_target["state_fips"],
        county_fips=primary_target["county_fips"],
        year=year,
        geography="county",
    )


def run_county_trend_pipeline(
    city: str,
    state: str,
    years: list[int],
    output_dir: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    comparison_locations: Optional[list[dict]] = None,
    survey: str = "acs1",
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")
    if not years or len(set(years)) < 2:
        raise ValueError("Trend mode requires at least two years.")

    tables = tables or (ACS1_COUNTY_PLACE_TABLES if survey == "acs1" else BLOCKGROUP_TABLES)
    sorted_years = sorted({int(year) for year in years})
    targets = []
    for item in dedupe_locations(city, state, comparison_locations):
        location = item["location"]
        target_state = item["state"]
        state_fips, county_fips, county_name_raw = geocode_city_state(location, target_state)
        targets.append(
            {
                "query_name": location,
                "query_state": target_state,
                "state_fips": state_fips,
                "county_fips": county_fips,
                "name_raw": county_name_raw,
                "display_name": f"{county_name_raw}, {target_state}",
                "safe_name": normalize_name(county_name_raw.replace(" County", "")),
            }
        )
    primary_target = targets[0]
    county_name = primary_target["safe_name"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = output_dir / f"{county_name}_ACS{survey.upper()}_County_Trend_{sorted_years[0]}_{sorted_years[-1]}.xlsx"

    logger(
        f"Using county trend: {primary_target['name_raw']} in {primary_target['query_state']} "
        f"(state {primary_target['state_fips']}, county {primary_target['county_fips']})"
    )
    if len(targets) > 1:
        logger("Peer comparison counties: " + ", ".join(t["display_name"] for t in targets[1:]))
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    combined_frames_by_sheet: dict[str, list[pd.DataFrame]] = {}
    yearly_summaries_by_target: dict[str, dict[int, dict]] = {
        target["display_name"]: {} for target in targets
    }
    data_dictionary_rows: list[dict] = []

    for year in sorted_years:
        logger(f"Building trend data for {year}")
        summary_frames_by_target: dict[str, dict[str, pd.DataFrame]] = {
            target["display_name"]: {} for target in targets
        }
        summary_label_maps: dict[str, dict[str, str]] = {}
        if survey == "acs1":
            table_configs = get_acs1_county_place_table_configs(year, tables)
            for config in table_configs:
                table_id = config["table_id"]
                sheet_name = config["sheet_name"]
                variables, labels = get_acs1_selected_variables_for_table(
                    year, table_id, config["dataset_path"], logger=logger
                )
                if not variables:
                    logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                    continue
                safe_labels = sanitize_field_names(labels)
                data_dictionary_rows.extend(
                    build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "county", survey, year)
                )
                summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
                for target in targets:
                    df = fetch_county_table(
                        year,
                        target["state_fips"],
                        target["county_fips"],
                        api_key,
                        table_id,
                        variables,
                        logger,
                        survey=survey,
                        dataset_path=config["dataset_path"],
                    )
                    if df.empty:
                        logger(f"WARN: no data returned for {table_id} ({target['display_name']}, {year})")
                        continue

                    df.rename(columns=safe_labels, inplace=True)
                    summary_frames_by_target[target["display_name"]][sheet_name] = df.copy()
                    out_df = prepare_trend_output_frame(
                        sheet_name,
                        df,
                        year,
                        location_value=target["display_name"],
                    )
                    combined_frames_by_sheet.setdefault(sheet_name, []).append(out_df)
        else:
            for table_id, sheet_name in tables.items():
                variables, labels = get_group_variables_and_labels(year, table_id, survey=survey)
                if not variables:
                    logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                    continue
                safe_labels = sanitize_field_names(labels)
                data_dictionary_rows.extend(
                    build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "county", survey, year)
                )
                summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
                for target in targets:
                    df = fetch_county_table(
                        year,
                        target["state_fips"],
                        target["county_fips"],
                        api_key,
                        table_id,
                        variables,
                        logger,
                        survey=survey,
                    )
                    if df.empty:
                        logger(f"WARN: no data returned for {table_id} ({target['display_name']}, {year})")
                        continue

                    df.rename(columns=safe_labels, inplace=True)
                    summary_frames_by_target[target["display_name"]][sheet_name] = df.copy()
                    out_df = prepare_trend_output_frame(
                        sheet_name,
                        df,
                        year,
                        location_value=target["display_name"],
                    )
                    combined_frames_by_sheet.setdefault(sheet_name, []).append(out_df)

        for target in targets:
            target_name = target["display_name"]
            yearly_summaries_by_target[target_name][year] = (
                build_acs1_dp05_summary(summary_frames_by_target[target_name])
                if survey == "acs1"
                else build_blockgroup_summary(summary_frames_by_target[target_name], summary_label_maps)
            )

    for sheet_name, frames in combined_frames_by_sheet.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(
                writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
            )

    primary_yearly_summaries = yearly_summaries_by_target[primary_target["display_name"]]
    write_summary_sheet(writer, primary_yearly_summaries[sorted_years[-1]])
    write_trend_sheet(writer, primary_yearly_summaries)
    for target in targets[1:]:
        target_name = target["display_name"]
        trend_sheet_name = sanitize_sheet_name(f"Trend - {target_name}")
        write_trend_sheet(writer, yearly_summaries_by_target[target_name], sheet_name=trend_sheet_name)
    write_data_dictionary_sheet(writer, data_dictionary_rows)
    writer.close()
    logger("Done.")

    return RunResult(
        output_xlsx=output_xlsx,
        output_gpkg=None,
        output_layer="",
        county_name=primary_target["name_raw"],
        state_fips=primary_target["state_fips"],
        county_fips=primary_target["county_fips"],
        year=sorted_years[-1],
        geography="county_trend",
    )


def run_place_pipeline(
    city: str,
    state: str,
    year: int,
    output_dir: Path,
    shapefile_path: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    comparison_locations: Optional[list[dict]] = None,
    survey: str = "acs1",
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")

    tables = tables or (ACS1_COUNTY_PLACE_TABLES if survey == "acs1" else BLOCKGROUP_TABLES)
    targets = []
    for item in dedupe_locations(city, state, comparison_locations):
        location = item["location"]
        target_state = item["state"]
        state_fips, place_fips, place_name_raw = lookup_place_state(location, target_state)
        targets.append(
            {
                "query_name": location,
                "query_state": target_state,
                "state_fips": state_fips,
                "place_fips": place_fips,
                "name_raw": place_name_raw,
                "display_name": f"{place_name_raw}, {target_state}",
                "safe_name": normalize_name(place_name_raw),
            }
        )
    primary_target = targets[0]
    place_name = primary_target["safe_name"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_xlsx = output_dir / f"{place_name}_ACS_{survey.upper()}_Place_{year}.xlsx"
    output_gpkg = output_dir / f"Merged_{place_name}_Place_{year}.gpkg"
    output_layer = f"places_acs_{year}"

    logger(
        f"Using place: {primary_target['name_raw']} "
        f"in {primary_target['query_state']} "
        f"(state {primary_target['state_fips']}, place {primary_target['place_fips']})"
    )
    if len(targets) > 1:
        logger("Peer comparison places: " + ", ".join(t["display_name"] for t in targets[1:]))
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    target_frames: dict[str, dict[str, pd.DataFrame]] = {t["display_name"]: {} for t in targets}
    summary_label_maps: dict[str, dict[str, str]] = {}
    data_dictionary_rows: list[dict] = []
    if survey == "acs1":
        table_configs = get_acs1_county_place_table_configs(year, tables)
        for config in table_configs:
            table_id = config["table_id"]
            sheet_name = config["sheet_name"]
            variables, labels = get_acs1_selected_variables_for_table(
                year, table_id, config["dataset_path"], logger=logger
            )
            if not variables:
                logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                continue
            safe_labels = sanitize_field_names(labels)
            data_dictionary_rows.extend(
                build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "place", survey, year)
            )
            summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
            combined_frames = []
            for target in targets:
                df = fetch_place_table(
                    year,
                    target["state_fips"],
                    target["place_fips"],
                    api_key,
                    table_id,
                    variables,
                    logger,
                    survey=survey,
                    dataset_path=config["dataset_path"],
                )
                if df.empty:
                    logger(f"WARN: no data returned for {table_id} ({target['display_name']})")
                    continue
                df.rename(columns=safe_labels, inplace=True)
                target_frames[target["display_name"]][sheet_name] = df.copy()
                combined_frames.append(df)
            if combined_frames:
                pd.concat(combined_frames, ignore_index=True).to_excel(
                    writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
                )
    else:
        for table_id, sheet_name in tables.items():
            variables, labels = get_group_variables_and_labels(year, table_id, survey=survey)
            if not variables:
                logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                continue

            safe_labels = sanitize_field_names(labels)
            data_dictionary_rows.extend(
                build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "place", survey, year)
            )
            summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
            combined_frames = []
            for target in targets:
                df = fetch_place_table(
                    year,
                    target["state_fips"],
                    target["place_fips"],
                    api_key,
                    table_id,
                    variables,
                    logger,
                    survey=survey,
                )
                if df.empty:
                    logger(f"WARN: no data returned for {table_id} ({target['display_name']})")
                    continue
                df.rename(columns=safe_labels, inplace=True)
                target_frames[target["display_name"]][sheet_name] = df.copy()
                combined_frames.append(df)
            if combined_frames:
                pd.concat(combined_frames, ignore_index=True).to_excel(
                    writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
                )

    peer_summaries = {
        target["display_name"]: (
            build_acs1_dp05_summary(target_frames[target["display_name"]])
            if survey == "acs1"
            else build_blockgroup_summary(target_frames[target["display_name"]], summary_label_maps)
        )
        for target in targets
    }
    write_summary_sheet(writer, peer_summaries[primary_target["display_name"]])
    write_peer_comparison_sheet(writer, peer_summaries)
    write_data_dictionary_sheet(writer, data_dictionary_rows)

    writer.close()

    logger("Reading Excel workbook for merge...")
    all_sheets = pd.read_excel(output_xlsx, sheet_name=None, dtype=str)

    sheets_to_use = list(tables.values())
    missing = [s for s in sheets_to_use if s not in all_sheets]
    if missing:
        logger(f"WARN: missing sheets: {missing}")

    sheet_frames = {s: all_sheets[s] for s in sheets_to_use if s in all_sheets}
    cleaned = {name: clean_sheet(df, 7, name) for name, df in sheet_frames.items()}
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
    gdf = prepare_geometry_geoid(gdf, 7, "Place shapefile")
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
        county_name=primary_target["display_name"],
        state_fips=primary_target["state_fips"],
        county_fips=primary_target["place_fips"],
        year=year,
        geography="place",
    )


def run_place_trend_pipeline(
    city: str,
    state: str,
    years: list[int],
    output_dir: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    comparison_locations: Optional[list[dict]] = None,
    survey: str = "acs1",
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")
    if not years or len(set(years)) < 2:
        raise ValueError("Trend mode requires at least two years.")

    tables = tables or (ACS1_COUNTY_PLACE_TABLES if survey == "acs1" else BLOCKGROUP_TABLES)
    sorted_years = sorted({int(year) for year in years})
    targets = []
    for item in dedupe_locations(city, state, comparison_locations):
        location = item["location"]
        target_state = item["state"]
        state_fips, place_fips, place_name_raw = lookup_place_state(location, target_state)
        targets.append(
            {
                "query_name": location,
                "query_state": target_state,
                "state_fips": state_fips,
                "place_fips": place_fips,
                "name_raw": place_name_raw,
                "display_name": f"{place_name_raw}, {target_state}",
                "safe_name": normalize_name(place_name_raw),
            }
        )
    primary_target = targets[0]
    place_name = primary_target["safe_name"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = output_dir / f"{place_name}_ACS{survey.upper()}_Place_Trend_{sorted_years[0]}_{sorted_years[-1]}.xlsx"

    logger(
        f"Using place trend: {primary_target['name_raw']} in {primary_target['query_state']} "
        f"(state {primary_target['state_fips']}, place {primary_target['place_fips']})"
    )
    if len(targets) > 1:
        logger("Peer comparison places: " + ", ".join(t["display_name"] for t in targets[1:]))
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    combined_frames_by_sheet: dict[str, list[pd.DataFrame]] = {}
    yearly_summaries_by_target: dict[str, dict[int, dict]] = {
        target["display_name"]: {} for target in targets
    }
    data_dictionary_rows: list[dict] = []

    for year in sorted_years:
        logger(f"Building trend data for {year}")
        summary_frames_by_target: dict[str, dict[str, pd.DataFrame]] = {
            target["display_name"]: {} for target in targets
        }
        summary_label_maps: dict[str, dict[str, str]] = {}
        if survey == "acs1":
            table_configs = get_acs1_county_place_table_configs(year, tables)
            for config in table_configs:
                table_id = config["table_id"]
                sheet_name = config["sheet_name"]
                variables, labels = get_acs1_selected_variables_for_table(
                    year, table_id, config["dataset_path"], logger=logger
                )
                if not variables:
                    logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                    continue
                safe_labels = sanitize_field_names(labels)
                data_dictionary_rows.extend(
                    build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "place", survey, year)
                )
                summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
                for target in targets:
                    df = fetch_place_table(
                        year,
                        target["state_fips"],
                        target["place_fips"],
                        api_key,
                        table_id,
                        variables,
                        logger,
                        survey=survey,
                        dataset_path=config["dataset_path"],
                    )
                    if df.empty:
                        logger(f"WARN: no data returned for {table_id} ({target['display_name']}, {year})")
                        continue

                    df.rename(columns=safe_labels, inplace=True)
                    summary_frames_by_target[target["display_name"]][sheet_name] = df.copy()
                    out_df = prepare_trend_output_frame(
                        sheet_name,
                        df,
                        year,
                        location_value=target["display_name"],
                    )
                    combined_frames_by_sheet.setdefault(sheet_name, []).append(out_df)
        else:
            for table_id, sheet_name in tables.items():
                variables, labels = get_group_variables_and_labels(year, table_id, survey=survey)
                if not variables:
                    logger(f"WARN: table {table_id} is not available for {survey.upper()} {year}; skipping.")
                    continue
                safe_labels = sanitize_field_names(labels)
                data_dictionary_rows.extend(
                    build_data_dictionary_rows(table_id, sheet_name, labels, safe_labels, "place", survey, year)
                )
                summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
                for target in targets:
                    df = fetch_place_table(
                        year,
                        target["state_fips"],
                        target["place_fips"],
                        api_key,
                        table_id,
                        variables,
                        logger,
                        survey=survey,
                    )
                    if df.empty:
                        logger(f"WARN: no data returned for {table_id} ({target['display_name']}, {year})")
                        continue

                    df.rename(columns=safe_labels, inplace=True)
                    summary_frames_by_target[target["display_name"]][sheet_name] = df.copy()
                    out_df = prepare_trend_output_frame(
                        sheet_name,
                        df,
                        year,
                        location_value=target["display_name"],
                    )
                    combined_frames_by_sheet.setdefault(sheet_name, []).append(out_df)

        for target in targets:
            target_name = target["display_name"]
            yearly_summaries_by_target[target_name][year] = (
                build_acs1_dp05_summary(summary_frames_by_target[target_name])
                if survey == "acs1"
                else build_blockgroup_summary(summary_frames_by_target[target_name], summary_label_maps)
            )

    for sheet_name, frames in combined_frames_by_sheet.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(
                writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
            )

    primary_yearly_summaries = yearly_summaries_by_target[primary_target["display_name"]]
    write_summary_sheet(writer, primary_yearly_summaries[sorted_years[-1]])
    write_trend_sheet(writer, primary_yearly_summaries)
    for target in targets[1:]:
        target_name = target["display_name"]
        trend_sheet_name = sanitize_sheet_name(f"Trend - {target_name}")
        write_trend_sheet(writer, yearly_summaries_by_target[target_name], sheet_name=trend_sheet_name)
    write_data_dictionary_sheet(writer, data_dictionary_rows)
    writer.close()
    logger("Done.")

    return RunResult(
        output_xlsx=output_xlsx,
        output_gpkg=None,
        output_layer="",
        county_name=primary_target["name_raw"],
        state_fips=primary_target["state_fips"],
        county_fips=primary_target["place_fips"],
        year=sorted_years[-1],
        geography="place_trend",
    )


def run_tract_profile_pipeline(
    city: str,
    state: str,
    year: int,
    output_dir: Path,
    shapefile_path: Path,
    api_key: Optional[str] = None,
    groups: Optional[list[str]] = None,
    selected_profile_vars: Optional[dict[str, list[str]]] = None,
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
    data_dictionary_rows: list[dict] = []
    for group_id in groups:
        variables, labels = get_group_variables_and_labels(year, group_id, profile=True)
        if not variables:
            logger(f"WARN: table {group_id} is not available for ACS5 profile {year}; skipping.")
            continue

        # ✅ NEW: If the user picked specific variables in the UI, only request those
        if selected_profile_vars and group_id in selected_profile_vars:
            selected = set(selected_profile_vars[group_id] or [])
            variables = [v for v in variables if v in selected]
            labels = {k: v for k, v in labels.items() if k in variables}
            logger(f"{group_id}: using {len(variables)} selected variables")

        if not variables:
            logger(f"WARN: {group_id} has 0 variables after filtering; skipping")
            continue

        df = fetch_profile_group(
            year, state_fips, county_fips, api_key, group_id, variables, logger
        )
        if df.empty:
            logger(f"WARN: no data returned for {group_id}")
            continue

        safe_labels = sanitize_field_names(labels)
        data_dictionary_rows.extend(
            build_data_dictionary_rows(group_id, group_id, labels, safe_labels, "tracts", "acs5_profile", year)
        )
        df.rename(columns=safe_labels, inplace=True)
        df.to_excel(writer, sheet_name=sanitize_sheet_name(group_id), index=False)
    write_data_dictionary_sheet(writer, data_dictionary_rows)
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
    gdf = prepare_geometry_geoid(gdf, 11, "Tract shapefile")
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


def run_tract_pipeline(
    city: str,
    state: str,
    year: int,
    output_dir: Path,
    shapefile_path: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    geoids: Optional[list[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
    # [FIX-1] These parameters were on the overwritten first definition.
    # Accepted here and forwarded to run_tract_profile_pipeline so callers
    # (including the GUI Worker) can still use the profile/DP05 path.
    groups: Optional[list[str]] = None,
    selected_profile_vars: Optional[dict[str, list[str]]] = None,
) -> RunResult:
    # Delegate to the profile pipeline when the caller supplies DP05 groups.
    if groups is not None or selected_profile_vars is not None:
        return run_tract_profile_pipeline(
            city=city,
            state=state,
            year=year,
            output_dir=output_dir,
            shapefile_path=shapefile_path,
            api_key=api_key,
            groups=groups,
            selected_profile_vars=selected_profile_vars,
            logger=logger,
        )
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

    output_xlsx = output_dir / f"{county_name}_Tracts_{year}.xlsx"
    output_gpkg = output_dir / f"Merged_{county_name}_Tracts_{year}.gpkg"
    output_layer = f"tracts_acs_{year}"

    logger(f"Using county: {county_name_raw} (state {state_fips}, county {county_fips})")
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    summary_frames: dict[str, pd.DataFrame] = {}
    summary_label_maps: dict[str, dict[str, str]] = {}
    data_dictionary_rows: list[dict] = []

    for table_id, sheet_name in tables.items():
        effective_table_id = resolve_tract_table_id(table_id, year)
        variables, labels = get_group_variables_and_labels(year, effective_table_id, survey="acs5")
        if not variables:
            logger(acs5_table_availability_message(effective_table_id, year))
            continue

        df = fetch_tract_table(
            year, state_fips, county_fips, api_key, effective_table_id, variables, logger, survey="acs5"
        )
        if df.empty:
            logger(f"WARN: no data returned for {effective_table_id}")
            continue
        if geoids:
            df = df[df["GEOID"].isin(geoids)]
            if df.empty:
                logger(f"WARN: no data returned for {effective_table_id} after GEOID filtering")
                continue

        safe_labels = sanitize_field_names(labels)
        data_dictionary_rows.extend(
            build_data_dictionary_rows(effective_table_id, sheet_name, labels, safe_labels, "tracts", "acs5", year)
        )
        summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
        df.rename(columns=safe_labels, inplace=True)
        summary_frames[sheet_name] = df.copy()
        df.to_excel(writer, sheet_name=sanitize_sheet_name(sheet_name), index=False)

    write_summary_sheet(writer, build_blockgroup_summary(summary_frames, summary_label_maps))
    write_data_dictionary_sheet(writer, data_dictionary_rows)
    writer.close()

    logger("Reading Excel workbook for merge...")
    all_sheets = pd.read_excel(output_xlsx, sheet_name=None, dtype=str)
    sheets_to_use = [sanitize_sheet_name(name) for name in tables.values()]
    sheet_frames = {s: all_sheets[s] for s in sheets_to_use if s in all_sheets}
    if not sheet_frames:
        raise ValueError("No sheets found in the tract workbook.")

    cleaned = {name: clean_sheet(df, 11, name) for name, df in sheet_frames.items()}
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
    gdf = prepare_geometry_geoid(gdf, 11, "Tract shapefile")
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


def run_tract_trend_pipeline(
    city: str,
    state: str,
    years: list[int],
    output_dir: Path,
    api_key: Optional[str] = None,
    tables: Optional[dict] = None,
    geoids: Optional[list[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> RunResult:
    if logger is None:
        logger = print

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Set CENSUS_API_KEY in your environment or in the app.")
    if not years or len(set(years)) < 2:
        raise ValueError("Trend mode requires at least two years.")

    tables = tables or BLOCKGROUP_TABLES
    sorted_years = sorted({int(year) for year in years})
    state_fips, county_fips, county_name_raw = geocode_city_state(city, state)
    county_name = normalize_name(county_name_raw.replace(" County", ""))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_xlsx = output_dir / f"{county_name}_Tracts_Trend_{sorted_years[0]}_{sorted_years[-1]}.xlsx"

    logger(f"Using county tract trend: {county_name_raw} (state {state_fips}, county {county_fips})")
    logger(f"Writing Excel to: {output_xlsx}")

    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    combined_frames_by_sheet: dict[str, list[pd.DataFrame]] = {}
    yearly_summaries: dict[int, dict] = {}
    data_dictionary_rows: list[dict] = []

    for year in sorted_years:
        logger(f"Building tract trend data for {year}")
        summary_frames: dict[str, pd.DataFrame] = {}
        summary_label_maps: dict[str, dict[str, str]] = {}

        for table_id, sheet_name in tables.items():
            effective_table_id = resolve_tract_table_id(table_id, year)
            variables, labels = get_group_variables_and_labels(year, effective_table_id, survey="acs5")
            if not variables:
                logger(acs5_table_availability_message(table_id, year))
                continue

            df = fetch_tract_table(
                year, state_fips, county_fips, api_key, effective_table_id, variables, logger, survey="acs5"
            )
            if df.empty:
                logger(f"WARN: no data returned for {effective_table_id} ({year})")
                continue
            if geoids:
                df = df[df["GEOID"].isin(geoids)]
                if df.empty:
                    logger(f"WARN: no data returned for {effective_table_id} ({year}) after GEOID filtering")
                    continue

            safe_labels = sanitize_field_names(labels)
            data_dictionary_rows.extend(
                build_data_dictionary_rows(effective_table_id, sheet_name, labels, safe_labels, "tracts", "acs5", year)
            )
            summary_label_maps[sheet_name] = {safe_labels[var]: labels[var] for var in safe_labels}
            df.rename(columns=safe_labels, inplace=True)
            summary_frames[sheet_name] = df.copy()

            out_df = prepare_trend_output_frame(sheet_name, df, year)
            combined_frames_by_sheet.setdefault(sheet_name, []).append(out_df)

        yearly_summaries[year] = build_blockgroup_summary(summary_frames, summary_label_maps)

    for sheet_name, frames in combined_frames_by_sheet.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(
                writer, sheet_name=sanitize_sheet_name(sheet_name), index=False
            )

    if yearly_summaries:
        write_summary_sheet(writer, yearly_summaries[sorted_years[-1]])
        write_trend_sheet(writer, yearly_summaries)
    write_data_dictionary_sheet(writer, data_dictionary_rows)
    writer.close()
    logger("Done.")

    return RunResult(
        output_xlsx=output_xlsx,
        output_gpkg=None,
        output_layer="",
        county_name=county_name_raw,
        state_fips=state_fips,
        county_fips=county_fips,
        year=sorted_years[-1],
        geography="tracts_trend",
    )
