import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QCheckBox,
    QGroupBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)

from census_app_core import (
    ACS1_COUNTY_PLACE_TABLES,
    run_blockgroup_pipeline,
    run_blockgroup_trend_pipeline,
    run_county_pipeline,
    run_county_trend_pipeline,
    run_place_pipeline,
    run_place_trend_pipeline,
    run_tract_pipeline,
    run_tract_trend_pipeline,
    RunResult,
    STATE_ABBR_TO_NAME,
    BLOCKGROUP_TABLES,
    get_group_variables_and_labels,
)

def resource_path(name: str) -> Path:
    # PyInstaller one-file builds unpack bundled data under _MEIPASS.
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_dir / name


CONFIG_PATH = resource_path("config.json")
AVAILABLE_YEARS = [str(y) for y in range(2014, 2025)]


FIPS_PATH = resource_path("US_FIPS_Codes.xlsx")
PLACES_PATH = resource_path("Places_data.xlsx")


class ComparisonSelectionDialog(QDialog):
    def __init__(
        self,
        title: str,
        items_by_state: dict[str, list[str]],
        states: list[str],
        selected: list[dict] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 560)
        self.items_by_state = {
            str(state).strip(): sorted({str(item).strip() for item in items if str(item).strip()})
            for state, items in items_by_state.items()
            if str(state).strip()
        }
        self.states = [str(state).strip() for state in states if str(state).strip()]
        self.selected_entries = {
            (str(item.get("location", "")).strip(), str(item.get("state", "")).strip())
            for item in (selected or [])
            if str(item.get("location", "")).strip() and str(item.get("state", "")).strip()
        }

        layout = QVBoxLayout(self)
        self.state_combo = QComboBox()
        self.state_combo.addItems(self.states)
        self.state_combo.currentTextChanged.connect(self.populate_items)
        layout.addWidget(self.state_combo)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter locations...")
        self.filter_input.textChanged.connect(self.populate_items)
        layout.addWidget(self.filter_input)

        btn_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select visible")
        self.btn_select_all.clicked.connect(self.select_visible)
        self.btn_clear_all = QPushButton("Clear visible")
        self.btn_clear_all.clicked.connect(self.clear_visible)
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_clear_all)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self.on_item_changed)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.populate_items()

    def populate_items(self, _text: str = ""):
        query = self.filter_input.text().strip().casefold()
        current_state = self.state_combo.currentText().strip()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for name in self.items_by_state.get(current_state, []):
            if query and query not in name.casefold():
                continue
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, {"location": name, "state": current_state})
            item.setCheckState(
                Qt.Checked if (name, current_state) in self.selected_entries else Qt.Unchecked
            )
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def on_item_changed(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole) or {}
        key = (str(data.get("location", "")).strip(), str(data.get("state", "")).strip())
        if item.checkState() == Qt.Checked:
            self.selected_entries.add(key)
        else:
            self.selected_entries.discard(key)

    def select_visible(self):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)
            data = item.data(Qt.UserRole) or {}
            key = (str(data.get("location", "")).strip(), str(data.get("state", "")).strip())
            self.selected_entries.add(key)
        self.list_widget.blockSignals(False)

    def clear_visible(self):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)
            data = item.data(Qt.UserRole) or {}
            key = (str(data.get("location", "")).strip(), str(data.get("state", "")).strip())
            self.selected_entries.discard(key)
        self.list_widget.blockSignals(False)

    def selected_items(self) -> list[dict]:
        items = [{"location": location, "state": state} for location, state in self.selected_entries]
        return sorted(items, key=lambda item: (item["state"], item["location"]))


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def load_counties_and_states() -> tuple[list[str], list[str]]:
    if not FIPS_PATH.exists():
        return [], []
    try:
        df = pd.read_excel(FIPS_PATH)
        states = (
            df["State"]
            .astype(str)
            .str.strip()
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        counties = (
            df["County Name"]
            .astype(str)
            .str.strip()
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        return counties, states
    except Exception:
        return [], []


def load_counties_by_state() -> dict[str, list[str]]:
    if not FIPS_PATH.exists():
        return {}
    try:
        df = pd.read_excel(FIPS_PATH)
        df = df[["State", "County Name"]].dropna().copy()
        df["State"] = df["State"].astype(str).str.strip()
        df["County Name"] = df["County Name"].astype(str).str.strip()
        result = {}
        for state_name, group in df.groupby("State"):
            result[state_name] = sorted(group["County Name"].drop_duplicates().tolist())
        return result
    except Exception:
        return {}


def load_places_and_states() -> tuple[list[str], list[str]]:
    if not PLACES_PATH.exists():
        return [], []
    try:
        df = pd.read_excel(PLACES_PATH)
        df = df[df["PLACE"].fillna(0).astype(int) > 0].copy()
        states = (
            df["STNAME"]
            .astype(str)
            .str.strip()
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        places = (
            df["NAME"]
            .astype(str)
            .str.strip()
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        return places, states
    except Exception:
        return [], []


def load_places_by_state() -> dict[str, list[str]]:
    if not PLACES_PATH.exists():
        return {}
    try:
        df = pd.read_excel(PLACES_PATH)
        df = df[df["PLACE"].fillna(0).astype(int) > 0].copy()
        df = df[["STNAME", "NAME"]].dropna()
        df["STNAME"] = df["STNAME"].astype(str).str.strip()
        df["NAME"] = df["NAME"].astype(str).str.strip()
        result = {}
        for state_name, group in df.groupby("STNAME"):
            result[state_name] = sorted(group["NAME"].drop_duplicates().tolist())
        return result
    except Exception:
        return {}


def load_state_fips_map() -> dict[str, str]:
    if not FIPS_PATH.exists():
        return {}
    try:
        df = pd.read_excel(FIPS_PATH)
        df = df[["State", "FIPS State"]].dropna()
        df["State"] = df["State"].astype(str).str.strip()
        df["FIPS State"] = df["FIPS State"].astype(str).str.zfill(2)
        return dict(df.drop_duplicates("State")[["State", "FIPS State"]].values.tolist())
    except Exception:
        return {}


def normalize_state_name(state: str) -> str:
    if not state:
        return ""
    s = state.strip()
    if len(s) == 2:
        return STATE_ABBR_TO_NAME.get(s.upper(), s)
    return s


def get_cache_dir() -> Path:
    base = os.getenv("LOCALAPPDATA")
    if base:
        return Path(base) / "CensusTool" / "tiger"
    return Path.home() / ".census_tool" / "tiger"


def ensure_tract_shapefile(state: str, year: int, logger, progress_cb=None) -> Path | None:
    state_name = normalize_state_name(state)
    state_fips_map = load_state_fips_map()
    state_fips = state_fips_map.get(state_name)
    if not state_fips:
        logger("ERROR: Could not resolve state to FIPS. Please enter a valid state.")
        return None

    cache_dir = get_cache_dir() / f"TIGER{year}" / "TRACT"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if year == 2010:
        shp_name = f"tl_2010_{state_fips}_tract10.shp"
    else:
        shp_name = f"tl_{year}_{state_fips}_tract.shp"
    shp_path = cache_dir / shp_name
    if shp_path.exists():
        return shp_path

    if year == 2010:
        zip_name = f"tl_2010_{state_fips}_tract10.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/{zip_name}"
    else:
        zip_name = f"tl_{year}_{state_fips}_tract.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/{zip_name}"
    zip_path = cache_dir / zip_name

    logger(f"Downloading tract shapefile for {state_name} ({state_fips})...")
    if not download_file(url, zip_path, logger, progress_cb):
        return None

    logger("Extracting shapefile...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    if not shp_path.exists():
        logger("ERROR: extracted shapefile not found.")
        return None

    return shp_path


def auto_download_tract(logger, state: str, year: int) -> Path | None:
    try:
        return ensure_tract_shapefile(state, year, logger)
    except Exception as exc:
        logger(f"ERROR: failed to download tract shapefile: {exc}")
        return None


def ensure_blockgroup_shapefile(state: str, year: int, logger, progress_cb=None) -> Path | None:
    state_name = normalize_state_name(state)
    state_fips_map = load_state_fips_map()
    state_fips = state_fips_map.get(state_name)
    if not state_fips:
        logger("ERROR: Could not resolve state to FIPS. Please enter a valid state.")
        return None

    cache_dir = get_cache_dir() / f"TIGER{year}" / "BG"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if year == 2010:
        shp_name = f"tl_2010_{state_fips}_bg10.shp"
    else:
        shp_name = f"tl_{year}_{state_fips}_bg.shp"
    shp_path = cache_dir / shp_name
    if shp_path.exists():
        return shp_path

    if year == 2010:
        zip_name = f"tl_2010_{state_fips}_bg10.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER2010/BG/2010/{zip_name}"
    else:
        zip_name = f"tl_{year}_{state_fips}_bg.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/BG/{zip_name}"
    zip_path = cache_dir / zip_name

    logger(f"Downloading block group shapefile for {state_name} ({state_fips})...")
    if not download_file(url, zip_path, logger, progress_cb):
        return None

    logger("Extracting shapefile...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    if not shp_path.exists():
        logger("ERROR: extracted shapefile not found.")
        return None

    return shp_path


def auto_download_blockgroups(logger, state: str, year: int) -> Path | None:
    try:
        return ensure_blockgroup_shapefile(state, year, logger)
    except Exception as exc:
        logger(f"ERROR: failed to download block group shapefile: {exc}")
        return None


def ensure_place_shapefile(state: str, year: int, logger, progress_cb=None) -> Path | None:
    state_name = normalize_state_name(state)
    state_fips_map = load_state_fips_map()
    state_fips = state_fips_map.get(state_name)
    if not state_fips:
        logger("ERROR: Could not resolve state to FIPS. Please enter a valid state.")
        return None

    cache_dir = get_cache_dir() / f"TIGER{year}" / "PLACE"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if year == 2010:
        shp_name = f"tl_2010_{state_fips}_place10.shp"
    else:
        shp_name = f"tl_{year}_{state_fips}_place.shp"
    shp_path = cache_dir / shp_name
    if shp_path.exists():
        return shp_path

    if year == 2010:
        zip_name = f"tl_2010_{state_fips}_place10.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER2010/PLACE/2010/{zip_name}"
    else:
        zip_name = f"tl_{year}_{state_fips}_place.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/PLACE/{zip_name}"
    zip_path = cache_dir / zip_name

    logger(f"Downloading place shapefile for {state_name} ({state_fips})...")
    if not download_file(url, zip_path, logger, progress_cb):
        return None

    logger("Extracting shapefile...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    if not shp_path.exists():
        logger("ERROR: extracted shapefile not found.")
        return None

    return shp_path


def download_file(url: str, dest: Path, logger, progress_cb=None) -> bool:
    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        logger(f"ERROR: download failed ({resp.status_code}).")
        return False
    total = int(resp.headers.get("Content-Length", "0"))
    downloaded = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            if total and progress_cb:
                downloaded += len(chunk)
                pct = min(100, int(downloaded * 100 / total))
                progress_cb(pct)
    if progress_cb:
        progress_cb(100)
    return True


class DownloadWorker(QObject):
    progress = Signal(int)
    done = Signal(object)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, state: str, kind: str, year: int):
        super().__init__()
        self.state = state
        self.kind = kind
        self.year = year

    @Slot()
    def run(self):
        try:
            def logger(msg: str):
                self.log.emit(msg)

            progress_cb = lambda pct: self.progress.emit(pct)
            if self.kind == "block_groups":
                path = ensure_blockgroup_shapefile(self.state, self.year, logger, progress_cb)
            elif self.kind == "place":
                path = ensure_place_shapefile(self.state, self.year, logger, progress_cb)
            else:
                path = ensure_tract_shapefile(self.state, self.year, logger, progress_cb)
            if not path:
                self.error.emit("Download failed.")
                return
            self.done.emit(path)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class Worker(QObject):
    log = Signal(str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    @Slot()
    def run(self):
        try:
            def logger(msg: str):
                self.log.emit(msg)

            params = dict(self.params)
            geography = params.pop("geography", "block_groups")
            survey = params.pop("survey", "acs5")
            trend_mode = params.pop("trend_mode", False)
            if trend_mode and geography == "county":
                res = run_county_trend_pipeline(logger=logger, survey=survey, **params)
            elif trend_mode and geography == "place":
                res = run_place_trend_pipeline(logger=logger, survey=survey, **params)
            elif trend_mode and geography == "block_groups":
                res = run_blockgroup_trend_pipeline(logger=logger, **params)
            elif trend_mode and geography == "tracts":
                res = run_tract_trend_pipeline(logger=logger, **params)
            elif geography == "tracts":
                res = run_tract_pipeline(logger=logger, **params)
            elif geography == "county":
                res = run_county_pipeline(logger=logger, survey=survey, **params)
            elif geography == "place":
                res = run_place_pipeline(logger=logger, survey=survey, **params)
            else:
                res = run_blockgroup_pipeline(logger=logger, **params)
            self.done.emit(res)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TYLin Census Tool")

        self.thread = None
        self.worker = None
        self.download_thread = None
        self.download_worker = None
        self.pending_params = None
        self.pending_runs = []
        self.pending_downloads = []
        self.download_continue = False
        self.comparison_locations = []
        self.cancel_requested = False
        self.counties, self.states = load_counties_and_states()
        self.counties_by_state = load_counties_by_state()
        self.places, self.place_states = load_places_and_states()
        self.places_by_state = load_places_by_state()
        self.all_states = sorted(set(self.states) | set(self.place_states))

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        geo_box = QGroupBox("Geography")
        geo_layout = QHBoxLayout(geo_box)
        self.geo_combo = QComboBox()
        self.geo_combo.addItems(["Block groups", "Tracts"])
        self.geo_combo.currentIndexChanged.connect(self.update_geo_state)
        geo_layout.addWidget(QLabel("Output level:"))
        geo_layout.addWidget(self.geo_combo)
        self.btn_download = QPushButton("Download shapefile")
        self.btn_download.clicked.connect(self.download_clicked)
        geo_layout.addWidget(self.btn_download)
        geo_layout.addStretch(1)
        layout.addWidget(geo_box)

        # ---- Inputs ----
        input_box = QGroupBox("Inputs")
        input_layout = QGridLayout(input_box)

        self.county_combo = QComboBox()
        self.county_combo.setEditable(True)
        self.county_combo.setInsertPolicy(QComboBox.NoInsert)
        self.county_combo.currentTextChanged.connect(self.refresh_comparison_list)
        self.state_combo = QComboBox()
        self.state_combo.setEditable(True)
        self.state_combo.setInsertPolicy(QComboBox.NoInsert)
        self.state_combo.currentTextChanged.connect(self.on_state_changed)

        self.year_list = QListWidget()
        self.year_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for year in AVAILABLE_YEARS:
            self.year_list.addItem(QListWidgetItem(year))
        default_item = self.year_list.findItems("2024", Qt.MatchFixedString)
        if default_item:
            default_item[0].setSelected(True)
        self.year_list.setMinimumWidth(220)
        self.year_list.setMinimumHeight(180)
        self.year_list.setMaximumHeight(220)

        input_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["ACS 5-Year", "ACS 1-Year"])
        self.dataset_combo.currentIndexChanged.connect(self.update_geo_state)
        input_layout.addWidget(self.dataset_combo, 0, 1)
        self.trend_mode_check = QCheckBox("Combine selected years into one trend workbook")
        self.trend_mode_check.setChecked(True)
        input_layout.addWidget(self.trend_mode_check, 0, 2, 1, 2)

        input_layout.addWidget(QLabel("State:"), 1, 0)
        input_layout.addWidget(self.state_combo, 1, 1, 1, 3)

        self.location_label = QLabel("County:")
        input_layout.addWidget(self.location_label, 2, 0)
        input_layout.addWidget(self.county_combo, 2, 1, 1, 3)

        self.geoid_filter_label = QLabel("Filter GEOIDs:")
        self.geoid_filter_input = QLineEdit()
        self.geoid_filter_input.setPlaceholderText("Comma-separated GEOID(s)")
        self.geoid_filter_label.setVisible(False)
        self.geoid_filter_input.setVisible(False)
        input_layout.addWidget(self.geoid_filter_label, 3, 0)
        input_layout.addWidget(self.geoid_filter_input, 3, 1, 1, 3)

        input_layout.addWidget(QLabel("ACS year(s):"), 4, 0)
        year_row = QHBoxLayout()
        year_row.addWidget(self.year_list)
        year_btn_col = QVBoxLayout()
        self.btn_years_all = QPushButton("Select all years")
        self.btn_years_all.clicked.connect(self.select_all_years)
        self.btn_years_none = QPushButton("Clear years")
        self.btn_years_none.clicked.connect(self.clear_selected_years)
        year_btn_col.addWidget(self.btn_years_all)
        year_btn_col.addWidget(self.btn_years_none)
        year_btn_col.addStretch(1)
        year_row.addLayout(year_btn_col)
        year_row.addStretch(1)
        input_layout.addLayout(year_row, 3, 1, 1, 3)

        self.compare_box = QGroupBox("Peer Comparison")
        compare_layout = QVBoxLayout(self.compare_box)
        compare_btn_row = QHBoxLayout()
        self.btn_select_comparisons = QPushButton("Select comparison locations")
        self.btn_select_comparisons.clicked.connect(self.select_comparison_locations)
        self.btn_clear_comparisons = QPushButton("Clear")
        self.btn_clear_comparisons.clicked.connect(self.clear_comparison_locations)
        compare_btn_row.addWidget(self.btn_select_comparisons)
        compare_btn_row.addWidget(self.btn_clear_comparisons)
        compare_btn_row.addStretch(1)
        compare_layout.addLayout(compare_btn_row)
        self.comparison_list = QListWidget()
        self.comparison_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.comparison_list.setMinimumHeight(100)
        compare_layout.addWidget(self.comparison_list)
        input_layout.addWidget(self.compare_box, 4, 0, 1, 4)

        layout.addWidget(input_box)

        # ---- Tables ----
        self.bg_tables_box = QGroupBox("Tables")
        bg_layout = QVBoxLayout(self.bg_tables_box)
        self.bg_tables_list = QListWidget()
        self.bg_tables_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for table_id, label in BLOCKGROUP_TABLES.items():
            item = QListWidgetItem(f"{table_id} - {label}")
            item.setData(Qt.UserRole, table_id)
            self.bg_tables_list.addItem(item)
        bg_layout.addWidget(self.bg_tables_list)
        bg_btn_row = QHBoxLayout()
        self.btn_bg_all = QPushButton("Select all")
        self.btn_bg_all.clicked.connect(self.select_all_bg_tables)
        self.btn_bg_none = QPushButton("Select none")
        self.btn_bg_none.clicked.connect(self.select_none_bg_tables)
        bg_btn_row.addWidget(self.btn_bg_all)
        bg_btn_row.addWidget(self.btn_bg_none)
        bg_btn_row.addStretch(1)
        bg_layout.addLayout(bg_btn_row)
        layout.addWidget(self.bg_tables_box)

        # ---- DP05 Fields (Tracts) ----
        self.dp05_box = QGroupBox("DP05 Fields (Tracts)")
        dp05_layout = QVBoxLayout(self.dp05_box)
        dp05_btn_row = QHBoxLayout()
        self.btn_dp05_load = QPushButton("Load DP05 fields")
        self.btn_dp05_load.clicked.connect(self.load_dp05_fields)
        dp05_btn_row.addWidget(self.btn_dp05_load)
        self.btn_dp05_all = QPushButton("Select all")
        self.btn_dp05_all.clicked.connect(self.select_all_dp05_vars)
        dp05_btn_row.addWidget(self.btn_dp05_all)
        self.btn_dp05_none = QPushButton("Select none")
        self.btn_dp05_none.clicked.connect(self.select_none_dp05_vars)
        dp05_btn_row.addWidget(self.btn_dp05_none)
        dp05_btn_row.addStretch(1)
        dp05_layout.addLayout(dp05_btn_row)
        self.dp05_list = QListWidget()
        self.dp05_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dp05_list.setMinimumHeight(320)
        dp05_layout.addWidget(self.dp05_list)
        layout.addWidget(self.dp05_box)

        # ---- Paths ----
        path_box = QGroupBox("Paths")
        path_layout = QGridLayout(path_box)

        self.output_dir = QLineEdit(str(Path.cwd()))
        self.btn_output = QPushButton("Browse...")
        self.btn_output.clicked.connect(self.pick_output_dir)

        path_layout.addWidget(QLabel("Output folder:"), 0, 0)
        path_layout.addWidget(self.output_dir, 0, 1)
        path_layout.addWidget(self.btn_output, 0, 2)

        layout.addWidget(path_box)

        # ---- API Key ----
        key_box = QGroupBox("API Key (optional)")
        key_layout = QHBoxLayout(key_box)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(QLabel("Census API key:"))
        key_layout.addWidget(self.api_key)
        layout.addWidget(key_box)

        cfg = load_config()
        cfg_key = cfg.get("CENSUS_API_KEY")
        if cfg_key:
            self.api_key.setText(cfg_key)
            key_box.setVisible(False)

        self.populate_state_options()
        self.update_geo_state()

        # ---- Run + Log ----
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.run_clicked)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_active_work)
        self.btn_cancel.setEnabled(False)
        self.btn_open_out = QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(self.open_output_folder)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.btn_open_out)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.analysis_status_label = QLabel("Analysis status: Idle")
        layout.addWidget(self.analysis_status_label)
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        self.download_progress_label = QLabel("Download: idle")
        layout.addWidget(self.download_progress_label)
        layout.addWidget(self.download_progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)

    def append_log(self, msg: str):
        self.log.append(msg)

    def set_analysis_status(self, status: str):
        self.analysis_status_label.setText(f"Analysis status: {status}")

    def set_busy_state(self, busy: bool):
        self.btn_cancel.setEnabled(busy)
        if busy:
            self.btn_run.setEnabled(False)
            self.btn_download.setEnabled(False)
        else:
            self.btn_run.setEnabled(True)
            self.btn_download.setEnabled(True)

    def cancel_active_work(self):
        self.cancel_requested = True
        self.pending_runs = []
        self.pending_downloads = []
        self.pending_params = None
        self.download_continue = False
        self.append_log("Cancellation requested.")
        self.set_analysis_status("Cancelled")
        self.download_progress_label.setText("Download cancelled")
        self.download_progress.setValue(0)

        if self.thread and isValid(self.thread) and self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.quit()
            if not self.thread.wait(2000):
                self.thread.terminate()
                self.thread.wait(2000)
        if self.download_thread and isValid(self.download_thread) and self.download_thread.isRunning():
            self.download_thread.requestInterruption()
            self.download_thread.quit()
            if not self.download_thread.wait(2000):
                self.download_thread.terminate()
                self.download_thread.wait(2000)
        self.set_busy_state(False)

    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_dir.setText(d)

    def update_geo_state(self):
        current_geo = self.geo_combo.currentText()
        if self.dataset_combo.currentText() == "ACS 1-Year":
            allowed = ["County", "Places"]
        else:
            allowed = ["Places", "Tracts", "Block groups"]
        if [self.geo_combo.itemText(i) for i in range(self.geo_combo.count())] != allowed:
            self.geo_combo.blockSignals(True)
            self.geo_combo.clear()
            self.geo_combo.addItems(allowed)
            self.geo_combo.blockSignals(False)
            if current_geo in allowed:
                self.geo_combo.setCurrentText(current_geo)
        geo_text = self.geo_combo.currentText()
        is_block = geo_text == "Block groups"
        is_county = geo_text == "County"
        is_place = geo_text == "Places"
        is_tract = geo_text == "Tracts"
        self.bg_tables_box.setVisible(is_block or is_county or is_place or is_tract)
        self.bg_tables_box.setTitle("Tables")
        self.dp05_box.setVisible(False)
        self.compare_box.setVisible(is_county or is_place)
        self.trend_mode_check.setVisible(is_block or is_county or is_place or is_tract)
        self.btn_download.setEnabled(geo_text in {"Block groups", "Tracts", "Places"})
        self.location_label.setText("Place:" if is_place else "County:")
        self.geoid_filter_label.setVisible(is_block or is_tract)
        self.geoid_filter_input.setVisible(is_block or is_tract)
        self.populate_state_options()
        self.populate_location_options()
        if not (is_county or is_place):
            self.comparison_locations = []
        self.refresh_comparison_list()
        self.populate_table_options()

        for i in range(self.year_list.count()):
            item = self.year_list.item(i)
            if item.text() == "2020":
                if self.dataset_combo.currentText() == "ACS 1-Year":
                    item.setSelected(False)
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                else:
                    item.setFlags(item.flags() | Qt.ItemIsEnabled)
                break

    def available_states_for_current_geo(self) -> list[str]:
        if self.geo_combo.currentText() == "Places":
            return self.place_states
        return self.all_states

    def available_locations_for_state(self, state_text: str) -> list[str]:
        state_name = normalize_state_name(state_text)
        if not state_name:
            return []
        if self.geo_combo.currentText() == "Places":
            return self.places_by_state.get(state_name, [])
        return self.counties_by_state.get(state_name, [])

    def populate_state_options(self):
        if not hasattr(self, "state_combo"):
            return
        current_state = normalize_state_name(self.state_combo.currentText())
        available_states = self.available_states_for_current_geo()
        self.state_combo.blockSignals(True)
        self.state_combo.clear()
        self.state_combo.addItem("")
        if available_states:
            self.state_combo.addItems(available_states)
        if current_state:
            idx = self.state_combo.findText(current_state, Qt.MatchFixedString)
            if idx >= 0:
                self.state_combo.setCurrentIndex(idx)
            else:
                self.state_combo.setEditText(current_state)
        self.state_combo.blockSignals(False)

    def populate_location_options(self):
        if not hasattr(self, "county_combo"):
            return
        current_text = self.county_combo.currentText().strip()
        items = self.available_locations_for_state(self.state_combo.currentText())
        self.county_combo.blockSignals(True)
        self.county_combo.clear()
        self.county_combo.addItem("")
        if items:
            self.county_combo.addItems(items)
        if current_text:
            idx = self.county_combo.findText(current_text, Qt.MatchFixedString)
            if idx >= 0:
                self.county_combo.setCurrentIndex(idx)
            elif normalize_state_name(self.state_combo.currentText()):
                self.county_combo.setEditText("")
            else:
                self.county_combo.setEditText(current_text)
        self.county_combo.blockSignals(False)
        self.refresh_comparison_list()

    def on_state_changed(self, _text: str):
        self.populate_location_options()

    def selected_years(self) -> list[int]:
        items = self.year_list.selectedItems()
        if not items:
            return []
        years = sorted({int(item.text()) for item in items})
        if self.dataset_combo.currentText() == "ACS 1-Year":
            years = [year for year in years if year != 2020]
        return years

    def select_all_years(self):
        for i in range(self.year_list.count()):
            item = self.year_list.item(i)
            if item.flags() & Qt.ItemIsEnabled:
                item.setSelected(True)

    def clear_selected_years(self):
        self.year_list.clearSelection()

    def selected_bg_tables(self) -> dict:
        tables = {}
        table_options = self.current_table_options()
        for item in self.bg_tables_list.selectedItems():
            table_id = item.data(Qt.UserRole)
            if table_id:
                tables[table_id] = table_options.get(table_id, table_id)
        return tables

    def current_table_options(self) -> dict:
        if self.dataset_combo.currentText() == "ACS 1-Year" and self.geo_combo.currentText() in {"County", "Places"}:
            return ACS1_COUNTY_PLACE_TABLES
        return BLOCKGROUP_TABLES

    def populate_table_options(self):
        if not hasattr(self, "bg_tables_list"):
            return
        current_options = self.current_table_options()
        existing_ids = []
        for i in range(self.bg_tables_list.count()):
            item = self.bg_tables_list.item(i)
            table_id = item.data(Qt.UserRole)
            if table_id:
                existing_ids.append(table_id)
        if existing_ids == list(current_options.keys()):
            return
        self.bg_tables_list.clear()
        for table_id, label in current_options.items():
            item = QListWidgetItem(f"{table_id} - {label}")
            item.setData(Qt.UserRole, table_id)
            self.bg_tables_list.addItem(item)
            item.setSelected(True)

    def select_all_bg_tables(self):
        for i in range(self.bg_tables_list.count()):
            self.bg_tables_list.item(i).setSelected(True)

    def select_none_bg_tables(self):
        self.bg_tables_list.clearSelection()

    def tidy_dp05_label(self, label: str) -> str:
        if not label:
            return ""
        tidy = label
        if tidy.startswith("Estimate!!"):
            tidy = tidy[len("Estimate!!") :]
        tidy = tidy.replace("!!", " > ").strip()
        return tidy

    def dp05_keep_label(self, label: str) -> bool:
        if not label:
            return False
        label_upper = label.upper()
        sex_age_prefix = "ESTIMATE!!SEX AND AGE!!TOTAL POPULATION"
        if label_upper.startswith(sex_age_prefix):
            return True
        if "HISPANIC" in label_upper:
            return True
        if "!!RACE!!" in label_upper:
            return False
        return True

    def load_dp05_fields(self):
        if self.geo_combo.currentText() != "Tracts":
            QMessageBox.information(self, "DP05 fields", "DP05 fields are for Tracts only.")
            return
        years = self.selected_years()
        if len(years) != 1:
            QMessageBox.warning(
                self,
                "Select one year",
                "Please select exactly one year to load DP05 fields.",
            )
            return
        year = years[0]
        try:
            variables, labels = get_group_variables_and_labels(year, "DP05", profile=True)
        except Exception as exc:
            QMessageBox.critical(self, "DP05 fields", f"Failed to load DP05 fields: {exc}")
            return
        if not variables:
            QMessageBox.warning(self, "DP05 fields", f"No DP05 fields found for {year}.")
            return
        self.dp05_list.clear()
        sex_age_items = []
        hispanic_items = []
        other_items = []
        for var_id in variables:
            if not str(var_id).endswith("E") or str(var_id).endswith("PE"):
                continue
            label = labels.get(var_id, var_id)
            if not self.dp05_keep_label(label):
                continue
            display = self.tidy_dp05_label(label)
            item = QListWidgetItem(display or label)
            item.setData(Qt.UserRole, var_id)
            item.setFlags(item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            label_upper = label.upper()
            if "SEX AND AGE" in label_upper:
                sex_age_items.append(item)
            elif "HISPANIC" in label_upper:
                hispanic_items.append(item)
            else:
                other_items.append(item)

        def add_section(title: str, items: list[QListWidgetItem]):
            if not items:
                return
            header = QListWidgetItem(title)
            header.setFlags(Qt.ItemIsEnabled)
            self.dp05_list.addItem(header)
            for it in items:
                self.dp05_list.addItem(it)

        add_section("Sex and Age", sex_age_items)
        add_section("Hispanic or Latino", hispanic_items)
        add_section("Other", other_items)

    def selected_dp05_vars(self) -> list[str]:
        vars_selected = []
        for item in self.dp05_list.selectedItems():
            var_id = item.data(Qt.UserRole)
            if isinstance(var_id, str) and var_id:
                vars_selected.append(var_id)
        return vars_selected

    def select_all_dp05_vars(self):
        for i in range(self.dp05_list.count()):
            item = self.dp05_list.item(i)
            if item.flags() & Qt.ItemIsSelectable:
                item.setSelected(True)

    def select_none_dp05_vars(self):
        self.dp05_list.clearSelection()

    def available_comparison_items_by_state(self) -> tuple[dict[str, list[str]], list[str]]:
        if self.geo_combo.currentText() == "Places":
            return self.places_by_state, self.place_states
        return self.counties_by_state, self.states

    def refresh_comparison_list(self):
        if not hasattr(self, "comparison_list"):
            return
        self.comparison_list.clear()
        current_primary = self.county_combo.currentText().strip()
        current_state = self.state_combo.currentText().strip()
        filtered = []
        seen = set()
        for entry in self.comparison_locations:
            clean_name = str(entry.get("location", "")).strip()
            clean_state = str(entry.get("state", "")).strip()
            key = (clean_name.casefold(), clean_state.casefold())
            if not clean_name or not clean_state or key in seen:
                continue
            if clean_name == current_primary and clean_state == current_state:
                continue
            seen.add(key)
            filtered.append({"location": clean_name, "state": clean_state})
        self.comparison_locations = filtered
        for entry in self.comparison_locations:
            self.comparison_list.addItem(
                QListWidgetItem(f"{entry['location']}, {entry['state']}")
            )
        has_items = bool(self.comparison_locations)
        self.btn_clear_comparisons.setEnabled(has_items)

    def select_comparison_locations(self):
        geo_text = self.geo_combo.currentText()
        if geo_text not in {"County", "Places"}:
            QMessageBox.information(
                self,
                "Peer comparison",
                "Comparison locations are available for County and Places only.",
            )
            return
        items_by_state, states = self.available_comparison_items_by_state()
        dialog = ComparisonSelectionDialog(
            f"Select comparison {'places' if geo_text == 'Places' else 'counties'}",
            items_by_state,
            states,
            self.comparison_locations,
            self,
        )
        if dialog.exec() == QDialog.Accepted:
            self.comparison_locations = dialog.selected_items()
            self.refresh_comparison_list()

    def clear_comparison_locations(self):
        self.comparison_locations = []
        self.refresh_comparison_list()

    def start_next_run(self):
        if not self.pending_runs:
            self.set_busy_state(False)
            return
        params = self.pending_runs.pop(0)
        self.pending_params = params
        if params["geography"] == "county" or params.get("trend_mode"):
            self.pending_params = None
            self.start_pipeline(params)
            return
        kind_map = {
            "block_groups": "block_groups",
            "place": "place",
            "tracts": "tracts",
        }
        kind = kind_map.get(params["geography"], "tracts")
        self.append_log(f"Starting shapefile download for {params['year']}...")
        self.start_download(kind, params["state"], params["year"])

    def start_next_download(self):
        if not self.pending_downloads:
            self.set_busy_state(False)
            return
        next_job = self.pending_downloads.pop(0)
        self.append_log(f"Starting shapefile download for {next_job['year']}...")
        self.start_download(next_job["kind"], next_job["state"], next_job["year"])

    # year selection uses a dropdown

    def open_output_folder(self):
        d = self.output_dir.text().strip()
        if not d:
            return
        if os.path.isdir(d):
            if sys.platform.startswith("win"):
                os.startfile(d)
            elif sys.platform == "darwin":
                os.system(f'open "{d}"')
            else:
                os.system(f'xdg-open "{d}"')

    def run_clicked(self):
        city = self.county_combo.currentText().strip()
        state = self.state_combo.currentText().strip()
        years = self.selected_years()
        out_dir = self.output_dir.text().strip()
        geography_text = self.geo_combo.currentText()
        is_block = geography_text == "Block groups"
        is_county = geography_text == "County"
        is_place = geography_text == "Places"
        api_key = self.api_key.text().strip() or None
        survey = "acs1" if self.dataset_combo.currentText() == "ACS 1-Year" else "acs5"
        tables = None

        if not city or not state:
            QMessageBox.warning(self, "Missing input", "Please enter county and state.")
            return
        if not years:
            QMessageBox.warning(self, "Missing input", "Please select at least one year.")
            return
        raw_selected_years = sorted({int(item.text()) for item in self.year_list.selectedItems()})
        if survey == "acs1" and 2020 in raw_selected_years:
            QMessageBox.information(
                self,
                "ACS 1-Year 2020 unavailable",
                "2020 standard ACS 1-Year estimates are not available in the Census API and will be skipped.",
            )
            years = [year for year in years if year != 2020]
            if not years:
                self.set_busy_state(False)
                return
        geoids: list[str] = []
        if geography_text in {"Block groups", "Tracts"}:
            raw_geoids = self.geoid_filter_input.text().strip()
            if raw_geoids:
                geoids = [token.strip() for token in re.split(r"[\s,;]+", raw_geoids) if token.strip()]
                invalid = [g for g in geoids if not g.isdigit() or len(g) not in {11, 12}]
                if invalid:
                    QMessageBox.warning(
                        self,
                        "Invalid GEOID",
                        "Enter only 11- or 12-digit numeric GEOIDs separated by commas, spaces, or semicolons.",
                    )
                    return
                if geography_text == "Tracts":
                    geoids = [g[:11] if len(g) == 12 else g for g in geoids]
                else:
                    geoids = [g for g in geoids]
        if is_block or is_county or is_place or geography_text == "Tracts":
            tables = self.selected_bg_tables()
            if not tables:
                QMessageBox.warning(
                    self, "Missing input", "Please select at least one table."
                )
                return
        if not out_dir:
            QMessageBox.warning(self, "Missing output folder", "Choose an output folder.")
            return
        self.cancel_requested = False
        self.set_analysis_status("Running")
        self.set_busy_state(True)
        self.pending_runs = []
        trend_mode = self.trend_mode_check.isChecked() and (is_block or is_county or is_place or geography_text == "Tracts") and len(years) > 1
        if trend_mode:
            self.append_log(
                f"{self.dataset_combo.currentText()} trend mode detected: combining selected years into one workbook."
            )
            params = dict(
                city=city,
                state=state,
                years=years,
                output_dir=Path(out_dir),
                api_key=api_key,
                geography="place" if is_place else ("county" if is_county else ("block_groups" if is_block else "tracts")),
                survey=survey,
                trend_mode=True,
                tables=tables,
            )
            if is_block or geography_text == "Tracts":
                params["geoids"] = geoids or None
            if is_county or is_place:
                comparison_locations = [
                    entry
                    for entry in self.comparison_locations
                    if entry.get("location") and entry.get("state")
                    and not (
                        str(entry.get("location", "")).strip() == city
                        and str(entry.get("state", "")).strip() == state
                    )
                ]
                params["comparison_locations"] = comparison_locations
            self.pending_runs.append(params)
            self.start_next_run()
            return
        for year in years:
            params = dict(
                city=city,
                state=state,
                year=year,
                output_dir=Path(out_dir),
                api_key=api_key,
                geography="place" if is_place else ("county" if is_county else ("block_groups" if is_block else "tracts")),
                survey=survey,
            )
            if is_block or geography_text == "Tracts":
                params["geoids"] = geoids or None
            if is_block or is_county or is_place or geography_text == "Tracts":
                params["tables"] = tables
            if is_county or is_place:
                comparison_locations = [
                    entry
                    for entry in self.comparison_locations
                    if entry.get("location") and entry.get("state")
                    and not (
                        str(entry.get("location", "")).strip() == city
                        and str(entry.get("state", "")).strip() == state
                    )
                ]
                if comparison_locations:
                    params["comparison_locations"] = comparison_locations
            self.pending_runs.append(params)
        self.start_next_run()

    def start_pipeline(self, params: dict):
        self.set_busy_state(True)
        self.set_analysis_status("Running")
        self.append_log("---- RUN START ----")

        self.thread = QThread()
        self.worker = Worker(params)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)

        self.worker.done.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.on_run_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def download_clicked(self):
        geography = self.geo_combo.currentText()
        if self.dataset_combo.currentText() == "ACS 1-Year" and geography == "County":
            QMessageBox.information(
                self,
                "Download shapefile",
                "ACS 1-Year county mode does not use tract or block-group shapefiles.",
            )
            return
        state = self.state_combo.currentText().strip()
        if not state:
            QMessageBox.warning(self, "Missing input", "Please enter a state.")
            return
        if geography == "Block groups":
            kind = "block_groups"
        elif geography == "Places":
            kind = "place"
        else:
            kind = "tracts"
        years = self.selected_years()
        if not years:
            QMessageBox.warning(self, "Missing input", "Please select at least one year.")
            return
        self.cancel_requested = False
        self.set_analysis_status("Downloading")
        self.set_busy_state(True)
        self.pending_downloads = [{"kind": kind, "state": state, "year": y} for y in years]
        self.start_next_download()

    def start_download(self, kind: str, state: str, year: int):
        if self.download_thread and isValid(self.download_thread):
            if self.download_thread.isRunning():
                QMessageBox.information(self, "Download", "A download is already in progress.")
                return
        elif self.download_thread and not isValid(self.download_thread):
            self.download_thread = None
            self.download_worker = None
        self.download_progress.setValue(0)
        self.download_progress_label.setText(f"Download {year}: 0%")
        self.set_busy_state(True)

        self.download_thread = QThread()
        self.download_worker = DownloadWorker(state, kind, year)
        self.download_worker.moveToThread(self.download_thread)

        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.log.connect(self.append_log)
        self.download_worker.progress.connect(self.on_download_progress)
        self.download_worker.done.connect(self.on_download_done)
        self.download_worker.error.connect(self.on_download_error)

        self.download_worker.done.connect(self.download_thread.quit)
        self.download_worker.error.connect(self.download_thread.quit)
        self.download_thread.finished.connect(self.on_download_thread_finished)
        self.download_thread.start()

    @Slot(object)
    def on_download_done(self, path: Path):
        kind = self.download_worker.kind if self.download_worker else ""
        self.append_log(f"Download complete: {path}")
        year = self.download_worker.year if self.download_worker else ""
        if year:
            self.download_progress_label.setText(f"Download {year}: 100%")
        if self.pending_params:
            params = dict(self.pending_params)
            params["shapefile_path"] = Path(path)
            self.pending_params = None
            self.set_analysis_status("Running")
            self.start_pipeline(params)
            return
        self.download_continue = True
        if not self.pending_downloads:
            self.set_analysis_status("Download complete")

    @Slot(str)
    def on_download_error(self, msg: str):
        if self.cancel_requested:
            return
        self.append_log(f"Download failed: {msg}")
        self.download_progress_label.setText("Download failed")
        self.set_analysis_status("Failed")
        self.pending_params = None
        if self.pending_runs:
            self.start_next_run()
            return
        if self.pending_downloads:
            self.download_continue = True

    @Slot()
    def on_download_thread_finished(self):
        if self.download_thread:
            self.download_thread.deleteLater()
        self.download_thread = None
        self.download_worker = None
        if self.cancel_requested:
            self.cancel_requested = False
            self.set_busy_state(False)
            return
        if self.download_continue:
            self.download_continue = False
            self.start_next_download()

    @Slot()
    def on_run_thread_finished(self):
        self.thread = None
        self.worker = None
        if self.cancel_requested:
            self.cancel_requested = False
            self.set_busy_state(False)

    @Slot(int)
    def on_download_progress(self, pct: int):
        self.download_progress.setValue(pct)
        year = self.download_worker.year if self.download_worker else ""
        if year:
            self.download_progress_label.setText(f"Download {year}: {pct}%")

    @Slot(object)
    def on_done(self, res: RunResult):
        self.append_log("---- RUN COMPLETE ----")
        self.append_log(str(res))
        self.append_log("ANALYSIS COMPLETE")
        gpkg_text = str(res.output_gpkg) if res.output_gpkg else "Not generated"
        layer_text = res.output_layer or "N/A"
        self.set_analysis_status("Complete")
        QMessageBox.information(
            self,
            "Analysis complete",
            f"Analysis complete.\n\nType: {res.geography}\n\nExcel:\n{res.output_xlsx}\n\nGeoPackage:\n{gpkg_text}\nLayer: {layer_text}",
        )
        if self.pending_runs:
            self.start_next_run()
        else:
            self.set_busy_state(False)

    @Slot(str)
    def on_error(self, msg: str):
        if self.cancel_requested:
            return
        self.append_log("---- RUN FAILED ----")
        self.append_log(msg)
        self.set_analysis_status("Failed")
        if self.pending_runs:
            self.start_next_run()
        else:
            self.set_busy_state(False)
        QMessageBox.critical(self, "Error", msg)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QWidget {
            background: #FFF4E6;
        }
        QGroupBox {
            border: 1px solid #F1C27D;
            margin-top: 10px;
            padding: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 3px 0 3px;
            color: #7A4B00;
        }
        QLineEdit, QComboBox, QTextEdit {
            background: #FFF9F2;
            border: 1px solid #F1C27D;
            padding: 4px;
        }
        QPushButton {
            background: #FFD8A8;
            border: 1px solid #F1C27D;
            padding: 6px 10px;
        }
        QPushButton:hover { 
            background: #FFC078;
        }
        QPushButton:disabled {
            background: #FFE8CC;
            color: #8C6D3D;
        }
        """
    )
    w = MainWindow()
    w.resize(840, 560)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
