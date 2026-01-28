import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot, QThread
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QMessageBox,
    QGroupBox,
    QComboBox,
)

from census_app_core import run_blockgroup_pipeline, run_tract_pipeline, RunResult

CONFIG_PATH = Path.cwd() / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


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
            if geography == "tracts":
                res = run_tract_pipeline(logger=logger, **params)
            else:
                res = run_blockgroup_pipeline(logger=logger, **params)
            self.done.emit(res)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Census Tool Test")

        self.thread = None
        self.worker = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # ---- Inputs ----
        input_box = QGroupBox("Inputs")
        input_layout = QGridLayout(input_box)

        self.city_input = QLineEdit()
        self.state_input = QLineEdit()

        self.year_combo = QComboBox()
        for y in range(2018, 2024):
            self.year_combo.addItem(str(y))
        self.year_combo.setCurrentText("2023")

        input_layout.addWidget(QLabel("County:"), 0, 0)
        input_layout.addWidget(self.city_input, 0, 1)
        input_layout.addWidget(QLabel("State (name or abbrev):"), 0, 2)
        input_layout.addWidget(self.state_input, 0, 3)
        input_layout.addWidget(QLabel("ACS year:"), 1, 0)
        input_layout.addWidget(self.year_combo, 1, 1)

        layout.addWidget(input_box)

        geo_box = QGroupBox("Geography")
        geo_layout = QHBoxLayout(geo_box)
        self.geo_combo = QComboBox()
        self.geo_combo.addItems(["Block groups", "Tracts"])
        self.geo_combo.currentIndexChanged.connect(self.update_geo_state)
        geo_layout.addWidget(QLabel("Output level:"))
        geo_layout.addWidget(self.geo_combo)
        geo_layout.addStretch(1)
        layout.addWidget(geo_box)

        # ---- Paths ----
        path_box = QGroupBox("Paths")
        path_layout = QGridLayout(path_box)

        self.output_dir = QLineEdit(str(Path.cwd()))
        self.btn_output = QPushButton("Browse...")
        self.btn_output.clicked.connect(self.pick_output_dir)

        default_shp = Path.cwd() / "shapefile" / "USA Census Block Group Boundaries.shp"
        self.bg_shp = QLineEdit(str(default_shp))
        self.btn_bg_shp = QPushButton("Browse...")
        self.btn_bg_shp.clicked.connect(self.pick_bg_shapefile)

        default_tract_shp = Path.cwd() / "shapefile" / "tl_rd22_55_tract.shp"
        self.tract_shp = QLineEdit(str(default_tract_shp))
        self.btn_tract_shp = QPushButton("Browse...")
        self.btn_tract_shp.clicked.connect(self.pick_tract_shapefile)

        path_layout.addWidget(QLabel("Output folder:"), 0, 0)
        path_layout.addWidget(self.output_dir, 0, 1)
        path_layout.addWidget(self.btn_output, 0, 2)

        path_layout.addWidget(QLabel("Block group shapefile:"), 1, 0)
        path_layout.addWidget(self.bg_shp, 1, 1)
        path_layout.addWidget(self.btn_bg_shp, 1, 2)

        path_layout.addWidget(QLabel("Tract shapefile:"), 2, 0)
        path_layout.addWidget(self.tract_shp, 2, 1)
        path_layout.addWidget(self.btn_tract_shp, 2, 2)

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

        self.update_geo_state()

        # ---- Run + Log ----
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.run_clicked)
        self.btn_open_out = QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(self.open_output_folder)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_open_out)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)

    def append_log(self, msg: str):
        self.log.append(msg)

    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_dir.setText(d)

    def pick_bg_shapefile(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select block group shapefile", "", "Shapefile (*.shp)"
        )
        if f:
            self.bg_shp.setText(f)

    def pick_tract_shapefile(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select tract shapefile", "", "Shapefile (*.shp)"
        )
        if f:
            self.tract_shp.setText(f)

    def update_geo_state(self):
        is_block = self.geo_combo.currentText() == "Block groups"
        self.bg_shp.setEnabled(is_block)
        self.btn_bg_shp.setEnabled(is_block)
        self.tract_shp.setEnabled(not is_block)
        self.btn_tract_shp.setEnabled(not is_block)

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
        city = self.city_input.text().strip()
        state = self.state_input.text().strip()
        year = int(self.year_combo.currentText())
        out_dir = self.output_dir.text().strip()
        is_block = self.geo_combo.currentText() == "Block groups"
        shp_path = self.bg_shp.text().strip() if is_block else self.tract_shp.text().strip()
        api_key = self.api_key.text().strip() or None

        if not city or not state:
            QMessageBox.warning(self, "Missing input", "Please enter county and state.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Missing output folder", "Choose an output folder.")
            return
        if not shp_path or not os.path.exists(shp_path):
            QMessageBox.warning(self, "Missing shapefile", "Select a valid shapefile.")
            return

        params = dict(
            city=city,
            state=state,
            year=year,
            output_dir=Path(out_dir),
            shapefile_path=Path(shp_path),
            api_key=api_key,
            geography="block_groups" if is_block else "tracts",
        )

        self.btn_run.setEnabled(False)
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
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(object)
    def on_done(self, res: RunResult):
        self.append_log("---- RUN COMPLETE ----")
        self.append_log(str(res))
        self.btn_run.setEnabled(True)
        QMessageBox.information(
            self,
            "Done",
            f"Type: {res.geography}\n\nExcel:\n{res.output_xlsx}\n\nGeoPackage:\n{res.output_gpkg}\nLayer: {res.output_layer}",
        )

    @Slot(str)
    def on_error(self, msg: str):
        self.append_log("---- RUN FAILED ----")
        self.append_log(msg)
        self.btn_run.setEnabled(True)
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
    w.resize(900, 600)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
