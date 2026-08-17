# -*- coding: utf-8 -*-
"""
EIS Equivalent-Circuit Fitting

Desktop analysis tool for impedance spectra measured by the Raspberry Pi
instrument. Reads a CSV, fits the equivalent circuit Rs + (Rct || Cdl), and
reports the fitted parameters with Nyquist, Bode and residual plots.

Requires: PySide6, numpy, scipy, matplotlib
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QToolBar, QVBoxLayout, QWidget,
)

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"

APP_TITLE = "EIS Equivalent-Circuit Fitting"
APP_SUBTITLE = "Rs + (Rct || Cdl) Analysis and Visualization"

NAVY = "#12335B"
TEAL = "#00A6B2"
LIGHT_BG = "#F4F7FA"
CARD_BG = "#FFFFFF"
TEXT_DARK = "#1F2937"
MUTED = "#64748B"
BORDER = "#D9E2EC"
ACCENT = "#E8A33D"

HEADER_HEIGHT = 108

# Logos are shown left to right in this order, each scaled to fit inside its
# own (width, height) box with the aspect ratio preserved. Boxes rather than a
# single dimension, because the three logos have very different proportions and
# a fixed height either crops the wide ones or shrinks the tall one.
LOGO_FILES = [
    ("uin_walisongo.png", (72, 66)),
    ("sampoerna.png", (156, 52)),
    ("lpdp.png", (140, 52)),
]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

@dataclass
class FitResult:
    frequency: np.ndarray
    z_real: np.ndarray
    z_imag: np.ndarray
    fit_real: np.ndarray
    fit_imag: np.ndarray
    rs: float
    rct: float
    cdl: float
    fc: float
    rmse: float
    nrmse: float


def circuit_impedance(frequency, rs, rct, cdl):
    """Impedance of a resistor in series with a parallel RC pair."""
    omega = 2.0 * np.pi * frequency
    return rs + rct / (1.0 + 1j * omega * rct * cdl)


def _residual(params, frequency, z_real, z_imag):
    rs, rct, cdl = params
    z = circuit_impedance(frequency, rs, rct, cdl)
    return np.concatenate([z.real - z_real, z.imag - z_imag])


def fit_circuit(frequency, z_real, z_imag) -> FitResult:
    """Nonlinear least-squares fit of Rs + (Rct || Cdl) to measured data."""
    if frequency.size < 4:
        raise ValueError(
            "At least 4 points are needed to fit three parameters.\n"
            "The selected frequency range contains %d." % frequency.size
        )

    # Starting estimates read straight off the curve: the smallest real part is
    # close to Rs, the span of the real part is close to Rct, and the apex of
    # the arc marks the characteristic frequency.
    rs0 = max(float(np.min(z_real)), 1e-9)
    rct0 = max(float(np.max(z_real)) - rs0, 1.0)
    apex = int(np.argmax(-z_imag))
    fc0 = max(float(frequency[apex]), 1e-9)
    cdl0 = 1.0 / (2.0 * np.pi * fc0 * rct0)

    solution = least_squares(
        _residual,
        [rs0, rct0, cdl0],
        args=(frequency, z_real, z_imag),
        bounds=([0.0, 0.0, 1e-15], [np.inf, np.inf, 1.0]),
        max_nfev=20000,
    )

    rs, rct, cdl = (float(v) for v in solution.x)
    fitted = circuit_impedance(frequency, rs, rct, cdl)

    residuals = solution.fun
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    scale = float(np.sqrt(np.mean(z_real ** 2 + z_imag ** 2)))
    nrmse = rmse / scale * 100.0 if scale > 0 else float("nan")

    fc = 1.0 / (2.0 * np.pi * rct * cdl) if rct > 0 and cdl > 0 else float("nan")

    return FitResult(
        frequency=frequency, z_real=z_real, z_imag=z_imag,
        fit_real=fitted.real, fit_imag=fitted.imag,
        rs=rs, rct=rct, cdl=cdl, fc=fc, rmse=rmse, nrmse=nrmse,
    )


# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------

def detect_delimiter(header_line: str) -> str:
    """
    Picks the separator by counting candidates in the header line.

    This is deliberately simple rather than using csv.Sniffer, which guesses
    from a sample and can pick a character that merely happens to appear in
    the data. Counting the header is predictable and gives the same answer
    every time for the same file.
    """
    counts = {sep: header_line.count(sep) for sep in (",", ";", "\t", "|")}
    best = max(counts, key=lambda sep: counts[sep])
    return best if counts[best] > 0 else ","


def read_csv(path: Path):
    """Reads a CSV into a header list and a list of row lists."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        text = handle.read()

    if not text.strip():
        raise ValueError("The file is empty.")

    first_line = text.splitlines()[0]
    delimiter = detect_delimiter(first_line)

    rows = [
        row for row in csv.reader(text.splitlines(), delimiter=delimiter)
        if any(cell.strip() for cell in row)
    ]

    if len(rows) < 2:
        raise ValueError(
            "The file has no data rows below its header.\n"
            "Separator detected: %r" % delimiter
        )

    header = [cell.strip() for cell in rows[0]]
    return header, rows[1:]


def guess_column(header, keywords, fallback):
    """Finds the first column whose name contains one of the keywords."""
    lowered = [(i, name.lower()) for i, name in enumerate(header)]
    for keyword in keywords:
        for index, name in lowered:
            if keyword in name:
                return index
    return min(fallback, len(header) - 1)


def unit_multiplier(header_name: str) -> float:
    """
    Converts whatever unit the column header declares into ohms.

    This is why the multiplier is detected rather than typed. The instrument
    exports ohms, the original analyser export used kilohms, and getting it
    wrong shifts every fitted resistance by a factor of a thousand without
    anything visibly failing.
    """
    lowered = header_name.lower()
    if "mohm" in lowered or "megohm" in lowered:
        return 1e6
    if "kohm" in lowered or "kilohm" in lowered:
        return 1e3
    return 1.0


def describe_spacing(frequency: np.ndarray) -> str:
    """
    Reports whether the frequencies step by a constant amount or a constant
    ratio, so the file describes itself even when the filename does not.
    """
    frequency = np.unique(frequency[np.isfinite(frequency) & (frequency > 0)])
    if frequency.size < 3:
        return "too few points to tell"

    linear_steps = np.diff(frequency)
    log_steps = np.diff(np.log(frequency))

    def spread(values):
        mean = np.mean(values)
        return np.std(values) / abs(mean) if mean else np.inf

    return "linear" if spread(linear_steps) < spread(log_steps) else "logarithmic"


def demo_spectrum():
    """
    Synthetic spectrum from the reference circuit described in the report,
    so the interface can be demonstrated with no file loaded.
    """
    rs, rct, cdl = 46.35, 9524.0, 1.088e-6
    frequency = np.logspace(0, 5, 60)
    z = circuit_impedance(frequency, rs, rct, cdl)
    header = ["Freq [Hz]", "Zreal [Ohm]", "Zimag [Ohm]"]
    rows = [
        ["%.3f" % f, "%.6f" % zr, "%.6f" % zi]
        for f, zr, zi in zip(frequency, z.real, z.imag)
    ]
    return header, rows


# ----------------------------------------------------------------------
# Small widgets
# ----------------------------------------------------------------------

class NoScrollComboBox(QComboBox):
    """
    A combo box that ignores the mouse wheel unless it has keyboard focus.

    The default behaviour changes the selection whenever the pointer happens to
    be over the box while the sidebar is scrolled, which silently alters the
    column mapping without the user realising. Ignoring the event also lets it
    pass through to the scroll area, so the sidebar still scrolls normally.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class MetricCard(QFrame):
    """One of the summary tiles along the top of the window."""

    def __init__(self, caption: str):
        super().__init__()
        self.setObjectName("metricCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title = QLabel(caption)
        title.setObjectName("metricCaption")
        title.setWordWrap(True)

        self._value = QLabel("-")
        self._value.setObjectName("metricValue")

        layout.addWidget(title)
        layout.addWidget(self._value)

    def set_value(self, text: str):
        self._value.setText(text)


class PlotCanvas(FigureCanvasQTAgg):
    """A matplotlib figure sized for one tab."""

    def __init__(self):
        self.figure = Figure(figsize=(6, 4), dpi=100, facecolor=CARD_BG)
        super().__init__(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.reset("No data loaded")

    def reset(self, message: str):
        self.axes.clear()
        self.axes.text(0.5, 0.5, message, ha="center", va="center",
                       color=MUTED, fontsize=11, transform=self.axes.transAxes)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        self.figure.tight_layout()
        self.draw_idle()

    def prepare(self):
        self.axes.clear()
        for spine in self.axes.spines.values():
            spine.set_visible(True)
            spine.set_color(BORDER)
        self.axes.grid(True, color="#EDF1F5", linewidth=0.8)
        self.axes.set_facecolor(CARD_BG)


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    return label


def field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------

class EISWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)
        self.setAcceptDrops(True)

        self._header = []
        self._rows = []
        self._source_name = "No data loaded"
        self._result = None

        self._build_toolbar()

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(12)
        body_layout.addWidget(self._build_sidebar(), 0)
        body_layout.addWidget(self._build_main_area(), 1)
        root.addWidget(body, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready. Open a CSV, or drag one onto the window."
        )

        self.setStyleSheet(self._stylesheet())

    # -- construction ---------------------------------------------------

    def _build_toolbar(self):
        bar = QToolBar()
        bar.setMovable(False)
        bar.setObjectName("topBar")
        self.addToolBar(bar)

        open_action = QAction("Open CSV", self)
        open_action.triggered.connect(self.open_csv)

        demo_action = QAction("Load Demo", self)
        demo_action.triggered.connect(self.load_demo)

        run_action = QAction("Run Fitting", self)
        run_action.triggered.connect(self.run_fitting)

        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results)

        for action in (open_action, demo_action, run_action, export_action):
            bar.addAction(action)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(HEADER_HEIGHT)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 12, 22, 12)
        layout.setSpacing(14)

        titles = QVBoxLayout()
        titles.setSpacing(2)

        title = QLabel(APP_TITLE)
        title.setObjectName("appTitle")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("appSubtitle")

        titles.addStretch(1)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        titles.addStretch(1)

        layout.addLayout(titles)
        layout.addStretch(1)

        # Institutional logos, in the order requested.
        for filename, (max_width, max_height) in LOGO_FILES:
            path = ASSET_DIR / filename
            card = QFrame()
            card.setObjectName("logoCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 8, 12, 8)

            label = QLabel()
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                # Missing asset must not stop the application from starting.
                label.setText(path.stem.replace("_", " ").title())
                label.setObjectName("logoFallback")
                label.setFixedSize(max_width, max_height)
            else:
                scaled = pixmap.scaled(
                    max_width, max_height,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                label.setPixmap(scaled)
                # A fixed box keeps the three cards the same height even though
                # the logos inside them are different shapes.
                label.setFixedSize(scaled.width(), max_height)
            label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(label)
            layout.addWidget(card, 0, Qt.AlignVCenter)

        return header

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(268)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # Data source
        source_box = QFrame()
        source_box.setObjectName("card")
        source_layout = QVBoxLayout(source_box)
        source_layout.setContentsMargins(14, 12, 14, 14)
        source_layout.setSpacing(8)

        source_layout.addWidget(section_label("Data source"))

        self.source_label = QLabel(self._source_name)
        self.source_label.setObjectName("sourceName")
        self.source_label.setWordWrap(True)
        source_layout.addWidget(self.source_label)

        self.rows_label = QLabel("Drag a CSV here or use Open CSV")
        self.rows_label.setObjectName("hint")
        self.rows_label.setWordWrap(True)
        source_layout.addWidget(self.rows_label)

        # What the file itself contains, so the reader does not have to rely on
        # the filename to know the range, the point count or the spacing.
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("summaryText")
        self.summary_label.setWordWrap(True)
        source_layout.addWidget(self.summary_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        open_button = QPushButton("Open CSV")
        open_button.clicked.connect(self.open_csv)
        demo_button = QPushButton("Demo Data")
        demo_button.clicked.connect(self.load_demo)
        buttons.addWidget(open_button)
        buttons.addWidget(demo_button)
        source_layout.addLayout(buttons)

        outer.addWidget(source_box)

        # Column mapping
        mapping_box = QFrame()
        mapping_box.setObjectName("card")
        mapping_layout = QVBoxLayout(mapping_box)
        mapping_layout.setContentsMargins(14, 12, 14, 14)
        mapping_layout.setSpacing(6)

        mapping_layout.addWidget(section_label("Column mapping"))

        self.freq_combo = NoScrollComboBox()
        self.real_combo = NoScrollComboBox()
        self.imag_combo = NoScrollComboBox()
        for caption, combo in (
            ("Frequency", self.freq_combo),
            ("Real Z'", self.real_combo),
            ("Imaginary Z''", self.imag_combo),
        ):
            mapping_layout.addWidget(field_label(caption))
            mapping_layout.addWidget(combo)

        reset_button = QPushButton("Reset to auto-detected")
        reset_button.clicked.connect(self._reset_mapping)
        mapping_layout.addSpacing(6)
        mapping_layout.addWidget(reset_button)

        # Picking a column already used by another box swaps the two, so the
        # three mappings can never point at the same column.
        self._previous_index = {}
        for combo in (self.freq_combo, self.real_combo, self.imag_combo):
            combo.currentIndexChanged.connect(
                lambda _, box=combo: self._on_mapping_changed(box))

        outer.addWidget(mapping_box)

        # Fitting configuration
        config_box = QFrame()
        config_box.setObjectName("card")
        config_layout = QVBoxLayout(config_box)
        config_layout.setContentsMargins(14, 12, 14, 14)
        config_layout.setSpacing(6)

        config_layout.addWidget(section_label("Fitting configuration"))

        config_layout.addWidget(field_label("Unit multiplier"))
        self.multiplier_edit = QLineEdit("1")
        config_layout.addWidget(self.multiplier_edit)

        self.multiplier_hint = QLabel("Detected from the column header")
        self.multiplier_hint.setObjectName("hint")
        self.multiplier_hint.setWordWrap(True)
        config_layout.addWidget(self.multiplier_hint)

        config_layout.addWidget(field_label("Min frequency (Hz)"))
        self.fmin_edit = QLineEdit()
        config_layout.addWidget(self.fmin_edit)

        config_layout.addWidget(field_label("Max frequency (Hz)"))
        self.fmax_edit = QLineEdit()
        config_layout.addWidget(self.fmax_edit)

        range_hint = QLabel("Leave blank to use every point in the file.")
        range_hint.setObjectName("hint")
        range_hint.setWordWrap(True)
        config_layout.addWidget(range_hint)

        config_reset = QPushButton("Reset to auto-detected")
        config_reset.clicked.connect(self._reset_config_clicked)
        config_layout.addSpacing(6)
        config_layout.addWidget(config_reset)

        outer.addWidget(config_box)

        # Circuit
        circuit_box = QFrame()
        circuit_box.setObjectName("card")
        circuit_layout = QVBoxLayout(circuit_box)
        circuit_layout.setContentsMargins(14, 12, 14, 14)
        circuit_layout.setSpacing(4)
        circuit_layout.addWidget(section_label("Equivalent circuit"))

        model = QLabel("Rs + (Rct || Cdl)")
        model.setObjectName("circuitModel")
        model.setAlignment(Qt.AlignCenter)
        circuit_layout.addWidget(model)

        note = QLabel("Series resistance with a parallel RC pair")
        note.setObjectName("hint")
        note.setAlignment(Qt.AlignCenter)
        note.setWordWrap(True)
        circuit_layout.addWidget(note)

        outer.addWidget(circuit_box)

        run_button = QPushButton("Run Equivalent-Circuit Fitting")
        run_button.setObjectName("primaryButton")
        run_button.setFixedHeight(40)
        run_button.clicked.connect(self.run_fitting)
        outer.addWidget(run_button)

        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(panel)
        scroll.setFixedWidth(288)
        return scroll

    def _build_main_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Metric cards
        metrics = QWidget()
        metrics_layout = QHBoxLayout(metrics)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(10)

        self.card_rs = MetricCard("Series resistance, Rs")
        self.card_rct = MetricCard("Charge-transfer resistance, Rct")
        self.card_cdl = MetricCard("Double-layer capacitance, Cdl")
        self.card_fc = MetricCard("Characteristic frequency, fc")
        self.card_rmse = MetricCard("Complex RMSE")

        for card in (self.card_rs, self.card_rct, self.card_cdl,
                     self.card_fc, self.card_rmse):
            metrics_layout.addWidget(card)

        layout.addWidget(metrics)

        # Plot tabs
        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabs")

        self.nyquist = PlotCanvas()
        self.bode_mag = PlotCanvas()
        self.bode_phase = PlotCanvas()
        self.residual = PlotCanvas()

        for canvas, name in (
            (self.nyquist, "Nyquist"),
            (self.bode_mag, "Bode |Z|"),
            (self.bode_phase, "Bode phase"),
            (self.residual, "Residual"),
        ):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 8, 8, 8)
            page_layout.addWidget(canvas)
            self.tabs.addTab(page, name)

        self.preview = QTableWidget()
        self.preview.setObjectName("preview")
        self.preview.setAlternatingRowColors(True)
        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.addWidget(self.preview)
        self.tabs.addTab(preview_page, "Data preview")

        layout.addWidget(self.tabs, 1)
        return container

    # -- drag and drop --------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".csv", ".txt")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in (".csv", ".txt"):
                self.load_file(path)
                event.acceptProposedAction()
                return
        event.ignore()

    # -- loading --------------------------------------------------------

    def open_csv(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open measurement CSV", "",
            "CSV files (*.csv *.txt);;All files (*)",
        )
        if filename:
            self.load_file(Path(filename))

    def load_file(self, path: Path):
        try:
            header, rows = read_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open CSV", "Could not read the file.\n\n%s" % exc)
            return
        self._apply_data(header, rows, path.name)
        self.statusBar().showMessage("Loaded %s, %d rows." % (path.name, len(rows)))

    def load_demo(self):
        header, rows = demo_spectrum()
        self._apply_data(header, rows, "Synthetic demo data")
        self.statusBar().showMessage(
            "Loaded synthetic demo data generated from the reference circuit."
        )

    def _apply_data(self, header, rows, source_name):
        self._header = header
        self._rows = rows
        self._source_name = source_name
        self._result = None

        self.source_label.setText(source_name)
        self.rows_label.setText("%d rows, %d columns" % (len(rows), len(header)))
        self.summary_label.setText("")

        for combo in (self.freq_combo, self.real_combo, self.imag_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(header)

        self._auto_detect_columns()

        for combo in (self.freq_combo, self.real_combo, self.imag_combo):
            combo.blockSignals(False)

        self._remember_indices()
        self._refresh_multiplier()

        # The fitting range defaults to whatever the file actually covers,
        # so a valid file never produces an empty selection.
        self._reset_fitting_config()

        self._fill_preview()
        self._reset_outputs()

    def _auto_detect_columns(self):
        """
        Restores the mapping the application chooses on its own, so a manual
        change is always one click away from being undone.
        """
        header = self._header
        if not header:
            return

        blocked = [c.signalsBlocked() for c in
                   (self.freq_combo, self.real_combo, self.imag_combo)]
        for combo in (self.freq_combo, self.real_combo, self.imag_combo):
            combo.blockSignals(True)

        self.freq_combo.setCurrentIndex(
            guess_column(header, ("freq", "frequency", "hz"), 0))
        self.real_combo.setCurrentIndex(
            guess_column(header, ("zreal", "z'", "real", "rs", "resistance"), 1))
        self.imag_combo.setCurrentIndex(
            guess_column(header, ("zimag", "z''", "imag", "xs", "reactance"), 2))

        for combo, was_blocked in zip(
                (self.freq_combo, self.real_combo, self.imag_combo), blocked):
            combo.blockSignals(was_blocked)

        self._remember_indices()
        self._refresh_multiplier()

    def _reset_mapping(self):
        self._auto_detect_columns()
        self.statusBar().showMessage("Column mapping reset to automatic detection.")

    def _on_mapping_changed(self, combo):
        new_index = combo.currentIndex()
        if new_index < 0:
            return

        previous = self._previous_index.get(combo, -1)

        for other in (self.freq_combo, self.real_combo, self.imag_combo):
            if other is combo or other.currentIndex() != new_index:
                continue
            # Give the clashing box the index this one just left.
            other.blockSignals(True)
            other.setCurrentIndex(previous)
            other.blockSignals(False)
            self._previous_index[other] = previous

        self._previous_index[combo] = new_index
        self._refresh_multiplier()

    def _remember_indices(self):
        for combo in (self.freq_combo, self.real_combo, self.imag_combo):
            self._previous_index[combo] = combo.currentIndex()

    def _reset_fitting_config(self):
        """
        Puts the multiplier and the frequency range back to what the loaded
        file implies: the unit declared in the column header, and the full
        range the data covers.
        """
        self._refresh_multiplier()
        try:
            frequency = self._column_values(self.freq_combo.currentIndex())
            frequency = frequency[np.isfinite(frequency) & (frequency > 0)]
            if frequency.size:
                self.fmin_edit.setText("%g" % frequency.min())
                self.fmax_edit.setText("%g" % frequency.max())
                locale = QLocale(QLocale.English)
                self.summary_label.setText(
                    "%d measured points\n%s to %s Hz\n%s spacing"
                    % (frequency.size,
                       locale.toString(int(round(frequency.min()))),
                       locale.toString(int(round(frequency.max()))),
                       describe_spacing(frequency).capitalize())
                )
        except Exception:
            self.fmin_edit.clear()
            self.fmax_edit.clear()
            self.summary_label.setText("")

    def _reset_config_clicked(self):
        if not self._rows:
            return
        self._reset_fitting_config()
        self.statusBar().showMessage(
            "Fitting configuration reset to the values the file implies.")

    def _refresh_multiplier(self):
        index = self.real_combo.currentIndex()
        if index < 0 or index >= len(self._header):
            return
        name = self._header[index]
        factor = unit_multiplier(name)
        self.multiplier_edit.setText("%g" % factor)
        if factor == 1.0:
            self.multiplier_hint.setText("Column reads in ohms, no conversion applied.")
        else:
            self.multiplier_hint.setText(
                "Column header declares a scaled unit, converting to ohms.")

    @staticmethod
    def _to_float(text) -> float:
        """
        Parses one cell. Accepts a decimal comma as well as a decimal point,
        because a spreadsheet saved under some locales writes 1,234 for 1.234.
        """
        if text is None:
            return math.nan
        cleaned = str(text).strip().replace("\u00A0", "").replace(" ", "")
        if not cleaned:
            return math.nan
        try:
            return float(cleaned)
        except ValueError:
            pass
        if "," in cleaned and "." not in cleaned:
            try:
                return float(cleaned.replace(",", "."))
            except ValueError:
                return math.nan
        return math.nan

    def _column_values(self, index: int) -> np.ndarray:
        if index < 0:
            return np.asarray([], dtype=float)
        values = [
            self._to_float(row[index]) if index < len(row) else math.nan
            for row in self._rows
        ]
        return np.asarray(values, dtype=float)

    def _fill_preview(self):
        shown = self._rows[:250]
        self.preview.setColumnCount(len(self._header))
        self.preview.setHorizontalHeaderLabels(self._header)
        self.preview.setRowCount(len(shown))
        for r, row in enumerate(shown):
            for c in range(len(self._header)):
                text = row[c] if c < len(row) else ""
                self.preview.setItem(r, c, QTableWidgetItem(str(text)))
        self.preview.resizeColumnsToContents()

    def _reset_outputs(self):
        for card in (self.card_rs, self.card_rct, self.card_cdl,
                     self.card_fc, self.card_rmse):
            card.set_value("-")
        for canvas in (self.nyquist, self.bode_mag, self.bode_phase, self.residual):
            canvas.reset("Press Run Fitting")

    # -- fitting --------------------------------------------------------

    def run_fitting(self):
        if not self._rows:
            QMessageBox.information(
                self, "Run Fitting",
                "Load a measurement CSV first, or press Demo Data.")
            return

        try:
            chosen = [
                self.freq_combo.currentIndex(),
                self.real_combo.currentIndex(),
                self.imag_combo.currentIndex(),
            ]
            if len(set(chosen)) < 3:
                raise ValueError(
                    "Frequency, real Z' and imaginary Z'' must be three "
                    "different columns.\n\n"
                    "Currently selected:\n  Frequency: %s\n  Real Z': %s\n"
                    "  Imaginary Z'': %s"
                    % (self.freq_combo.currentText(),
                       self.real_combo.currentText(),
                       self.imag_combo.currentText())
                )

            multiplier = float(self.multiplier_edit.text())
            if not math.isfinite(multiplier) or multiplier == 0:
                raise ValueError("The unit multiplier must be a non-zero number.")

            frequency = self._column_values(self.freq_combo.currentIndex())
            z_real = self._column_values(self.real_combo.currentIndex()) * multiplier
            z_imag = self._column_values(self.imag_combo.currentIndex()) * multiplier

            numeric = np.isfinite(frequency) & np.isfinite(z_real) & np.isfinite(z_imag)
            numeric &= frequency > 0

            if not numeric.any():
                raise ValueError(
                    "The file loaded (%d rows), but the selected columns "
                    "contain no usable numbers.\n\n"
                    "Frequency: %s\nReal: %s\nImaginary: %s\n\n"
                    "Open the Data preview tab and check the column mapping."
                    % (len(self._rows), self.freq_combo.currentText(),
                       self.real_combo.currentText(), self.imag_combo.currentText())
                )

            valid = numeric.copy()
            fmin = self._optional_float(self.fmin_edit.text())
            fmax = self._optional_float(self.fmax_edit.text())
            if fmin is not None:
                valid &= frequency >= fmin
            if fmax is not None:
                valid &= frequency <= fmax

            if not valid.any():
                raise ValueError(
                    "%d points were read, but none fall inside the frequency "
                    "range %s to %s Hz.\n\nThe data covers %g to %g Hz. "
                    "Clear both boxes to use every point."
                    % (int(numeric.sum()),
                       self.fmin_edit.text() or "(none)",
                       self.fmax_edit.text() or "(none)",
                       frequency[numeric].min(), frequency[numeric].max())
                )

            frequency, z_real, z_imag = frequency[valid], z_real[valid], z_imag[valid]

            order = np.argsort(frequency)
            frequency, z_real, z_imag = frequency[order], z_real[order], z_imag[order]

            result = fit_circuit(frequency, z_real, z_imag)

        except Exception as exc:
            # Old results must not stay on screen next to an error, or the
            # cards would still show numbers that no longer describe anything.
            self._result = None
            self._reset_outputs()
            QMessageBox.critical(self, "Run Fitting", str(exc))
            self.statusBar().showMessage("Fitting failed. Previous results cleared.")
            return

        self._result = result
        self._update_cards(result)
        self._draw_all(result)
        self.statusBar().showMessage(
            "Fitting complete. Rct = %.4g kOhm, Cdl = %.4g uF, NRMSE = %.2f%%, "
            "%d points used."
            % (result.rct / 1000.0, result.cdl * 1e6, result.nrmse, frequency.size)
        )

    @classmethod
    def _optional_float(cls, text: str):
        """
        Reads a frequency box. Blank means no limit. A decimal comma is
        accepted as well as a decimal point, so 0,01 and 0.01 both work.
        """
        text = text.strip()
        if not text:
            return None
        value = cls._to_float(text)
        if not math.isfinite(value):
            raise ValueError(
                "%r is not a number. Enter a frequency such as 0.01, "
                "or leave the box blank for no limit." % text
            )
        return value

    def _update_cards(self, result: FitResult):
        self.card_rs.set_value(self._format_ohm(result.rs))
        self.card_rct.set_value(self._format_ohm(result.rct))
        self.card_cdl.set_value(self._format_farad(result.cdl))
        self.card_fc.set_value(self._format_hz(result.fc))
        self.card_rmse.set_value(self._format_ohm(result.rmse))

    @staticmethod
    def _format_ohm(value: float) -> str:
        if not math.isfinite(value):
            return "-"
        if abs(value) >= 1e6:
            return "%.3f M\u03A9" % (value / 1e6)
        if abs(value) >= 1e3:
            return "%.3f k\u03A9" % (value / 1e3)
        return "%.2f \u03A9" % value

    @staticmethod
    def _format_farad(value: float) -> str:
        if not math.isfinite(value):
            return "-"
        if value >= 1e-3:
            return "%.3f mF" % (value * 1e3)
        if value >= 1e-6:
            return "%.3f \u00B5F" % (value * 1e6)
        return "%.3f nF" % (value * 1e9)

    @staticmethod
    def _format_hz(value: float) -> str:
        if not math.isfinite(value):
            return "-"
        if value >= 1e3:
            return "%.3f kHz" % (value / 1e3)
        return "%.3f Hz" % value

    # -- plots ----------------------------------------------------------

    def _draw_all(self, result: FitResult):
        self._draw_nyquist(result)
        self._draw_bode_magnitude(result)
        self._draw_bode_phase(result)
        self._draw_residual(result)

    def _draw_nyquist(self, result: FitResult):
        axes = self.nyquist.axes
        self.nyquist.prepare()

        axes.scatter(result.z_real, -result.z_imag, s=18, color=TEAL,
                     zorder=3, label="Measurement")
        axes.plot(result.fit_real, -result.fit_imag, color=NAVY, linewidth=1.8,
                  zorder=2, label="Equivalent-circuit fitting")

        apex = int(np.argmax(-result.fit_imag))
        axes.scatter([result.fit_real[apex]], [-result.fit_imag[apex]],
                     marker="D", s=48, color=ACCENT, zorder=4,
                     label="Peak = %s" % self._format_hz(result.fc))

        axes.set_xlabel("Z' (\u03A9)")
        axes.set_ylabel("-Z'' (\u03A9)")
        axes.set_title("Nyquist Plot", color=TEXT_DARK)
        axes.set_aspect("equal", adjustable="datalim")
        axes.legend(loc="lower left", fontsize=8, framealpha=0.9)

        summary = ("Rs = %s\nRct = %s\nCdl = %s\nfc = %s"
                   % (self._format_ohm(result.rs), self._format_ohm(result.rct),
                      self._format_farad(result.cdl), self._format_hz(result.fc)))
        axes.text(0.98, 0.97, summary, transform=axes.transAxes,
                  ha="right", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                            edgecolor=BORDER))

        self.nyquist.figure.tight_layout()
        self.nyquist.draw_idle()

    def _draw_bode_magnitude(self, result: FitResult):
        axes = self.bode_mag.axes
        self.bode_mag.prepare()

        measured = np.hypot(result.z_real, result.z_imag)
        fitted = np.hypot(result.fit_real, result.fit_imag)

        axes.scatter(result.frequency, measured, s=16, color=TEAL, label="Measurement")
        axes.plot(result.frequency, fitted, color=NAVY, linewidth=1.8, label="Fitting")
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel("Frequency (Hz)")
        axes.set_ylabel("|Z| (\u03A9)")
        axes.set_title("Bode Magnitude", color=TEXT_DARK)
        axes.legend(fontsize=8)

        self.bode_mag.figure.tight_layout()
        self.bode_mag.draw_idle()

    def _draw_bode_phase(self, result: FitResult):
        axes = self.bode_phase.axes
        self.bode_phase.prepare()

        measured = np.degrees(np.arctan2(result.z_imag, result.z_real))
        fitted = np.degrees(np.arctan2(result.fit_imag, result.fit_real))

        axes.scatter(result.frequency, measured, s=16, color=TEAL, label="Measurement")
        axes.plot(result.frequency, fitted, color=NAVY, linewidth=1.8, label="Fitting")
        axes.set_xscale("log")
        axes.set_xlabel("Frequency (Hz)")
        axes.set_ylabel("Phase (degrees)")
        axes.set_title("Bode Phase", color=TEXT_DARK)
        axes.legend(fontsize=8)

        self.bode_phase.figure.tight_layout()
        self.bode_phase.draw_idle()

    def _draw_residual(self, result: FitResult):
        axes = self.residual.axes
        self.residual.prepare()

        axes.axhline(0.0, color=MUTED, linewidth=1.0)
        axes.scatter(result.frequency, result.z_real - result.fit_real,
                     s=16, color=TEAL, label="Z' residual")
        axes.scatter(result.frequency, result.z_imag - result.fit_imag,
                     s=16, color=ACCENT, marker="^", label="Z'' residual")
        axes.set_xscale("log")
        axes.set_xlabel("Frequency (Hz)")
        axes.set_ylabel("Measured minus fitted (\u03A9)")
        axes.set_title("Fitting Residuals", color=TEXT_DARK)
        axes.legend(fontsize=8)

        self.residual.figure.tight_layout()
        self.residual.draw_idle()

    # -- export ---------------------------------------------------------

    def export_results(self):
        if self._result is None:
            QMessageBox.information(
                self, "Export Results", "Run the fitting first.")
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the exported results")
        if not directory:
            return

        target = Path(directory)
        stem = Path(self._source_name).stem or "eis"
        result = self._result

        try:
            summary = target / ("%s_parameters.csv" % stem)
            with open(summary, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Parameter", "Value", "Unit"])
                writer.writerow(["Rs", "%.6f" % result.rs, "Ohm"])
                writer.writerow(["Rct", "%.6f" % result.rct, "Ohm"])
                writer.writerow(["Cdl", "%.9e" % result.cdl, "F"])
                writer.writerow(["fc", "%.6f" % result.fc, "Hz"])
                writer.writerow(["Complex RMSE", "%.6f" % result.rmse, "Ohm"])
                writer.writerow(["NRMSE", "%.4f" % result.nrmse, "%"])
                writer.writerow(["Points used", "%d" % result.frequency.size, ""])
                writer.writerow(["Source", self._source_name, ""])

            curve = target / ("%s_fitted.csv" % stem)
            with open(curve, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Freq [Hz]", "Zreal meas [Ohm]", "Zimag meas [Ohm]",
                                 "Zreal fit [Ohm]", "Zimag fit [Ohm]"])
                for row in zip(result.frequency, result.z_real, result.z_imag,
                               result.fit_real, result.fit_imag):
                    writer.writerow(["%.6f" % value for value in row])

            figures = []
            for canvas, name in ((self.nyquist, "nyquist"),
                                 (self.bode_mag, "bode_magnitude"),
                                 (self.bode_phase, "bode_phase"),
                                 (self.residual, "residual")):
                path = target / ("%s_%s.png" % (stem, name))
                canvas.figure.savefig(path, dpi=200, facecolor=CARD_BG)
                figures.append(path.name)

        except Exception as exc:
            QMessageBox.critical(self, "Export Results",
                                 "Could not write the results.\n\n%s" % exc)
            return

        QMessageBox.information(
            self, "Export Results",
            "Saved to:\n%s\n\n%s\n%s\n%s"
            % (target, summary.name, curve.name, ", ".join(figures)))
        self.statusBar().showMessage("Results exported to %s" % target)

    # -- appearance -----------------------------------------------------

    @staticmethod
    def _stylesheet() -> str:
        return """
        QWidget#central { background: %(bg)s; }
        QToolBar#topBar {
            background: %(navy)s; border: none; padding: 5px 10px; spacing: 6px;
        }
        QToolBar#topBar QToolButton {
            color: white; padding: 6px 14px; border-radius: 5px;
            font-size: 12px; font-weight: 600;
        }
        QToolBar#topBar QToolButton:hover { background: #1B4A7E; }
        QWidget#header { background: %(navy)s; }
        QLabel#appTitle { color: white; font-size: 21px; font-weight: 700; }
        QLabel#appSubtitle { color: #CDE6F3; font-size: 11px; }
        QFrame#logoCard { background: white; border-radius: 6px; }
        QLabel#logoFallback { color: %(navy)s; font-size: 10px; font-weight: 600; }
        QFrame#card, QFrame#metricCard {
            background: %(card)s; border: 1px solid %(border)s; border-radius: 8px;
        }
        QLabel#sectionLabel {
            color: %(navy)s; font-size: 12px; font-weight: 700; padding-bottom: 2px;
        }
        QLabel#fieldLabel { color: %(muted)s; font-size: 11px; }
        QLabel#hint { color: %(muted)s; font-size: 10px; }
        QLabel#mapName { color: %(muted)s; font-size: 11px; }
        QLabel#mapValue { color: %(navy)s; font-size: 11px; font-weight: 600; }
        QCheckBox#advancedCheck { color: %(muted)s; font-size: 10px; }
        QLabel#summaryText {
            color: %(navy)s; font-size: 10px; font-weight: 600;
            padding: 5px 7px; background: #EEF4F8; border-radius: 5px;
        }
        QLabel#sourceName { color: %(text)s; font-size: 12px; font-weight: 600; }
        QLabel#metricCaption { color: %(muted)s; font-size: 10px; }
        QLabel#metricValue { color: %(navy)s; font-size: 17px; font-weight: 700; }
        QLabel#circuitModel {
            color: %(navy)s; font-size: 15px; font-weight: 700; padding: 6px;
        }
        QComboBox, QLineEdit {
            background: white; border: 1px solid %(border)s; border-radius: 5px;
            padding: 5px 7px; font-size: 11px; color: %(text)s;
        }
        QComboBox:focus, QLineEdit:focus { border: 1px solid %(teal)s; }
        QPushButton {
            background: #EEF2F6; border: 1px solid %(border)s; border-radius: 5px;
            padding: 6px 10px; font-size: 11px; color: %(text)s;
        }
        QPushButton:hover { background: #E2E9F0; }
        QPushButton#primaryButton {
            background: %(teal)s; color: white; border: none;
            font-size: 12px; font-weight: 700;
        }
        QPushButton#primaryButton:hover { background: #00929D; }
        QTabWidget#tabs::pane {
            background: %(card)s; border: 1px solid %(border)s; border-radius: 8px;
        }
        QTabBar::tab {
            background: transparent; color: %(muted)s; padding: 7px 16px;
            font-size: 11px; border: none;
        }
        QTabBar::tab:selected {
            color: %(navy)s; font-weight: 700;
            border-bottom: 2px solid %(teal)s;
        }
        QTableWidget#preview {
            background: white; gridline-color: %(border)s;
            font-size: 11px; border: none;
        }
        QHeaderView::section {
            background: #EEF2F6; color: %(navy)s; padding: 5px;
            border: none; font-size: 11px; font-weight: 600;
        }
        QStatusBar { background: %(card)s; color: %(muted)s; font-size: 11px; }
        QScrollArea { background: transparent; }
        """ % {"bg": LIGHT_BG, "navy": NAVY, "card": CARD_BG, "border": BORDER,
               "muted": MUTED, "text": TEXT_DARK, "teal": TEAL}


def main() -> int:
    app = QApplication(sys.argv)
    window = EISWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
