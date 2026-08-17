# EIS Equivalent-Circuit Fitting

Desktop analysis tool for impedance spectra measured by the Raspberry Pi
instrument. Reads a CSV, fits the equivalent circuit Rs + (Rct || Cdl), and
reports the fitted parameters with Nyquist, Bode and residual plots.

## Setup

From this folder, with the virtual environment active:

```
pip install PySide6 numpy scipy matplotlib
python eis_gui.py
```

No other packages are required.

## Using it

Drag a CSV onto the window, or press Open CSV. Press Demo Data to load a
synthetic spectrum generated from the reference circuit, which is useful for
checking the interface without a measurement file.

Press Run Fitting. The five cards along the top show the fitted parameters,
and the tabs show the Nyquist plot, both Bode plots, the residuals, and the
raw table.

Export Results writes a parameter summary, the measured and fitted curves,
and all four plots as PNG files into a folder you choose.

## Units

The unit multiplier is detected from the column header rather than typed.
A header containing `kOhm` sets it to 1000, `MOhm` sets it to 1000000, and
anything else is treated as ohms.

This matters: files exported by the instrument are already in ohms, while the
original analyser export used kilohms. Applying the wrong factor shifts every
fitted resistance by a thousand without anything visibly failing. The detected
value is shown in the sidebar and can be overridden.

## Fitting range

Minimum and maximum frequency default to the range the loaded file covers, so
a valid file never produces an empty selection. Narrow them to fit only part
of the spectrum. Leaving a box blank removes that limit.

Fitted Rs is sensitive to the upper limit, because Rs is what the impedance
approaches at high frequency and the recorded values there are coarse. Rct,
Cdl and fc are far more stable.

## Files it accepts

Any CSV with a frequency column and real and imaginary impedance columns. The
columns are matched by name and can be changed in the sidebar. Both the
instrument export and the original analyser export are recognised.
