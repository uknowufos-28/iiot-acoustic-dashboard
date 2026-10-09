# Project documentation

- [Raspberry Pi hybrid model setup guide](raspberry-pi/README.md) — software setup, secure API configuration, GitHub Pages connection, CSV testing, and the remaining sensor integration work.
- [Download the Raspberry Pi setup guide (PDF)](raspberry-pi/Raspberry_Pi_Hybrid_Model_Setup_Guide.pdf)

The repository groups Python code by purpose in `acoustic_model/`, `training/`, and `raspberry_pi/`. The full project map and run commands are in the [main README](../README.md). This folder contains operator-facing setup documentation and the PDF build helper.

Regenerate the PDF after editing its Markdown source with:

```bash
python -m pip install reportlab
python docs/raspberry-pi/build_guide_pdf.py
```
