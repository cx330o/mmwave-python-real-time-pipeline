# CX330 Radar (WIP)

Work in progress. I'm building a mmWave radar processing/visualization toolkit.
- Device: TI IWR6843 (and related)
- Language: Python
- Status: 🚧 Initial repo setup. Code & docs are being organized.

> Raw recordings, firmware binaries, and vendor libs are intentionally excluded.

## Offline processing (DCA1000)

- Use `convert_raw_to_npy.py` to convert raw ADC `.bin` files recorded by DCA1000 into NumPy arrays (`.npy`).
- The notebook `offline_dca1000_processing.ipynb` shows how to load the converted data and run the same range–Doppler / range–angle processing as in the real-time GUI.
- Raw `.bin` recordings are kept locally in the `recordings/` directory and are not pushed to GitHub.
