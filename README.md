# CPI-Thermo: Cloud particle imagery with thermodynamics

## Overview
This repository provides tools and scripts to create a unified dataset that merges CPI imagery with co-located thermodynamic mesurements. It includes modular data parsers for multiple field campaigns, a main processing script, and configuration files for reproducible research.

## Features
- Modular parsers for various airborne field campaigns
- Integration of cloud particle imagery with thermodynamic measurements
- Configurable processing pipeline
- Easily extensible for new campaigns or data sources

## Directory Structure
```
├── main.py                # Main entry point for processing
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── parsers/               # Campaign-specific and utility parsers
│   ├── __init__.py
│   ├── ...
├── .gitignore             # Files and folders to ignore in git
├── LICENSE                # License information
```

## Getting Started
1. **Clone the repository:**
   ```
   git clone https://github.com/<your-username>/cpi-thermo.git
   cd cpi-thermo
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Configure your run:**
   - Edit `config.yaml` as needed for your data and environment.
4. **Run the main script:**
   ```
   python main.py
   ```

## Adding New Campaign Parsers
- Add a new Python file in the `parsers/` directory following the existing parser structure.
- Update `main.py` or configuration as needed to include the new parser.

## License
This project is licensed under the terms of the LICENSE file.

## Acknowledgments
- todo

## Contact
For questions or contributions, please open an issue or submit a pull request. Or email <jk4730@columbia.edu>
