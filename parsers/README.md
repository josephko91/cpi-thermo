
# Data Provenance: Field Campaign Parsers

This folder contains Python modules for parsing and standardizing environmental and positional data from multiple atmospheric field campaigns. Each parser is tailored to the unique data formats and metadata conventions of its respective campaign.

## Field Campaigns and Data Sources

- **ARM (Atmospheric Radiation Measurement):**
  - Southern Great Plains (SGP) 2000 Spring Cloud campaign.
  - [ARM Data Center](https://www.arm.gov/research/campaigns/sgp2000sprcloud)
  - Citation aircraft archive files (`*.t4archive.gz`).
  - [Direct data access](https://www.archive.arm.gov/data/sgp2000sprcloud/citation/)

- **CRYSTAL-FACE (NASA & UND):**
  - Cirrus Regional Study of Tropical Anvils and Cirrus Layers – Florida Area Cirrus Experiment.
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/crystal_face)
  - [UND Data](https://www.atmos.und.edu/und_cloud_aerosol/crystalface.html)
  - Includes MIS.CIT files and other formats.

- **MC3E (Midlatitude Continental Convective Clouds Experiment):**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/mc3e)
  - Environmental and positional measurements.

- **MIDCIX:**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/midcix)
  - Environmental and positional measurements.

- **OLYMPEX:**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/olympex)
  - Environmental and positional measurements.

- **AIRS-II:**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/airs_ii)
  - netCDF files (`*.nc`).

- **ATTREX:**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/attrex)
  - NASA ICARTT files (`*.ict`).

- **IPHEX (Integrated Precipitation and Hydrology Experiment):**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/iphex)
  - Custom `.iphex` files.

- **ISDAC (Indirect and Semi-Direct Aerosol Campaign):**
  - [ARM Data Center](https://www.arm.gov/research/campaigns/isdac2008)
  - Comma-delimited text files.
  - [Direct data access](https://www.archive.arm.gov/data/isdac2008/strapp-convair_bulk/CommaDelimited/)

- **POSIDON (Profiling of Winter Storms):**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/posidon)
  - NASA ICARTT files (`*.ict`).

- **ESCAPE (Earth Science Campaigns for Aerosol Profiling Experiment):**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/escape)
  - Learjet state measurements, NASA ICARTT-like CSV text files (`*.ict`).

- **ICE-L (Ice in Clouds Experiment - Layer clouds):**
  - [NCAR Data Archive](https://data.eol.ucar.edu/dataset/346.1)
  - netCDF files (`*.PNI.nc`).

- **MACPEX (Mid-latitude Airborne Cirrus Properties Experiment):**
  - [NASA ESPO Archive](https://espoarchive.nasa.gov/archive/browse/macpex)
  - NASA ICARTT files (`*.ict`).

## Data Processing and Standardization

Each parser:
- Reads raw data files from the campaign's official archive or data center.
- Extracts standardized columns: timestamp, temperature, supersaturation (Si), latitude, longitude, altitude, and campaign metadata.
- Handles campaign-specific quirks, file formats, and metadata conventions.

## Provenance and Reproducibility
- All data sources are referenced in the campaign-specific parser docstrings.
- Processing steps are documented in the main script and notebooks.
