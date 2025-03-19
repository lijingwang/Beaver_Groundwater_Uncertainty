# Quantifying Groundwater Response and Uncertainty in Beaver-Influenced Mountainous Floodplains Using Machine Learning-Based Model Calibration

This repository contains the data and code associated with the manuscript titled **"Quantifying Groundwater Response and Uncertainty in Beaver-Influenced Mountainous Floodplains Using Machine Learning-Based Model Calibration"**, submitted to *Water Resources Research*.

## Corresponding Author
- **Lijing Wang**  
  Department of Earth Sciences, University of Connecticut  
  Email: lijing.wang@uconn.edu

  Earth and Environmental Sciences Area, Lawrence Berkeley National Laboratory  
  Email: lijingwang@lbl.gov

## Contents
- **/data/**: Includes characterization data (geophysics, geological data, DEM) and response data (hydrologic measurements).
- **/model/**: Contains the floodplain structure model derived from geophysical and geological data.
- **/scripts/**: Python code for hydrologic modeling using Flopy and MODFLOW for different hydrological periods (baseflow, snowmelt, dry ponds).
- **/notebooks/**: Five Jupyter notebooks covering:
  1. Floodplain structure reconstruction (Section 4.1)
  2. Hydrologic measurement visualization (Section 4.2)
  3. Prior simulations (Set 1 and Set 3)
  4. Model calibration using a neural density estimator (Section 4.3)
  5. Local and regional groundwater responses, and sensitivity analysis (Section 4.4, Section 5.1)

## License
This repository is shared under the [MIT License](https://opensource.org/licenses/MIT), allowing free use, modification, and distribution with attribution to the original author.