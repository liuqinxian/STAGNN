# Requirements
`pip install -r requirements.txt`
# Data
## Multi-step forecasting
PeMSD7(M) saved in folder `/PeMS`, and PeMS_BAY saved in folder `/PeMS_BAY`.
## Single-step forecasting
Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress them and move them to the data folder.
# Train
## Multi-step forecasting
`bash scripts/multi.sh` 
## Single-step forecasting
`bash scripts/single.sh` 