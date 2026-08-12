# Harmonized model-ready data

The four CSV files are the frozen inputs used by the revision experiment pipeline. Every file has the same schema:

`Date, Holi, Temp, Humi, WS, Consumption`

The model code recomputes all calendar cycles, THI/WCT, nonlinear thermal terms, safe daily/weekly demand context, historical-only weather proxies, and the seasonal anchor from this common base schema. Precomputed derived columns from older versions of the datasets are intentionally not consumed by the training pipeline.
