# Flow Dataset and DNN
Dependencies: Python version: 3.6+ (for used packages see [utils.py](utils.py))

## Desciption and Usage

The Flow Dataset and DNN module implements the Aggregation and Normalization step of the Data Preparation as well as the Data Processing stage.

![Flow Data Pipeline](/images/flow_data_pipeline.png)

The process can be started with `python3 main.py`.
In the first run, the dataset is created and subsequent runs perform training and testing.

## Create Dataset
The output of the [Anonymizer](https://gitlab.cs.hs-fulda.de/flow-data-ml/icann19/anonymizer) module is used as input for the dataset creation.
Therefore, the raw dataset file `*.flows.pkl.gz` must be stored in `./datasets/` and specified in [/dataset/hs_flows.py](/dataset/hs_flows.py) (see variable `DATASET`).

## Training and Testing Parameters
All experiments can be configured by command line parameters (see [defaultParser.py](defaultParser.py)).

## Internal Structure

The current workflow (Data Collection, Data Preparation and Data Processing) of the flow data pipeline is as follows:

![Internal_structure](/images/Internal_structure.png)

1. Flow data is exported from network devices.
2. Flows are stored by a flow collector (OpenNMS) in a database (Elasticsearch).
3. Flow data is enriched with internal and external metadata and anonymized (everything up to here is processed in the data center).
4. The "raw dataset" is converted into a dataset suitable for DNNs.
5. The dataset is used as input for DNN training and testing.

## Paper Reference
A Study of Deep Learning for Network Traffic Data Forecasting, ICANN 2019