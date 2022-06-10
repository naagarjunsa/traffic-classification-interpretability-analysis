# traffic-classification-interpretability-analysis

This project focuses on analysis of Machine Learning Solutions for Network Traffic Classification with focus on interpretability of the solution as well as [nPrintML](https://nprint.github.io/) approach for data representation for network packets.

### Steps to Run this project

#### Data Processing

We picked two network traffic classification applications namely Country of Origin Classfication Task and the Active Device Identification Task nPrintML dataset avaialable [here](https://nprint.github.io/datasets.html).

To preprocess the dataset available we used the nPrintML CLI tools pcapML and nPrint wrapped in a python script to scale it to the entire dataset. 

We add the metdata.csv file in the pcapML extraction to the nprint dataset and use it to load the data into memory. 


#### Model Building

We used the interpretability focused ML framework called [InterpretML](https://github.com/interpretml/interpret) This model helped us build the local and global interpretability and the source code for this can be found in src/interpretml

We used [Neural Additive Models](https://neural-additive-models.github.io) pytorch based implementation by [AmrMKayid]( https://github.com/AmrMKayid/nam). 

The notebooks contain the implementation and steps to run the experiments. 


