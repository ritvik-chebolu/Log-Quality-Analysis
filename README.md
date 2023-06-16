# Linguistic Quality Analysis

This is a project for performing linguistic quality analysis on log messages. The project involves preprocessing the data, performing one-hot encoding, feature selection, and building a neural network model for predicting the quality of the log messages.

## Getting Started
These instructions will help you to set up and run this project on your local machine.  

Run the [final_model.py](https://github.com/ritvik-chebolu/Log-Quality-Analysis/blob/main/code/final_model.py) and the [my_model.py](https://github.com/ritvik-chebolu/Log-Quality-Analysis/blob/main/code/my_model.py) using Google Colab for our the final script.

We tried our best to insert as much as possible information into the main manuscript. 


# QuLog
In the folder **code** can be found the evaluation scripts used to obtain the results. Alongside, given are the exact predictions.

All the data we collected and preprocessed can be found in the **[data](https://github.com/ritvik-chebolu/Log-Quality-Analysis/tree/main/data)** folder. 
Notably, due to proprietary issues, we do not disclose the data on the internal systems. 

To start with the code clone this GitHub repo: 

1) git clone https://github.com/DSCI-644/project-dsci-644-02-group-9-ritvik-disha-deepika.git
2) create a virtual enviorment python3 -m venv venv
3) Install requirements: python -m pip install -r requirments.txt
4) To check QuLog log level navigate to: ./code/level_quality/qulog_attention_nn_type1/qulog_attention_nn_type1/ (This folder contains the model and the classes)
6) Run: python3 qulog_attention_nn_type1.py; to check the log level for an example log message. You can modify line 120 for arbitrary log instruction. 
7) To check QuLog sufficient linguistic structure navigate to: .code/ling_quality/qulog_attention_nn_type1/qulog_attention_nn_type1/ (This folder contains the model and the classes)
8) Run: python3 qulog_attention_nn_type1.py; to check the log level for an example log message. You can modify line 327 for arbitrary log instruction.

Additionally in "./code/training_scripts/" one can find the training model scripts. Note on training: After preparing the datasets, the methods can be accessed from each script. One need to modify the paths for the correct directories where the data is located. 

Spacy note: Additionally, after installation of spacy make sure it is properly installed. If training the scripts, make sure that you have installed:
"en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0-py3-none-any.whl"


    |code 
    |--- my_model.py
    |--- final_model.py
    |--- level_quality
    |------ qulog_attention_nn_type1
    |------ qulog_svc
    |------ level_qulog_sm_rf
    |--- ling_quality
    |------ qulog_attention_nn_type1
    |------ qulog_rf
    |------ qulog_sm_svc
    |--- training_scirpts
    |------ level
    |--------- qulog_svc
    |--------- qulog_sm_rf
    |--------- qulog_attention
    |------ ling
    |--------- qulog_svc
    |--------- qulog_sm_rf
    |--------- qulog_attention
    |data 
    |--- github_repos_data
    |--- linguistic_quality_inter.csv
    |--- nine_systems_data.csv
    |--- stars_repos.csv
    |requirments.txt
    |README.md

For the comments that were lacking space, we inserted them as supplementary material. Following the recommendations from the meta-reviewers, we upload a file: [Meta_Review_Comments_Addressed_SupplementaryMaterial_ICPC_QuLog.pdf](https://github.com/ritvik-chebolu/Log-Quality-Analysis/blob/main/Meta_Review_Comments_Addressed_SupplementaryMaterial_ICPC_QuLog.pdf). It addresses part of the comments raised by the meta reviewers. 

## Authors
* **Ritvik Chebolu** - [ritvik-chebolu](https://github.com/rtvik-chebolu)
* **Disha Shah** - [disha9896](https://github.com/disha9896)

** We tried to replicate and improve the results of the paper [QuLog: Data-Driven Approach for Log Instruction Quality Assessment](https://ieeexplore.ieee.org/document/9796196) and succeeded, which was reflected by an increased accuracy in the model performance after optimization.
