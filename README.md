# Implementation of OOCD framework

This is the anonymized github repository for the submission, entitled **"Out-of-Category Document Identification Using Target-Category Names as Weak Supervision"**.
For reproducibility, the codes and toy datasets are publicly available during the review phase.


## Run the codes

- sklearn
- python
- torch (GPU version only)

First, download the pretrained word vectors from [this link](https://www.dropbox.com/s/vvv8p4wzgelf5fo/pretrained_wiki_w_emb.zip?dl=0).
The pretrained embedding weights are obtained by using a large-scale wiki text.
Then, place the .zip file in the `./ouput` directory and unzip it as follows.
```
cd ./output
unzip pretrained_wiki_w_emb.zip
cd ../
```

You can simply run the code with the default setting by the following command:
```
python main.py
```

Target categories can be flexibly designated by `data/<dataset-name>/<target-name>_targets.txt`:
```
python main.py --dataset <dataset-name> --target <target-name> --gpuidx <gpu-device-idx>
```
