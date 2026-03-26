# Data and Weights Setup

Before running the examples and tutorials, you’ll need to set up the example **data** and **weights**. This is a one-time setup step.

The `data/` and `weights/` folders will be ignored by git.

***

## PBMED 210 Class

Create a symbolic link to data:
``` bash
ln -s /home/mbrown/data data
```

To download example weights, run the following once in the `simplemind` directory:
``` bash
python gdownload_data.py weights/cxr https://drive.google.com/file/d/1_BW4sPyCkuiisRFjL_tLMiAmTy76vNB-/view?usp=sharing
python gdownload_data.py weights/cxr https://drive.google.com/file/d/1gqbywicxKoMstYcymk20hL9TlgTrRMpQ/view?usp=sharing
```

***

## General Users

To download, run the following once in the `simplemind` directory:
### *Thinking* example
``` bash
python gdownload_data.py data https://drive.google.com/file/d/1M2gyesM7qalW9lF8ozDNodGGRLOIZAQm/view?usp=sharing
python gdownload_data.py weights/cxr https://drive.google.com/file/d/1_BW4sPyCkuiisRFjL_tLMiAmTy76vNB-/view?usp=sharing
python gdownload_data.py weights/cxr https://drive.google.com/file/d/1gqbywicxKoMstYcymk20hL9TlgTrRMpQ/view?usp=sharing
```

### _Learning_ example and tutorials
``` bash
python gdownload_data.py data https://drive.google.com/file/d/1y5eR9I-f5KCgnkSt7yxd1VrrVQfTDv4L/view?usp=sharing
python gdownload_data.py data https://drive.google.com/file/d/1kTdErSwH1c5RL1f1vQD5g6DvpMLCT3UJ/view?usp=sharing
python gdownload_data.py data https://drive.google.com/file/d/1Am4jFDlkBrVGFwuCsWRfwGsXubuN-NPx/view?usp=sharing
```

***

## Dev Team: Uploading Data and Weights to Google Drive

* Log in to the **simplemindai** Gmail account in Chrome.
* The Google Drive has `data` and `weights` folders.

### Upload a dataset

If your dataset CSV and folder are formatted as described in [`upload_dataset.py`](../upload_dataset.py), zip them:
```bash
zip -r zip_data_name.zip data_folder csv_name.csv
# zip -r cxr_images.zip cxr_images cxr_images.csv
```

Upload `zip_data_name.zip` to the google drive `data` folder (under the simplemind account). Share with “Anyone with link” and copy the link. This becomes the argument to [gdownload_data.py](../gdownload_data.py).

### Upload weights

```bash
zip -r zip_weights_name.zip weights.pth
# zip -r right_lung_weights.zip right_lung_weights.pth
```
Upload `zip_weights_name.zip` to the `weights` folder.
