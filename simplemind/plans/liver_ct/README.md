# Liver CT

Tests the TotalSegmentator tool.

***

## Environment Setup

Activate the environment:
``` bash
micromamba activate smcore
```

Download data (once):
``` bash
python gdownload_data.py data https://drive.google.com/file/d/1NxzGgWCzEsVERuCmOoY0QcZrk0k7lMPm/view?usp=sharing
```

***

## Blackboard Messages

BB-1:
``` bash
python -m smcore.listen --addr bb-1.heph.com:8080
```

***

## Running the Plan

``` bash
python run_plan.py plans/liver_ct --dataset_csv data/liver_ct_images.csv --addr bb-1.heph.com:8080
```

***

## Git Commit
``` bash
git add -u
git status
git commit -m"updating sm pipeline"
git push
```
