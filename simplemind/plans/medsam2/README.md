# Biomechanics Video

Applies spatial reasoning to mask videos.

***

## Environment Setup

Activate the environment:
``` bash
micromamba activate smcore
```

Data: `~/example_data/biomech_video_masks/biomech_video_masks.csv`
* Symbolic link from `~/sm-incubator/example_data`

***

## Blackboard Messages

BB-1:
``` bash
python -m smcore.listen --addr bb-1.heph.com:8080
```

Cloud:
``` bash
python -m smcore.listen --addr public-bb.jmh.lol:8080
```

***

## Running the Plan

### Federated on BB-1
``` bash
python start_mind.py biomech_video_plan --addr bb-1.heph.com:8080
```
``` bash
python upload_listen.py --dataset_csv example_data/biomech_video_masks.csv --listen_tag heel_strike --addr bb-1.heph.com:8080
```
``` bash
python upload_dataset.py --dataset_csv example_data/biomech_video_masks.csv --addr bb-1.heph.com:8080
```
``` bash
rm -r working-*
```

### Federated Cloud
Cloud BB Restart:
``` bash
core control post bb-for-testing bb-command erase --addr public-bb.jmh.lol:8080
```
SM Team:
``` bash
python start_mind.py biomech_video_plan --addr bb-1.heph.com:8080 --io_addr public-bb.jmh.lol:8080 --io_input_tags biomech_video_mask image file --io_output_tags heel_strike
```
Video Team:
``` bash
python upload_listen.py --dataset_csv example_data/biomech_video_masks.csv --addr public-bb.jmh.lol:8080 --listen_tag heel_strike
```

### Unfederated
``` bash
python run_plan.py biomech_video_plan --dataset_csv example_data/biomech_video_masks.csv --addr bb-1.heph.com:8080
```


***

## Git Commit
``` bash
git add -u
git status
git commit -m"updating sm pipeline"
git push
```
