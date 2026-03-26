# SimpleMind adds thinking to deep neural networks

<!-- ![SM_logo_v0](docs/figure/SIMPLEMIND_logo_multi_gradient_side_text_smaller.png) -->

<img src="docs/figure/SIMPLEMIND_logo_multi_gradient_side_text_smaller.png" alt="logo" width="30%"/>

** December 12, 2025: Welcome to SimpleMind to the UCLA PBMED 210 class. For support, please email m w ahian war @ med net . ucla  . edu (remove all spaces). We're here to help!

---

**SimpleMind (SM)** provides computer vision tools that can be connected into pipelines using a JSON plan. 
* Tools receive input messages, process data, and send output messages. 
* Plans can be created and refined collaboratively by humans and AI. 

If you know basic **Python** and **JSON**, you can learn SimpleMind in about **90 minutes** using the instructions and exercises below.

***

## 1. Environment Setup

### Set up vscode

We recommend using [Visual Studio Code](https://code.visualstudio.com) for running SimpleMind and viewing outputs.
* If connecting to a GPU server, see [vscode connection instructions](docs/vscode.md).
* Use `Terminal > New Terminal` to open a **bash terminal** on either your local computer or GPU server.

### Clone the repository
``` bash
git clone git@gitlab.com:sm-ai-team/simplemind.git
cd simplemind
git checkout mbrown/new_sm
```

### Micromamba

Create or update the environment:
``` bash
micromamba create -f env.yaml
```
Activate the environment:
``` bash
micromamba activate smcore
```

### Install Core
To install or update to a new version:
``` bash
go install gitlab.com/hoffman-lab/core@v1.1.1
echo 'export PATH="$PATH:$HOME/go/bin"' >> ~/.bashrc
```

### Data download

Perform a one-time setup of data and weights by following [data setup instructions](docs/data_download.md).

**Note:** Unless otherwise specified, run all commands inside the micromamba environment from the `simplemind` directory.

***

## 2. Blackboard Service

Tools exchange messages via a **Blackboard (BB)**.

Start a local Core BB:
``` bash
core start server
```

Leave this terminal running:
* You will see BB messages appear when running plans.
* Open a new terminal for subsequent commands.
* The address for your local BB is `127.0.0.1:8080`.

If you are on the [gpu-1 server](docs/gpu1.md), there are additional instructions to access the BB service.

***

## 3. Running a Plan

**Thinking** (inference) example in SM: segment the trachea and lungs on chest x-rays. 

``` bash
python run_plan.py plans/cxr --dataset_csv data/cxr_images.csv --gpu_num 0 --addr 127.0.0.1:8080
```
* `run_plan.py` is a [**controller**](docs/controller.md) 
* It displays a [**dashboard**](docs/dashboard.md) for the run
* Directs terminal output to `working-<id>/stdout-<id>.log` and `working-<id>/stderr-<id>.log`
* Tool outputs for case 0: `working-<id>/output-<id>/samples/0/`
    * `trachea_overlay.png` and `lungs_overlay.png` are the final segmented masks
    * They can be opened in vscode

Clean up:
``` bash
rm -r working-*
```

***

## 4. Tutorials

Work through these exercises to get hands-on with SimpleMind:

[Exercise 1](docs/exercise1.md): Plans and tools for inference (*Thinking*)<br>

[Exercise 2](docs/exercise2.md): Developing a tool<br>

[Exercise 3](docs/exercise3.md): Decision trees<br>

[Exercise 4](docs/exercise4.md): Training a tool (*Learning*)<br>

***

## 5. How To

[Docker Environment](docs/docker.md): User a Docker environment (instead of micromamba).<br>
[Input Parameters](docs/input_parameters.md): Specify an input parameter using "from".<br>
[Process Cleanup](docs/process_cleanup.md): Clean up tool processes.<br>
[Tool Environments](docs/tool_envs.md): Create a tool that runs in its own environment.<br>
[Tune PyTorch](docs/tune_pytorch.md): Tune PyTorch learning parameters for segmentation.<br>


***

## 6. Tools

Tool documentation: [tools/README.md](tools/README.md)

To update docs:
``` bash
python tool_doc.py
git add tools/README.md
git commit -m "update tools/README.md"
git push
```

***

## 7. Git Commit
``` bash
git add -u
git status
git commit -m"updating sm pipeline"
git push
```

***

## License

SimpleMind is licensed under the 3-clause BSD license. For details, please see the [LICENSE](./LICENSE) file.

If you use this software in your research or project, please cite our [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283587):
``` code
Choi Y, Wahi-Anwar MW, Brown MS. SimpleMind: An open-source software environment that adds thinking to deep neural networks. PLoS One. 2023 Apr 13;18(4):e0283587. doi: 10.1371/journal.pone.0283587. PMID: 37053159; PMCID: PMC10101376.
```

***

## About Us

SimpleMind is a [Cognitive AI](docs/cognitive_ai.md) environment for combining human knowledge and reasoning with deep neural networks. 

While our core applications are in medical imaging, SimpleMind is designed to be general-purpose. Our vision is for the community to expand it toward the broader goal of **human-level intelligence and beyond**. 
