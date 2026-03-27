---
layout: page
title: Getting Started
permalink: /getting-started/
---

# Getting Started with SimpleMind

This guide will help you set up SimpleMind and run your first AI pipeline in under 30 minutes.

## Prerequisites

- **Python 3.8+** with basic knowledge
- **JSON** understanding
- **Git** for version control
- **Visual Studio Code** (recommended)

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone git@gitlab.com/sm-ai-team/simplemind.git
cd simplemind
git checkout mbrown/new_sm
```

### Step 2: Environment Setup

We recommend using **micromamba** for environment management:

```bash
# Create the environment
micromamba create -f env.yaml

# Activate the environment
micromamba activate smcore
```

### Step 3: Install Core

SimpleMind uses the Core blackboard service for tool communication:

```bash
go install gitlab.com/hoffman-lab/core@v1.1.1
echo 'export PATH="$PATH:$HOME/go/bin"' >> ~/.bashrc
```

### Step 4: Data Setup

Download the required data and model weights:

```bash
# Follow the data download instructions
python gdownload_data.py
```

## 🚀 Your First Pipeline

### Start the Blackboard Service

Open a terminal and start the Core blackboard:

```bash
core start server
```

**Keep this terminal running** - you'll see messages appear when running plans.

### Run a Medical Imaging Example

Open a new terminal and run the chest X-ray segmentation pipeline:

```bash
python run_plan.py plans/cxr --dataset_csv data/cxr_images.csv --gpu_num 0 --addr 127.0.0.1:8080
```

This command:
- Loads the chest X-ray segmentation plan
- Processes images from the CSV dataset
- Uses GPU 0 for acceleration
- Connects to the local blackboard service

### View Results

Results are saved in the `working-<id>/output-<id>/samples/0/` directory:
- `trachea_overlay.png` - Segmented trachea mask
- `lungs_overlay.png` - Segmented lung mask

Open these files in VS Code to see the segmentation results!

## 📊 Understanding the Dashboard

The controller displays a real-time dashboard showing:

- **Tool Status** - Which tools are active/inactive
- **Sample Progress** - Current processing status
- **Message Flow** - Communication between tools

### Dashboard Symbols

| Symbol | Meaning |
|--------|---------|
| **-** | Tool has started |
| **S / M** | S = highest consecutive sample processed, M = max samples |
| **S* / M** | Sample S is being processed |
| **a / a** | Aggregate method processing/completed |

## 🧹 Cleanup

After running your pipeline:

```bash
# Remove working directories
rm -r working-*

# Stop the blackboard service (Ctrl+C in the terminal)
```

## 🎯 What's Next?

Now that you have SimpleMind running, explore these tutorials:

1. **[Exercise 1](../simplemind/docs/exercise1.html)** - Understanding plans and tools
2. **[Exercise 2](../simplemind/docs/exercise2.html)** - Developing your own tool
3. **[Exercise 3](../simplemind/docs/exercise3.html)** - Adding decision logic
4. **[Exercise 4](../simplemind/docs/exercise4.html)** - Training tools with data

## 🔧 Development Setup

### VS Code Configuration

For the best development experience:

1. Install the **Python** extension
2. Set up **remote development** if using a GPU server
3. Configure the **integrated terminal** for bash

### GPU Server Connection

If you're using a remote GPU server, see our [VS Code connection guide](../simplemind/docs/vscode.html).

## 🆘 Troubleshooting

### Common Issues

**Environment activation fails:**
```bash
# Make sure micromamba is installed
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```

**Core installation fails:**
```bash
# Ensure Go is installed
go version
# If not installed, visit: https://golang.org/doc/install
```

**GPU not detected:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help

- 📧 **Email Support**: Contact the SimpleMind team
- 📚 **Documentation**: Check our comprehensive guides
- 🐛 **Issues**: Report problems on the repository

---

Ready to dive deeper? Continue with our [tutorial series](tutorials.html) or explore the [full documentation](documentation.html).