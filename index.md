---
layout: default
title: SimpleMind - Adding Thinking to Deep Neural Networks
description: A Cognitive AI environment for combining human knowledge and reasoning with deep neural networks
---

# SimpleMind
## Adding Thinking to Deep Neural Networks

<img src="simplemind/docs/figure/SIMPLEMIND_logo_multi_gradient_side_text_smaller.png" alt="SimpleMind Logo" width="300"/>

**SimpleMind (SM)** is a Cognitive AI environment that provides computer vision tools connected into pipelines using JSON plans. It combines human knowledge and reasoning with deep neural networks to create more reliable and transparent AI systems.

---

## 🚀 Quick Start

SimpleMind allows you to create AI pipelines that can **think** and **learn**:
- **Tools** receive input messages, process data, and send output messages
- **Plans** can be created and refined collaboratively by humans and AI
- Learn SimpleMind in about **90 minutes** with basic Python and JSON knowledge

### Key Features

- 🧠 **Cognitive AI Architecture** - Human-like thinking combined with neural networks
- 🔧 **Modular Tools** - Connect computer vision tools into complex pipelines
- 📋 **JSON Plans** - Easy-to-understand pipeline definitions
- 🎯 **Few-Shot Learning** - Teach systems with minimal examples
- 🏥 **Medical Imaging Focus** - Specialized for healthcare applications
- 🌐 **General Purpose** - Extensible to other domains

---

## 📸 Screenshots & Examples

### Pipeline Dashboard
*Real-time monitoring of your AI pipeline execution*

![Dashboard Screenshot](assets/images/dashboard-screenshot.png)
*The SimpleMind dashboard shows real-time status of pipeline runs, tool execution, and sample processing*

### Medical Imaging Example
*Chest X-ray segmentation pipeline in action*

![CXR Segmentation](assets/images/cxr-segmentation-example.png)
*Example output: Trachea and lung segmentation on chest X-rays using SimpleMind's thinking pipeline*

### Tool Development
*Visual Studio Code integration for tool development*

![Tool Development](assets/images/tool-development-screenshot.png)
*Developing and testing SimpleMind tools with full IDE support*

---

## 🏗️ Architecture

SimpleMind uses a **Blackboard Architecture** where:

1. **Tools** communicate through a central message board
2. **Controllers** orchestrate pipeline execution
3. **Plans** define the workflow in JSON format
4. **Data** flows through the system with full traceability

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Tool A   │    │    Tool B   │    │    Tool C   │
│ (Preprocess)│    │  (Segment)  │    │(Postprocess)│
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                   ┌──────▼──────┐
                   │  Blackboard │
                   │   Service   │
                   └─────────────┘
```

---

## 🎯 Use Cases

### Medical Imaging
- **Chest X-ray Analysis** - Automated lung and trachea segmentation
- **OCT Imaging** - Corneal analysis and measurement
- **Multi-modal Processing** - Combine different imaging modalities

### Research & Development
- **Rapid Prototyping** - Quick pipeline development and testing
- **Collaborative AI** - Human-AI collaboration in model development
- **Reproducible Research** - Version-controlled pipelines and results

---

## 📚 Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone git@gitlab.com/sm-ai-team/simplemind.git
cd simplemind

# Create environment
micromamba create -f env.yaml
micromamba activate smcore

# Install Core
go install gitlab.com/hoffman-lab/core@v1.1.1
```

### 2. Start the Blackboard Service
```bash
core start server
```

### 3. Run Your First Pipeline
```bash
python run_plan.py plans/cxr --dataset_csv data/cxr_images.csv --gpu_num 0
```

---

## 📖 Documentation

### Tutorials
- [Exercise 1: Plans and Tools](simplemind/docs/exercise1.html) - Basic inference (*Thinking*)
- [Exercise 2: Tool Development](simplemind/docs/exercise2.html) - Create your own tools
- [Exercise 3: Decision Trees](simplemind/docs/exercise3.html) - Add logic to pipelines
- [Exercise 4: Training](simplemind/docs/exercise4.html) - Tool training (*Learning*)

### Technical Guides
- [Controller Documentation](simplemind/docs/controller.html) - Pipeline orchestration
- [Dashboard Guide](simplemind/docs/dashboard.html) - Monitoring and debugging
- [Docker Setup](simplemind/docs/docker.html) - Containerized deployment
- [GPU Configuration](simplemind/docs/gpu1.html) - GPU server setup

---

## 🔬 Research & Publications

SimpleMind is based on peer-reviewed research in Cognitive AI:

> **Citation:** Choi Y, Wahi-Anwar MW, Brown MS. SimpleMind: An open-source software environment that adds thinking to deep neural networks. *PLoS One*. 2023 Apr 13;18(4):e0283587. 
> 
> [📄 Read the Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283587)

### Key Research Areas
- **Cognitive AI Systems** - Human-like intelligence in machines
- **Hybrid Neural-Symbolic AI** - Combining deep learning with reasoning
- **Medical AI Deployment** - Addressing the Health AI Paradox
- **Few-Shot Learning** - Teaching with minimal examples

---

## 🤝 Community & Support

### Get Help
- 📧 **Email Support**: For UCLA PBMED 210 class support
- 📚 **Documentation**: Comprehensive guides and tutorials
- 🐛 **Issues**: Report bugs and request features
- 💬 **Discussions**: Community Q&A and sharing

### Contributing
SimpleMind is open-source and welcomes contributions:
- 🔧 **Tool Development** - Create new processing tools
- 📖 **Documentation** - Improve guides and examples
- 🧪 **Testing** - Help validate new features
- 🎨 **Examples** - Share your use cases

---

## 📄 License

SimpleMind is licensed under the **3-clause BSD license**, making it free for both academic and commercial use.

---

## 🌟 Vision

*"Our vision is for the community to expand SimpleMind toward the broader goal of **human-level intelligence and beyond**."*

SimpleMind represents a step toward Cognitive AI that:
- Thinks like humans do
- Learns from few examples
- Explains its reasoning
- Collaborates naturally with people
- Scales to real-world applications

---

<div class="text-center">
  <a href="https://github.com/yourusername/simplemind" class="btn btn-primary">View on GitHub</a>
  <a href="simplemind/docs/exercise1.html" class="btn btn-secondary">Start Tutorial</a>
</div>