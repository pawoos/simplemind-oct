---
layout: page
title: Documentation
permalink: /documentation/
---

# SimpleMind Documentation

Comprehensive technical documentation for SimpleMind developers and researchers.

## 📖 Core Concepts

### Architecture Overview

SimpleMind uses a **Blackboard Architecture** where tools communicate through a central message board:

```
┌─────────────────────────────────────────────────────────┐
│                    SimpleMind System                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Tool A  │  │ Tool B  │  │ Tool C  │  │ Tool D  │   │
│  │         │  │         │  │         │  │         │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │            │            │            │        │
│       └────────────┼────────────┼────────────┘        │
│                    │            │                     │
│              ┌─────▼────────────▼─────┐               │
│              │    Blackboard Service   │               │
│              │   (Message Exchange)    │               │
│              └─────────────────────────┘               │
├─────────────────────────────────────────────────────────┤
│                   Controller Layer                      │
│  ┌─────────────┐              ┌─────────────┐          │
│  │ run_plan.py │              │start_mind.py│          │
│  │ (Transient) │              │(Persistent) │          │
│  └─────────────┘              └─────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Tools
Self-contained processing units that:
- Receive input messages from the blackboard
- Process data using AI models or algorithms
- Send output messages back to the blackboard
- Can be chained together in complex workflows

#### 2. Blackboard Service
Central communication hub that:
- Manages message passing between tools
- Provides data persistence and caching
- Handles tool discovery and registration
- Monitors system health and performance

#### 3. Controllers
Orchestration layer that:
- Manages tool lifecycle (start/stop)
- Coordinates data flow
- Provides monitoring and debugging
- Handles error recovery

#### 4. Plans
JSON configuration files that:
- Define tool connections and parameters
- Specify data flow and dependencies
- Enable version control and reproducibility
- Support collaborative development

## 🔧 Technical Reference

### [Controller Documentation](simplemind/docs/controller.html)

**Controllers** manage pipeline execution and tool orchestration.

#### Run Plan Controller (`run_plan.py`)
- **Purpose**: Transient tools for single data instances
- **Use Case**: Batch processing, one-time analysis
- **Lifecycle**: Tools start → process data → terminate

```bash
python run_plan.py plans/my_plan --dataset_csv data.csv --gpu_num 0
```

#### Start Mind Controller (`start_mind.py`)
- **Purpose**: Persistent tools for multiple data instances
- **Use Case**: Real-time processing, continuous operation
- **Lifecycle**: Tools start → wait for data → process → repeat

```bash
python start_mind.py plans/my_plan --gpu_num 0
# In another terminal:
python upload_dataset.py data.csv
```

### [Dashboard Guide](simplemind/docs/dashboard.html)

Real-time monitoring interface showing:

#### Status Indicators
| Format | Meaning |
|--------|---------|
| **-** | Tool has started |
| **S / M** | S = highest consecutive sample, M = max samples |
| **S* / M** | Sample S being processed by execute method |
| **a / a** | Aggregate method processing/completed |

#### Performance Metrics
- Processing speed (samples/second)
- Memory utilization
- GPU usage
- Error rates

### Tool Development

#### Tool Structure
```python
import simplemind as sm

class MyTool(sm.Tool):
    def __init__(self, plan_id, tool_id, **kwargs):
        super().__init__(plan_id, tool_id, **kwargs)
        self.model = self.load_model()
    
    def execute(self, sample_id, input_data):
        """Process a single sample"""
        result = self.model.predict(input_data)
        self.send_output(sample_id, result)
    
    def aggregate(self):
        """Post-processing after all samples"""
        self.generate_summary_report()
```

#### Tool Registration
```json
{
  "tool_name": "my_custom_tool",
  "class_path": "tools.my_tool.MyTool",
  "parameters": {
    "model_path": "models/my_model.pth",
    "threshold": 0.5
  }
}
```

### Message Format

#### Input Message
```json
{
  "sample_id": "001",
  "data_type": "image",
  "data_path": "/path/to/image.jpg",
  "metadata": {
    "patient_id": "P001",
    "study_date": "2024-01-15"
  }
}
```

#### Output Message
```json
{
  "sample_id": "001",
  "tool_id": "segmentation_tool",
  "result_type": "mask",
  "result_path": "/path/to/mask.nii.gz",
  "confidence": 0.95,
  "processing_time": 2.3
}
```

## 🏥 Medical Imaging Specifics

### Supported Formats
- **DICOM** - Medical imaging standard
- **NIfTI** - Neuroimaging format (.nii, .nii.gz)
- **PNG/JPEG** - Standard image formats
- **NumPy** - Array data (.npy, .npz)

### Common Workflows

#### Segmentation Pipeline
```json
{
  "plan_name": "organ_segmentation",
  "tools": [
    {
      "name": "dicom_loader",
      "type": "data_loader",
      "output_format": "nifti"
    },
    {
      "name": "preprocessor",
      "type": "image_processor",
      "operations": ["normalize", "resize"]
    },
    {
      "name": "segmenter",
      "type": "ai_model",
      "model_type": "unet",
      "classes": ["background", "organ"]
    },
    {
      "name": "postprocessor",
      "type": "mask_processor",
      "operations": ["smooth", "fill_holes"]
    }
  ]
}
```

#### Quality Assessment
```json
{
  "plan_name": "qa_pipeline",
  "tools": [
    {
      "name": "image_qa",
      "type": "quality_checker",
      "metrics": ["snr", "contrast", "artifacts"]
    },
    {
      "name": "segmentation_qa",
      "type": "mask_validator",
      "checks": ["connectivity", "volume_range"]
    }
  ]
}
```

## 🚀 Deployment

### Docker Configuration

#### Dockerfile
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    golang-go \
    && rm -rf /var/lib/apt/lists/*

# Copy SimpleMind
COPY . /app/simplemind
WORKDIR /app/simplemind

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install Core
RUN go install gitlab.com/hoffman-lab/core@v1.1.1

EXPOSE 8080
CMD ["python3", "start_mind.py"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  simplemind:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Production Considerations

#### Scaling
- **Horizontal**: Multiple SimpleMind instances
- **Vertical**: GPU/CPU resource allocation
- **Load Balancing**: Distribute processing load

#### Monitoring
- **Health Checks**: Tool and service status
- **Metrics Collection**: Performance and usage data
- **Alerting**: Failure notifications

#### Security
- **Authentication**: User access control
- **Data Encryption**: At rest and in transit
- **Network Security**: Firewall and VPN setup

## 🔬 Research Applications

### Academic Use Cases

#### Medical AI Research
- **Dataset**: Large-scale medical imaging studies
- **Models**: Novel architecture development
- **Validation**: Multi-center clinical trials
- **Publication**: Reproducible research workflows

#### Computer Vision Research
- **Benchmarking**: Standardized evaluation protocols
- **Ablation Studies**: Component-wise analysis
- **Novel Architectures**: Experimental model testing
- **Transfer Learning**: Cross-domain adaptation

### Industry Applications

#### Healthcare Systems
- **Clinical Workflow**: Integration with PACS/RIS
- **Regulatory Compliance**: FDA/CE marking support
- **Quality Assurance**: Continuous monitoring
- **Scalability**: Enterprise deployment

#### Research Organizations
- **Collaborative Projects**: Multi-site coordination
- **Data Sharing**: Federated learning support
- **Resource Management**: Compute optimization
- **IP Protection**: Secure model deployment

## 📊 Performance Optimization

### GPU Optimization
```python
# Tool configuration for GPU efficiency
{
  "gpu_memory_fraction": 0.8,
  "batch_size": 16,
  "mixed_precision": true,
  "model_parallelism": false
}
```

### Memory Management
```python
# Blackboard configuration
{
  "object_store": {
    "target_gib": 16,
    "gc_threshold": 15
  },
  "server": {
    "ring_buffer_size": 8192
  }
}
```

### Profiling Tools
- **GPU Profiling**: NVIDIA Nsight, nvprof
- **CPU Profiling**: Python cProfile, line_profiler
- **Memory Profiling**: memory_profiler, tracemalloc
- **Network Profiling**: Wireshark, tcpdump

## 🛠️ Development Tools

### IDE Integration

#### VS Code Extensions
- **Python**: Full Python development support
- **Remote Development**: SSH/container development
- **Docker**: Container management
- **Git**: Version control integration

#### Debugging
```python
# Enable debug mode
import simplemind as sm
sm.set_debug_mode(True)

# Add breakpoints in tools
import pdb; pdb.set_trace()
```

### Testing Framework
```python
import unittest
from simplemind.testing import ToolTestCase

class TestMyTool(ToolTestCase):
    def setUp(self):
        self.tool = MyTool(plan_id="test", tool_id="test_tool")
    
    def test_processing(self):
        result = self.tool.execute("sample_001", test_data)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, expected_shape)
```

## 📚 API Reference

### Core Classes

#### Tool Base Class
```python
class Tool:
    def __init__(self, plan_id, tool_id, **kwargs)
    def execute(self, sample_id, input_data)
    def aggregate(self)
    def send_output(self, sample_id, data)
    def get_input(self, sample_id)
```

#### Message Classes
```python
class Message:
    sample_id: str
    tool_id: str
    data: Any
    metadata: Dict
    timestamp: datetime
```

#### Plan Configuration
```python
class PlanConfig:
    plan_name: str
    tools: List[ToolConfig]
    dependencies: Dict
    parameters: Dict
```

### Utility Functions
```python
# Data handling
sm.load_image(path)
sm.save_mask(mask, path)
sm.convert_format(data, target_format)

# Visualization
sm.overlay_mask(image, mask)
sm.plot_metrics(metrics)
sm.generate_report(results)
```

---

## 🆘 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Check environment
micromamba list
go version
python --version

# Reinstall dependencies
micromamba env remove -n smcore
micromamba create -f env.yaml
```

#### Runtime Errors
```bash
# Check blackboard service
core status

# View logs
tail -f working-*/stdout-*.log
tail -f working-*/stderr-*.log
```

#### Performance Issues
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check memory usage
free -h
```

### Getting Help

- 📧 **Technical Support**: Contact the development team
- 📚 **Documentation**: Search this comprehensive guide
- 🐛 **Bug Reports**: Submit issues with detailed logs
- 💬 **Community**: Join discussions and Q&A

---

Ready to start developing? Check out our [getting started guide](getting-started.html) or jump into the [tutorials](tutorials.html)!