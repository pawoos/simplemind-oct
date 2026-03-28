---
layout: page
title: Tutorials
permalink: /tutorials/
---

# SimpleMind Tutorials

Learn SimpleMind through hands-on exercises that build from basic concepts to advanced AI pipeline development.

## 📚 Learning Path

### 🎯 Beginner Level

#### [Exercise 1: Plans and Tools for Inference](simplemind/docs/exercise1.html)
**Duration:** ~20 minutes  
**Focus:** Understanding the basics of *Thinking*

Learn how SimpleMind tools communicate and process data:
- Understanding JSON plans
- Tool input/output messages
- Running inference pipelines
- Interpreting results

**What you'll build:** A simple image processing pipeline

---

#### [Exercise 2: Developing a Tool](simplemind/docs/exercise2.html)
**Duration:** ~25 minutes  
**Focus:** Creating your first custom tool

Build your own SimpleMind tool from scratch:
- Tool structure and templates
- Message handling
- Integration with the blackboard
- Testing and debugging

**What you'll build:** A custom image filter tool

---

### 🚀 Intermediate Level

#### [Exercise 3: Decision Trees](simplemind/docs/exercise3.html)
**Duration:** ~20 minutes  
**Focus:** Adding logic and reasoning

Implement decision-making in your pipelines:
- Conditional processing
- Rule-based logic
- Multi-path workflows
- Error handling

**What you'll build:** An intelligent routing system

---

#### [Exercise 4: Training a Tool](simplemind/docs/exercise4.html)
**Duration:** ~25 minutes  
**Focus:** Machine learning and *Learning*

Train your tools with data:
- Dataset preparation
- Training workflows
- Model evaluation
- Deployment strategies

**What you'll build:** A trainable segmentation tool

---

## 🎨 Practical Examples

### Medical Imaging Workflows

#### Chest X-ray Analysis
![CXR Pipeline](assets/images/cxr-pipeline-diagram.png)

**Components:**
- Image preprocessing
- Lung segmentation
- Trachea detection
- Quality assessment
- Report generation

**Code Example:**
```json
{
  "plan_name": "chest_xray_analysis",
  "tools": [
    {
      "name": "preprocess_cxr",
      "type": "image_processor",
      "parameters": {
        "resize": [512, 512],
        "normalize": true
      }
    },
    {
      "name": "segment_lungs",
      "type": "segmentation_tool",
      "model": "lung_segmentation_v2"
    }
  ]
}
```

#### OCT Corneal Analysis
![OCT Pipeline](assets/images/oct-pipeline-diagram.png)

**Features:**
- 3D volume processing
- Corneal boundary detection
- Thickness measurements
- Dewarping correction

##### OCT Image Format Conversion Tool

**Purpose:** Convert PNG images to NIfTI format for OCT analysis pipelines

SimpleMind includes a specialized reformat tool for converting PNG images to NIfTI format with proper medical imaging orientation and metadata. This tool is essential for preparing OCT data for analysis.

**Key Features:**
- Automatic PNG to NIfTI conversion with proper orientation
- Batch processing of multiple images
- Consistent output structure for SimpleMind pipelines
- CSV metadata generation for dataset tracking

**Usage:**
```bash
# Convert PNG images from any folder
python reformat/reformat.py path/to/png/folder

# Output:
# - NIfTI images saved to: simplemind/data/oct_images/
# - Dataset CSV created: simplemind/data/oct_images.csv
```

**Example Workflow:**
```bash
# 1. Place PNG images in a folder (e.g., scan0/)
# 2. Run the conversion tool
python reformat/reformat.py reformat/scan0

# 3. Converted files are automatically organized:
#    - simplemind/data/oct_images/image1.nii.gz
#    - simplemind/data/oct_images/image2.nii.gz
#    - simplemind/data/oct_images.csv (dataset metadata)
```

**CSV Output Format:**
The tool generates a CSV file compatible with SimpleMind's data loading system:
```csv
image,label_mask,sample_args
oct_images/scan_001.nii.gz,,--upload_tags nifti
oct_images/scan_002.nii.gz,,--upload_tags nifti
```

**Technical Details:**
- Applies 90° counter-clockwise rotation for proper medical imaging orientation
- Preserves image data as uint8 for efficient storage
- Creates standardized NIfTI headers for consistent processing
- Automatically handles batch conversion of entire directories

### Computer Vision Pipelines

#### Multi-Modal Processing
Combine different data types in a single pipeline:
- Image + metadata processing
- Cross-modal validation
- Ensemble predictions
- Result fusion

## 🛠️ Advanced Topics

### Custom Tool Development

#### Tool Architecture
```python
class MyCustomTool(SimpleMindTool):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.load_model()
    
    def process(self, input_message):
        # Your processing logic here
        result = self.model.predict(input_message.data)
        return self.create_output_message(result)
```

#### Best Practices
- **Modular Design** - Keep tools focused and reusable
- **Error Handling** - Graceful failure and recovery
- **Logging** - Comprehensive debugging information
- **Testing** - Unit tests for reliability

### Pipeline Optimization

#### Performance Tuning
- **GPU Utilization** - Efficient memory management
- **Parallel Processing** - Multi-tool execution
- **Caching** - Avoid redundant computations
- **Batch Processing** - Optimize throughput

#### Monitoring & Debugging
- **Dashboard Usage** - Real-time pipeline monitoring
- **Log Analysis** - Troubleshooting failures
- **Profiling** - Performance bottleneck identification

## 📊 Interactive Examples

### Try It Yourself

#### Quick Start Notebook
```python
# Load SimpleMind
import simplemind as sm

# Create a simple pipeline
pipeline = sm.Pipeline([
    sm.tools.ImageLoader(),
    sm.tools.Preprocessor(resize=(256, 256)),
    sm.tools.Segmenter(model='default'),
    sm.tools.Visualizer()
])

# Run on sample data
results = pipeline.run('sample_image.jpg')
```

#### Web Interface Demo
Experience SimpleMind through our interactive web demo:
- Upload your own images
- Select processing pipelines
- View real-time results
- Download processed outputs

## 🎓 Certification Path

### SimpleMind Fundamentals
Complete all four exercises to earn your SimpleMind Fundamentals certificate:

- ✅ Exercise 1: Plans and Tools
- ✅ Exercise 2: Tool Development  
- ✅ Exercise 3: Decision Trees
- ✅ Exercise 4: Training

### Advanced SimpleMind Developer
Master advanced concepts:
- Custom tool architectures
- Performance optimization
- Production deployment
- Research applications

## 📝 Tutorial Resources

### Code Templates
- [Basic Tool Template](assets/code/basic_tool_template.py)
- [Training Tool Template](assets/code/training_tool_template.py)
- [Pipeline Configuration](assets/code/pipeline_config.json)

### Sample Datasets
- [Chest X-ray Images](data/cxr_sample.zip)
- [OCT Volumes](data/oct_sample.zip)
- [Natural Images](data/natural_sample.zip)

### Video Tutorials
- 🎥 [SimpleMind Overview](https://youtube.com/watch?v=example1) (10 min)
- 🎥 [Tool Development Deep Dive](https://youtube.com/watch?v=example2) (25 min)
- 🎥 [Production Deployment](https://youtube.com/watch?v=example3) (15 min)

## 🤝 Community Tutorials

### User-Contributed Examples
- **Satellite Image Analysis** by @researcher_jane
- **Audio Processing Pipeline** by @audio_expert
- **Time Series Forecasting** by @data_scientist

### Tutorial Requests
Have an idea for a tutorial? [Submit a request](https://github.com/yourusername/simplemind/issues/new?template=tutorial_request.md) or contribute your own!

---

## Next Steps

Ready to apply your knowledge? Check out our [examples gallery](examples.html) or dive into the [full documentation](documentation.html).

Need help? Join our [community discussions](https://github.com/yourusername/simplemind/discussions) or reach out for support!