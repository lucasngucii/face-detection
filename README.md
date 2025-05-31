# Attention Monitoring System

A system for real-time attention monitoring using facial feature analysis. It detects states such as *Attentive*, *Distracted*, and *Sleepy* by integrating face detection, feature extraction, synthetic data generation, and attention classification.

## System Architecture
The system comprises five core components:
1. **FaceDetector**: Detects faces using MTCNN.
2. **FaceFeatureExtractor**: Extracts 6 facial features (EAR, MAR, Head Pose X/Y, Face Ratio, Eye Distance) using MediaPipe or ResNet-based methods.
3. **SyntheticDataGenerator**: Generates synthetic training data with realistic correlations.
4. **AttentionClassifier**: Trains machine learning models (Random Forest, SVM, Neural Network) with hyperparameter optimization.
5. **AttentionMonitoringSystem**: Integrates all components for frame processing, state tracking, and reporting.

## Dependencies
The system requires the following Python packages:
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `facenet-pytorch`
- `mediapipe`
- `torch`
- `torchvision`

To install, create a `requirements.txt` file:
```bash
echo -e "opencv-python\nnumpy\npandas\nmatplotlib\nseaborn\nscikit-learn\nfacenet-pytorch\nmediapipe\ntorch\ntorchvision" > requirements.txt
pip install -r requirements.txt
```

## Dataset
- **Default Dataset**: WIDER FACE, stored in `datasets/WIDER_FACE`.
- **Synthetic Data**: Generated during the pipeline, saved as `synthetic_attention_data.csv`.
- **Custom Dataset**: Organize images in a folder structure compatible with `DatasetLoader`:
  ```
  datasets/WIDERFACE/
  ├── person1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  ├── person2/
  │   ├── image1.jpg
  ...
  ```

## Documentation
Detailed documentation for each component is available in PDF format under the `docs/` directory:
- `AttentionClassifier.pdf`: Classification model details.
- `AttentionMonitoringSystem.pdf`: Integrated monitoring system.
- `FaceDetector.pdf`: Face detection logic.
- `FaceFeatureExtractor.pdf`: Feature extraction methods.
- `SyntheticDataGenerator.pdf`: Synthetic data generation.
- `Full_demo.pdf`: Full pipeline walkthrough.


## Acknowledgements
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) for the dataset.
- [MediaPipe](https://mediapipe.dev/) and [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for face detection and feature extraction.
- [scikit-learn](https://scikit-learn.org/) for machine learning models.
- Developed by `Lucas Aleh` for attention monitoring research.

---
*For issues or questions, open an issue on GitHub or contact [lucasalehwork@gmail.com].*
