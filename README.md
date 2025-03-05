# ASL Recognition Using Mediapipe Hand Landmarks

This project aims to build an American Sign Language (ASL) recognition system using Mediapipe's Hand Landmarker and a Convolutional Neural Network (CNN) for classification.

## Project Structure

```
.
├── models/                # Directory for trained models
├── handLandmarker.py      # Script for hand landmark detection using Mediapipe
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/dehadeaaryan/ASL-Mediapipe.git
cd ASL-Mediapipe
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Current Progress

1. ✅ Implemented hand landmark detection using Mediapipe (see `handLandmarker.py`)
2. 🔜 Next step: Train a CNN model on the extracted hand landmarks to recognize ASL gestures
3. 🔜 Final step: Integrate the Mediapipe Hand Landmarker with the CNN model for real-time ASL recognition

## Usage

### Run the Hand Landmarker

```bash
python handLandmarker.py
```

## Future Goals

- Train and evaluate a CNN on ASL gesture datasets.
- Optimize the model for real-time performance.
- Expand the system to recognize a larger vocabulary of ASL signs.

## Author

Created by [dehadeaaryan](https://github.com/dehadeaaryan).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.