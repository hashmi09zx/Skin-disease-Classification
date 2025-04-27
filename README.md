# 🩺 Skin Disease Classification using MobileNet

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![Validation Accuracy](https://img.shields.io/badge/Validation_Accuracy-87.84%25-brightgreen)
![Model](https://img.shields.io/badge/Model-MobileNet-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-HMNIST_10000-lightgrey)

---


🚀 Project Overview
This project is an end-to-end deep learning solution for multiclass skin disease prediction.
It uses the HMNIST 10000 dataset and a MobileNet pretrained model for efficient and accurate classification.



🗂️ Project Structure

skin-cancer-detection/

├── app/                   # Python backend code

├── static/                 # Static files (images, css, js if needed)

├── templates/              # HTML templates

├── venv/                   # Virtual environment (ignored in GitHub)

├── mobilenet_skin_model_final.keras  # Trained model

├── requirements.txt        # Python dependencies

├── Skin_Disease_Classification_MobileNet.ipynb  # Model training notebook

├── .gitignore

└── README.md




📊 Model Performance

![Model Evaluation](static\images\model_evaluation_table.png)
✅ The model shows strong generalization with high top-2 and top-3 accuracies, suitable for real-world applications!


📦 Installation and Running
1. Clone the repository:
    git clone https://github.com/hashmi09zx/Skin-disease-Classification.git
cd Skin-disease-Classification
2. Create and activate a virtual environment:
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
3. Install dependencies:
    pip install -r requirements.txt
4. Run the application:
    python app/app.py


📷 Demo Screenshots

![Demo Image](static\images\demo.png)

![Prediction Image](static\images\image.png)


✨ Future Improvements
1. Improve classification accuracy with further fine-tuning.
2. Deploy using cloud services like AWS, GCP, or Azure.
3. Add more visualization for prediction results.



🙌 Acknowledgements

MobileNet — Lightweight model ideal for mobile and embedded vision applications.

Kaggle — For providing the HMNIST dataset.

TensorFlow & Keras — For deep learning frameworks.

