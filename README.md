🚀 Automated Glaucoma Screening from Retinal Fundus Images Using CNNs
Deep Learning · Medical Imaging · Segmentation · Clinical Explainability

This project implements a full clinical-grade glaucoma screening pipeline using retinal fundus images. It performs:

- Optic disc & optic cup segmentation using a multi-fold U-Net ensemble
- Vertical CDR (Cup-to-Disc Ratio) computation
- Glaucoma risk classification
- Explainability visualization (color masks & overlays)
- ONNX export for deployment
- Interactive Streamlit Web App for real-time inference

This project is suitable for research, academic demonstration, and deployment as an interactive medical AI tool.

📊 Dataset

We use the large combined multi-dataset glaucoma collection:

📌 Kaggle Dataset:
https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets

This dataset includes:

- REFUGE (train/val/test)
- ORIGA
- G1020
- Mask annotations for disc & cup segmentation
- Cropped & square-aligned images

For this project, we train primarily on:

- REFUGE Images_Square
- REFUGE Masks_Square

REFUGE is a standard benchmark dataset for glaucoma research.

🧠 Model Architecture

✔ U-Net Backbone

- Input size: 256 × 256
- Output: 3-class segmentation
  - Background = 0
  - Disc = 1
  - Cup = 2

✔ 5-Fold Cross-Validation

Each fold trains independently and saves:

```
outputs/
    fold_0/best_model.pth
    fold_1/best_model.pth
    ...
    fold_4/best_model.pth
```

The Streamlit app loads all folds as an ensemble.

📈 5-Fold Evaluation Results

After full 5-fold training:

| Metric    | Mean   | Std    |
|-----------|--------|--------|
| Disc Dice | 0.9038 | 0.0019 |
| Cup Dice  | 0.8836 | 0.0033 |
| Mean Dice | 0.8937 | 0.0013 |

These results are consistent with published performance in glaucoma segmentation literature.

🗂 Project Structure

```
glaucoma-screening-cnn/
│-- app.py
│-- main.py
│-- trainer.py
│-- dataset.py
│-- model.py
│-- utils.py
│-- visualize.py
│-- evaluate.py
│-- export_onnx.py
│-- predict.py
│-- summarize_segmentation_from_models.py
│-- sample_images/
│-- outputs/
│-- glaucoma_unet.onnx
│-- requirements.txt
└── README.md
```

⚙️ Installation

1️⃣ Clone the repo

```bash
git clone https://github.com/1avishek/glaucoma-screening-cnn.git
cd glaucoma-screening-cnn
```

2️⃣ Create & activate environment

```bash
conda create -n glaucoma python=3.10 -y
conda activate glaucoma
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

🏋️ Training the Model (5-Fold)

Run each fold (example for fold 0):

```bash
python main.py --fold_index 0 --device cuda
python main.py --fold_index 1 --device cuda
python main.py --fold_index 2 --device cuda
python main.py --fold_index 3 --device cuda
python main.py --fold_index 4 --device cuda
```

🧪 Evaluate 5-Fold Results

```bash
python summarize_segmentation_from_models.py
```

Output example:

```
Disc Dice mean = 0.9038
Cup Dice mean  = 0.8836
```

🟦 Export to ONNX (deployment-ready)

```bash
python export_onnx.py
```

This generates:

- glaucoma_unet.onnx
- glaucoma_unet.onnx.data

🌐 Run the Streamlit Web App (Local)

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

The app supports:

- Upload your own image
- Use sample images from /sample_images/
- Segmentation visualization
- CDR measurement
- Glaucoma risk prediction

☁️ Deploy on Streamlit Cloud (Free)

Step 1: Push repo to GitHub

Make sure the repo contains:

- ✔ app.py
- ✔ requirements.txt
- ✔ sample_images/
- ✔ outputs/fold_*/best_model.pth

Step 2: Go to:

👉 https://share.streamlit.io

Step 3: Create a new app

- Connect your GitHub account
- Select your repo 1avishek/glaucoma-screening-cnn
- Set Main file = app.py

Step 4: Deploy

Streamlit Cloud will:

- Install dependencies
- Download your model files
- Host your app online

📌 You will get a URL like:

```
https://glaucoma-screening-cnn.streamlit.app/
```

You can share this with:

- Teachers
- Friends
- Doctors
- Recruiters

As a portfolio link

🖼 Example Output

- ✔ Segmentation Mask
  - Cup=Red, Disc=Green
- ✔ Overlay
  - Shows clinical CDR value + risk level.

🧾 Citation


```
A. Avishek, Automated Glaucoma Screening from Fundus Images Using CNNs, 2025.
```

📜 License

MIT License — free to modify & use.
