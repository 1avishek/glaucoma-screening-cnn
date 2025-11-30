## Streamlit Deployment Guide

Follow these steps to host the glaucoma screening demo on [Streamlit Community Cloud](https://streamlit.io/cloud) or any Streamlit-compatible platform.

### 1. Prepare the repository
1. Commit/push the latest version of this project, including:
   - The `outputs/fold_* /best_model.pth` checkpoints that the app loads.
   - A few demo images under `sample_images/` (place JPEG/PNG files there).
   - The new `requirements.txt` file.
2. (Optional) Add a short project description to your repo README so visitors understand the app.

### 2. Create the Streamlit app
1. Visit https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, choose the repository + branch, and set the main file to `app.py`.
3. Under **Advanced settings**, optionally set environment variables, e.g.:
   ```
   STREAMLIT_SERVER_HEADLESS=true
   ```
4. Deploy. Streamlit Cloud will install everything from `requirements.txt`, download your repo, and start `app.py`.

### 3. Verify sample assets
- The sidebar will show the compute device and let users select **Upload your own** or **Use sample image**.
- Ensure `sample_images/` contains at least one demo JPG/PNG so visitors without their own fundus scan can try the workflow.

### 4. Share
- Once the app is running, copy the Streamlit URL and share it with your teacher/friends.
- They can drag-and-drop new images or pick the bundled samples. The interface displays:
  - Original/resized fundus photo
  - Segmentation masks + overlay
  - Computed CDR and qualitative glaucoma risk

### Local testing
Before deploying remotely, confirm things work on your machine:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Visit the local URL (usually `http://localhost:8501`) to verify inference, sample selection, and UI components.

### Notes
- GPU acceleration is optional but makes inference faster. Streamlit Cloud currently provides CPU runtimes only, so keep batch size = 1 during inference (already the case).
- If you plan to update the model, regenerate `outputs/fold_* /best_model.pth` and redeploy.
