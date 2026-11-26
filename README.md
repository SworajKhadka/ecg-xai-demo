# Explainable ECG Analysis: demo app

## Folder layout (after unzipping ecg_models.zip from Kaggle into this folder)
ecg-xai-demo/
  app.py            Streamlit demo
  ecgxai.py         shared code (models, pre-processing, explanations)
  requirements.txt
  samples/          00001_lr.hea + 00001_lr.dat (PTB-XL record 00001)
  models/           resnet.pt, inception.pt, bilstm.pt, meta.json   <- from ecg_models.zip
  results.json      test-set metrics                                 <- from ecg_models.zip
  figures/          Grad-CAM / IG / SHAP figures                     <- from ecg_models.zip

## Run
python -m venv .venv
.venv\Scripts\activate          (Windows)    |    source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
streamlit run app.py
Then open http://localhost:8501
# ecg-xai-demo
