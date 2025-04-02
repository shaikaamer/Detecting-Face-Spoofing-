def project_summary():
    summary = """
TME 6017 Detecting-Face-Spoofing Final Project Summary
------------------------------

Dataset
- CASIA-FASD (subset used for training/testing)
- Directory structure: train/live, train/spoof
- Preprocessed: resized to 150x150, RGB format, normalized to [0, 1]

Models
Autoencoder
- Trained only on live faces to learn reconstruction
- Detects spoof faces via high reconstruction error (MSE > 0.02)

Classifier (MobileNetV2)
- Transfer learning with ImageNet weights
- Fine-tuned for binary classification: live vs. spoof
- Works alongside autoencoder for added robustness

 Decision Logic (Hybrid)
- If MSE > 0.02 and Classifier > 0.7 →  Spoof
- If MSE < 0.02 and Classifier > 0.5 →  Spoof
- Else →  Live

Evaluation
- Classifier Accuracy: 99.5% (train), ~74% (validation)
- Autoencoder: error histograms, boxplots, summary stats
- CSV outputs, saved reconstructions, confusion matrix, ROC curve

 Flask Web App
- Upload a face image via browser
- Returns classification + MSE score
- Real-time integration of both models (AE + Classifier)

Final Notes
- Modular code structure, easy to maintain
- Report-ready with visuals and summaries
- Fully reproducible and demo-ready
"""
    print(summary)


project_summary()
