# AV-IO
--------------------
AV-IO: Audio-Visual Indoor-Outdoor Classification Dataset via Multimodal Weak Supervision
An activity can happen indoor or outdoor irrespective of the type or the class labels involved, etc. To this end neccessiated need to learn multimodal represntation to solve the problem of indoor-outdoor classification.\\
As there was a lacuna for the multimodal annotated dataset and inavailability of dedicated manpower to annotate the videos, we went for a automated captioning approach with CLIP and CLAP models. 
# AV-IO preparation 
--------------------
<img width="940" height="647" alt="image" src="dataset.jpg" />
# AV-IO benchmarks
--------------------

<img width="940" height="647" alt="image" src="benchmark.jpg" />


# Model Evaluation Results
----------------------------

| Model            | Parameters | TAU Val (%) | TAU Test (%) |   AVE (%) |
| ---------------- | ---------: | ----------: | -----------: | --------: |
| VO               |      2.55M |       94.04 |        93.47 |     74.13 |
| AO (Raw)         |      0.43M |       74.15 |        77.61 |     46.52 |
| AO (Spectrogram) |     28.01M |       87.70 |        83.15 |     50.75 |
| FF (Raw)         |      0.19M |   **94.99** |        93.41 |     73.13 |
| FF (Spectrogram) |      0.26M |       94.45 |        93.09 |     73.13 |
| DF (Raw)         |          - |       94.72 |        93.88 |     73.63 |
| EE (Raw)         |      2.90M |       94.59 |    **94.45** | **79.85** |
| EE (Spectrogram) |     31.30M |       93.64 |        94.27 |     74.63 |

**Abbreviations**

* **VO**: Vision Only
* **AO**: Audio Only
* **FF**: Feature Fusion
* **DF**: Decision Fusion
* **EE**: Early Embedding Fusion

**Key Observations**

* Feature Fusion (Raw) achieves the highest validation accuracy on TAU (94.99%).
* Early Embedding Fusion (Raw) achieves the highest test accuracy on TAU (94.45%).
* Early Embedding Fusion (Raw) significantly outperforms all other approaches on AVE, reaching 79.85%.
* Raw audio representations consistently outperform spectrogram-based representations when fused with visual features.

# How to Use
------------------
Will be updated soon.


#BibTex
--------
If you find our work interesting, please cite the work.
