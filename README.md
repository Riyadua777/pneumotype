# PneumoType: what kind of pneumonia is this X-ray?

**Track:** Health
**Team:** [add team names]

> This project is a prototype and is not medical advice.

## The problem
In a busy children's clinic, a doctor often has to decide quickly whether a child with a cough and fever needs antibiotics. Bacterial pneumonia needs antibiotics, viral pneumonia usually doesn't, and guessing wrong either way causes harm: missed treatment, or unnecessary antibiotics that drive resistance. Our user is a clinic doctor without a radiologist on hand who wants a fast second opinion on a chest X-ray.

## What we built
**Input:** one chest X-ray. **Output:** Normal, Bacterial pneumonia or Viral pneumonia, with a confidence score for each class and a heatmap of where the model looked. When the model isn't confident, it says **"Unsure: refer to a doctor"** instead of guessing. Demo: a Gradio web app (link in the notebook / video).

## What we added
Our own idea. Starting from the standard "pneumonia vs normal" task, we added:
1. **Pneumonia type** (bacterial vs viral), because the type changes the treatment.
2. **A safety tuning step** that makes the most dangerous mistake (calling a bacterial case viral or normal) rarer, chosen on validation data, with the cost reported on test data.
3. **An "unsure, refer" mode** and a table showing how much accuracy improves when the tool is allowed to defer.
4. **A robustness test** simulating a different clinic's X-ray machine (low contrast, blur, low resolution).

## How it works
**Data:** Kermany et al. 2018 pediatric chest X-rays (Guangzhou Women and Children's Medical Center), about 5,800 images of children aged 1–5: normal, bacterial pneumonia and viral pneumonia. The test set has 799 X-rays (222 normal, 374 bacterial, 203 viral). Labels come from folder and file names. We **split by patient** (70/15/15) so no child appears in both training and test.

**Model:** our own convolutional neural network, **trained from scratch** (random starting weights, no pretrained model). It has 5 blocks of convolution → batch norm → ReLU → max-pool (32 to 512 filters), then global average pooling, dropout and a linear layer: about 4.7M parameters. We trained it for 25 epochs on 224×224 grayscale X-rays with class-weighted loss (so rarer classes count more) and light augmentation (small rotations, zoom, brightness). We kept the epoch with the best validation macro recall.

## How to run
Open `pneumotype.ipynb` on Kaggle (File → Import Notebook → GitHub), add the dataset "Chest X-Ray Images (Pneumonia)", turn on the GPU T4 and Internet, then Run All. The last cell launches the Gradio demo.

## Results (held-out test set)
| Model | Accuracy | Macro recall | Bacterial recall | Viral recall | Pneumonia recall (any type) | Dangerous misses |
|---|---|---|---|---|---|---|
| Baseline: always "bacterial" | 0.468 | 0.333 | 1.000 | 0.000 | 1.000 | 0 |
| Baseline: logistic regression on 32×32 pixels | 0.771 | 0.772 | 0.754 | 0.675 | 0.953 | 92 |
| **Our CNN (from scratch)** | **0.832** | **0.825** | 0.845 | 0.670 | 0.965 | 58 |
| **Our CNN + safety tuning** | 0.817 | 0.780 | **0.939** | 0.468 | **0.986** | **23** |

*Macro recall* is the average of the recall for each class, so a model can't score well by ignoring a class. *Dangerous misses* = bacterial pneumonia cases called viral or normal (the child could miss antibiotics).

Test-set confusion matrix for our CNN (rows = true, columns = predicted):

| | normal | bacterial | viral |
|---|---|---|---|
| **normal** | 213 | 4 | 5 |
| **bacterial** | 8 | 316 | 50 |
| **viral** | 12 | 55 | 136 |

- **Pneumonia vs normal:** our CNN caught 96.5% of pneumonia cases (missed 20 of 577, calling them normal) with only 9 false alarms out of 222 normal X-rays (95.9% of normal X-rays correctly cleared).
- **Safety tuning:** adding a bonus to the bacterial score (0.58, chosen on validation data only) cut dangerous misses from 58 to 23 and raised pneumonia recall to 98.6%. The cost: viral recall fell from 67% to 47% (more viral cases get called bacterial) and false alarms rose from 9 to 15. We think that's the right trade for a clinic, because over-treating a viral case is less harmful than missing a bacterial one.
- **Unsure mode:** if the tool only answers when it's at least 70% confident, it decides on 76% of X-rays with 90.3% accuracy and refers the other 24% to a doctor. At 80% confidence it decides on 61% with 93.7% accuracy.
- **Robustness (our biggest weakness):** when we made test X-rays look like they came from a different machine, macro recall fell from 0.780 to 0.551 (low contrast), 0.577 (blur) and 0.586 (low resolution). The model still flagged almost every pneumonia case, but it started calling many normal X-rays pneumonia and mixing up the types. It would need retraining or calibration before use at a new clinic.

**Where it fails:** most errors are bacterial vs viral mix-ups (105 of the 134 errors), which we expected, since even radiologists can't reliably tell the type from an X-ray alone. The most confident mistakes were pneumonia cases called *normal* with 97–98% confidence. Their lungs look fairly clear, so we think these are mild or early cases where the signs are too faint at 224×224 resolution. In the heatmaps, the model sometimes looks at the edges of the image instead of the lungs, which suggests it may be partly using image framing or exposure instead of disease.

**Fairness and limits:** the dataset has no age or sex labels and comes from one hospital (children aged 1–5), so we couldn't check performance across groups. The labels are doctors' readings, not lab-confirmed causes. The model has not been tested on adults or on X-rays from other hospitals.

## What we'd do next
1. Test on X-rays from a different hospital and train with stronger contrast and blur augmentation, to fix the robustness drop.
2. Add tuberculosis as a fourth class, with a check that the model learns disease rather than which hospital an image came from.

## Sources
- Dataset: Kermany, Zhang & Goldbaum (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data; Kaggle copy "Chest X-Ray Images (Pneumonia)" by Paul Mooney (CC BY 4.0).
- No pretrained models: the CNN is trained from scratch.
- Libraries: PyTorch, torchvision, scikit-learn, pandas, matplotlib, Gradio
- AI tools: Claude (Anthropic) helped plan the project and draft code and this write-up; [add any others]
