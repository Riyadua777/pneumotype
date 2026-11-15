# PneumoType: what kind of pneumonia is this X-ray?

**Track:** Health
**Team:** [names]

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
**Data:** Kermany et al. 2018 pediatric chest X-rays (Guangzhou Women and Children's Medical Center), [N] images from [N] patients: [n] normal, [n] bacterial, [n] viral. Labels come from folder and file names. We **split by patient** (70/15/15) so no child appears in both training and test.

**Model:** our own convolutional neural network, **trained from scratch** (random starting weights, no pretrained model). It has 5 blocks of convolution → batch norm → ReLU → max-pool (32 to 512 filters), then global average pooling, dropout and a linear layer: about 4.7M parameters. We trained it for [25] epochs on 224×224 grayscale X-rays with class-weighted loss (so rarer classes count more) and light augmentation (small rotations, zoom, brightness). We kept the epoch with the best validation macro recall.

## How to run
Open `pneumotype.ipynb` on Kaggle (File → Import Notebook → GitHub), add the dataset "Chest X-Ray Images (Pneumonia)", turn on the GPU T4 and Internet, then Run All. The last cell launches the Gradio demo.

## Results (held-out test set)
| Model | Accuracy | Macro recall | Bacterial recall | Viral recall | Pneumonia recall (any type) | Dangerous misses |
|---|---|---|---|---|---|---|
| Baseline: always the most common class | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Baseline: logistic regression on 32×32 pixels | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Our CNN (from scratch) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Our CNN + safety tuning | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

- **Missed cases vs false alarms:** the model missed [n] pneumonia cases (called them normal) and raised [n] false alarms (normal called pneumonia).
- **Unsure mode:** at a confidence cutoff of [0.6], the tool decides on [x]% of X-rays with [y]% accuracy and refers the rest.
- **Robustness:** macro recall dropped from [x] to [y] on low-contrast images and [z] on low-resolution images.

**Where it fails:** most errors are bacterial vs viral confusions, which is expected because even radiologists can't reliably tell them apart from an X-ray alone. Example: [describe one wrong case from step 11 and why you think it failed].

**Fairness and limits:** the dataset has no age or sex labels and comes from one hospital (children aged 1–5), so we couldn't check performance across groups. The labels are doctors' readings, not lab-confirmed causes. The model has not been tested on adults or on X-rays from other hospitals.

## What we'd do next
1. Test on X-rays from a different hospital and country to see if it still works.
2. Add tuberculosis as a fourth class, with a check that the model learns disease rather than which hospital an image came from.

## Sources
- Dataset: Kermany, Zhang & Goldbaum (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data; Kaggle copy "Chest X-Ray Images (Pneumonia)" by Paul Mooney (CC BY 4.0).
- No pretrained models: the CNN is trained from scratch.
- Libraries: PyTorch, torchvision, scikit-learn, pandas, matplotlib, Gradio
- AI tools: Claude (Anthropic) helped plan the project and draft code and this write-up; [add any others]
