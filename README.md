# PneumoType: what kind of pneumonia is this X-ray?

**Track:** Health
**Team:** [add team names]

> This project is a prototype and is not medical advice.

**In one sentence:** PneumoType catches **95% of pneumonia** on chest X-rays it has never seen, tells a doctor whether it looks **bacterial or viral**, says **"unsure, refer"** instead of guessing, and can be trained across hospitals with **federated learning**, so no hospital has to share a single patient X-ray.

Everything is our own convolutional neural network, **trained from scratch** (no pretrained models).

## Key findings
1. **95% of pneumonia caught** on 799 unseen children's X-rays (95% CI 93–97%), with only **8 false alarms** out of 222 healthy X-rays, beating a simple baseline.
2. **Safety tuning cut dangerous misses (bacterial cases called viral or normal) from 92 to 22**, and the tool says **"unsure, refer"** instead of guessing. When it's at least 90% confident, it's right **98%** of the time.
3. **We found a weakness and fixed it:** our first CNN fell to 73% on low-quality X-rays, and a second "clinic-robust" CNN brought it back to 82–83%.
4. **Federated learning across a children's and an adult hospital** gave the best worst-case result of any setup that keeps X-rays private.
5. **We caught our own TB model cheating:** it "detects" TB even with the lungs blacked out. Our pneumonia models pass that test, so we kept TB out of the demo.

![PneumoType system](figures/system.png)

![Results](figures/results.png)

## The problem
Pneumonia is the biggest infectious killer of children under five worldwide. When a child arrives at a clinic with a cough and fever, the doctor has to decide fast whether it's bacterial pneumonia (needs antibiotics), viral pneumonia (usually doesn't) or no pneumonia at all, often without a radiologist. Guessing wrong either way causes harm: a missed bacterial infection, or unnecessary antibiotics that drive resistance.

**Our user:** a doctor at a children's clinic with no radiologist on hand, who wants a fast, honest second opinion on a chest X-ray.

## What we built
- **Input:** one chest X-ray.
- **Output:**
  1. Normal / bacterial pneumonia / viral pneumonia, with calibrated probabilities.
  2. A heatmap of where the model looked.
  3. **"UNSURE: refer to a doctor"** when it isn't confident.
  4. A pneumonia check for **anyone, children or adults**, from a model trained with federated learning.
- **Demo:** a Gradio web app (shown in the video), launched from the last section of the notebook.

## What we added
Our own idea. Beyond a basic pneumonia classifier, we added:
1. **Pneumonia type** (bacterial vs viral), because the type changes the treatment.
2. **Safety tuning** that makes the most dangerous mistake (calling a bacterial case viral or normal) rarer, chosen on validation data only.
3. **An "unsure, refer" mode** with **calibrated** confidence.
4. **A robustness fix:** we found our first CNN breaks on lower-quality X-rays, so we trained a second CNN on simulated "different clinic" images and ensembled the two.
5. **A shortcut test:** we blacked out the lungs to check the model isn't cheating off image borders.
6. **Children and adults with federated learning:** two simulated hospitals train one model and share only weights, never X-rays.
7. **95% confidence intervals**, bootstrapped by patient.
8. **Tuberculosis with four hospital-shortcut checks.** They caught our TB model cheating, so we left TB out of the demo (see below).

## How it works
**Data**
- **Children:** Kermany et al. 2018 pediatric chest X-rays from Guangzhou Women and Children's Medical Center, about 5,800 images of children aged 1–5 (normal / bacterial / viral). Split **by patient** 70/15/15, so no child appears in both training and test. The test set has 799 X-rays.
- **Adults:** COVID-19 Radiography Database. We used 6,000 X-rays: normal and lung-opacity X-rays from RSNA, and COVID-19 pneumonia X-rays.
- **Leakage check:** that database re-uses 1,341 "normal" X-rays from the children's dataset. We **removed** them using its source metadata.

**Models** (all trained from random weights)
- **The CNN:** 5 blocks of convolution → batch norm → ReLU → max-pool (32 to 512 filters), then global average pooling, dropout and a linear layer. About 4.7M parameters, 224×224 grayscale input, class-weighted loss, 25 epochs. We kept the epoch with the best validation macro recall.
- **CNN 2 ("clinic-robust"):** the same architecture, trained with random contrast, blur and low-resolution augmentation.
- **Final model:** the ensemble of CNN 1 and CNN 2, plus test-time augmentation. It was **chosen on validation**, not test. Temperature calibration was checked too (it was already well calibrated: ECE 0.034).
- **Federated model:** FedAvg across a "children's hospital" and an "adult hospital". Each trains locally for one epoch per round and a server averages the weights, for 8 rounds.

## Results (held-out test set, never used for training or model choice)

### Children: normal / bacterial / viral (799 test X-rays)
| Model | Accuracy | Macro recall | Bacterial recall | Viral recall | Pneumonia caught | False alarms (of 222 normal) | Dangerous misses* |
|---|---|---|---|---|---|---|---|
| Baseline: always "bacterial" | 0.468 | 0.333 | 1.000 | 0.000 | 1.000 | 222 | 0 |
| Baseline: logistic regression on 32×32 pixels | 0.771 | 0.772 | 0.754 | 0.675 | 0.953 | 25 | 92 |
| **Final model (ensemble + TTA)** | **0.826** | **0.828** | 0.805 | **0.714** | 0.953 | **8** | 73 |
| **Final + safety tuning** | 0.809 | 0.767 | **0.941** | 0.419 | **0.984** | 13 | **22** |

*Dangerous misses = bacterial pneumonia called viral or normal, so the child could miss antibiotics. *Macro recall* averages recall across classes, so a model can't score well by ignoring one.

- **95% confidence intervals** (bootstrapped by patient): accuracy **0.826 [0.796–0.855]**, macro recall **0.828 [0.799–0.856]**, pneumonia caught **0.953 [0.932–0.970]**.
- **Safety tuning** cut dangerous misses from 73 to 22 and raised pneumonia caught to 98.4%. The cost: more viral cases get called bacterial (viral recall falls from 71% to 42%). We think that's the right trade for a clinic, because over-treating a viral case is less harmful than missing a bacterial one.
- **Unsure mode:** at 70% confidence the tool decides on 70% of X-rays with **90.1% accuracy** and refers the rest. At 90% confidence it decides on 35% with **98.2% accuracy**.

Final confusion matrix (rows = true, columns = predicted):

| | normal | bacterial | viral |
|---|---|---|---|
| **normal** | 214 | 2 | 6 |
| **bacterial** | 15 | 301 | 58 |
| **viral** | 12 | 46 | 145 |

### Robustness: we found a weakness and fixed it
Macro recall on test X-rays made to look like they came from a different machine:

| Test images | CNN 1 (original) | CNN 2 (clinic-robust) | Ensemble |
|---|---|---|---|
| Original | 0.844 | 0.811 | 0.829 |
| Low contrast | **0.728** | 0.820 | 0.816 |
| Blur | 0.778 | 0.812 | **0.830** |
| Low resolution | 0.781 | 0.816 | **0.830** |

CNN 1 dropped to 0.728 on low-contrast X-rays. The ensemble stays at about 0.82–0.83 on every kind of image.

### Shortcut test: is it really looking at the lungs?
| Input | Macro recall |
|---|---|
| Full X-ray | 0.829 |
| **Lungs blacked out** (only the border visible) | **0.333 (= chance)** |
| Border blacked out (only the lungs visible) | 0.587 |

With the lungs hidden, the model is no better than guessing. It is **not** using hospital markers or image borders as a shortcut: its skill comes from the lungs.

### Everyone: children + adults with federated learning (pneumonia vs normal)
Balanced accuracy on each population's test set (AUC in brackets):

| Model | X-rays shared? | Children | Adults | Worst case |
|---|---|---|---|---|
| Children's hospital only | no | 0.940 (0.989) | 0.783 (0.858) | 0.783 |
| Adult hospital only | no | 0.660 (0.917) | 0.870 (0.958) | 0.660 |
| Pooled (all X-rays in one place) | **yes** | 0.946 (0.987) | 0.862 (0.930) | 0.862 |
| **Federated (FedAvg)** | **no, weights only** | 0.852 (0.965) | 0.832 (0.908) | **0.832** |

- **Federated learning has the best worst-case of any private option.** A model trained at one hospital does badly at the other: the adult-only model caught only 32% of pneumonia in children.
- **Pooling all the data does best,** but it needs hospitals to hand over patient X-rays.
- **Honest limitation:** the federated model is cautious on children. It caught 72% of pneumonia, with 98.6% of normal X-rays correctly cleared. Its AUC of 0.965 shows the ranking is good, so tuning its decision threshold per hospital is our next step.

### Tuberculosis: we caught our own model cheating
We added the **Tuberculosis (TB) Chest X-ray Database** (adults; 700 TB + 1,400 normal sampled) as a third hospital, after removing **duplicate images across datasets** with an image fingerprint. Then we trained a normal / pneumonia / TB model.

| Model (normal / pneumonia / TB) | Balanced acc (all 3 hospitals) | TB caught | Normal X-rays called TB (children / adults / TB hospital) |
|---|---|---|---|
| Pooled (shares X-rays) | 0.854 | 91.3% | 0% / 6.7% / 10.3% |
| Federated, 3 hospitals | 0.536 | **0%** | n/a |

![Shortcut test](figures/shortcut_tb.png)

On paper, TB detection looks great: 91% caught. **Our four shortcut checks say not to trust it:**
1. **Hospital detector:** a CNN trained only to guess the hospital gets **92.4%** right (chance is 33%), so the hospitals' X-rays look very different.
2. **Same-hospital test:** inside the TB hospital it still catches 91.3% of TB, with 89.7% of normals cleared.
3. **Normals called TB:** this happens more often at the TB hospital (10.3%) than for adults (6.7%) or children (0%), a hint of a hospital effect.
4. **Lungs blacked out:** the model **still "detects" TB 100% of the time with no lungs visible.** That's the smoking gun. Its TB decision is driven by something outside the lungs, such as the image source, borders or exposure, not by the disease.

**Conclusion:** our pneumonia models pass the lungs-blacked-out test (they drop to chance), but the TB model fails it. This is the same trap as the well-known COVID-19 X-ray AI shortcut problem (DeGrave et al., *Nature Machine Intelligence*, 2021). We therefore **do not offer TB detection** in the demo. With only one hospital supplying TB cases, federated learning also failed: the TB hospital's signal was averaged away and the federated model caught 0% of TB. A trustworthy TB model needs TB *and* normal X-rays from many hospitals.

## Where it fails
- **Bacterial vs viral mix-ups** cause most errors (104 of 139). That's expected, because even radiologists can't reliably tell the type from an X-ray alone.
- **The most confident mistakes are pneumonia cases called normal.** These look like mild or early cases where the signs are faint at 224×224.
- **Federated model on children:** it's too cautious, as shown above.
- **Data limits:**
  - The children's data comes from one hospital, with no age or sex labels, so we couldn't check fairness across those groups.
  - Labels are doctors' readings, not lab-confirmed causes.
  - Adult X-rays come from different sources than the children's, so some differences between them may come from equipment, not age.

## What we'd do next
1. Tune the federated model's threshold at each hospital, and add more hospitals to the federation.
2. Fix the TB shortcut: gather TB and normal X-rays from the *same* hospitals, and train with lung segmentation so the model can only see the lungs.
3. Test with real clinicians at a clinic, on X-rays from a new hospital.

## Sources
- **Children's dataset:** Kermany, Zhang & Goldbaum (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data. Kaggle copy: "Chest X-Ray Images (Pneumonia)" by Paul Mooney (CC BY 4.0).
- **TB dataset:** Tuberculosis (TB) Chest X-ray Database, Rahman et al. 2020, Kaggle (tawsifurrahman/tuberculosis-tb-chest-xray-dataset).
- **Adult dataset:** COVID-19 Radiography Database, Chowdhury et al. 2020 and Rahman et al. 2021, Kaggle (tawsifurrahman/covid19-radiography-database). It includes RSNA Pneumonia Detection Challenge images.
- **Methods:** shortcut learning in COVID-19 X-ray AI (DeGrave et al., 2021), FedAvg (McMahan et al., 2017), Grad-CAM (Selvaraju et al., 2017), temperature scaling (Guo et al., 2017).
- **Libraries:** PyTorch, torchvision, scikit-learn, pandas, matplotlib, Gradio.
- **No pretrained models.**
- **AI tools:** Claude (Anthropic) helped plan the project and draft code and this write-up. [Add any others.]

## How to run
- **Full project:** open `pneumotype.ipynb` on Kaggle and add the three datasets above as inputs. Turn on GPU T4 and Internet, then Run All. It takes about 90 minutes.
- **Demo only:** open `pneumotype_demo.ipynb` and add the PneumoType notebook's output (the trained weights) as an input. It launches the Gradio app in about 2 minutes.
