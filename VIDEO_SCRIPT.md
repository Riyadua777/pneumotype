# PneumoType: 2-minute demo video script

Record your screen with your voice over it. Numbers in [brackets] get updated from the final Kaggle run (section 21, "Numbers for your write-up").

---

## 0:00–0:15 · The problem (show: title slide or the demo app)
> "Pneumonia is the biggest infectious killer of children under five. When a child comes into a clinic coughing with a fever, the doctor has to decide fast: is it bacterial, which needs antibiotics, or viral, which doesn't? Guess wrong, and a child either misses treatment or gets antibiotics they don't need."

## 0:15–1:15 · The demo (show: Gradio app)
1. Click a **bacterial** example X-ray.
   > "We built PneumoType. A doctor uploads a chest X-ray, and in about a second it says whether there's pneumonia, what type, and how confident it is."
2. Point at the **heatmap**.
   > "This heatmap shows where the model looked, so a doctor can check it's focusing on the lungs."
3. Point at the **"pneumonia check for anyone"** line.
   > "It also works on adults. We trained it across a children's hospital and an adult hospital using federated learning, so neither hospital ever had to share a single patient X-ray. Only model weights were shared."
4. Upload or click a harder case that comes back **UNSURE**.
   > "And when it isn't confident, it doesn't guess. It says: refer to a doctor. That's on purpose."

## 1:15–1:45 · How it works and how well (show: the results table and the robustness/federated charts)
> "Everything is our own convolutional neural network, trained from scratch, with no pretrained models. On [799] X-rays it had never seen, it caught [96.5%] of pneumonia cases, compared with [77%] for a simple baseline, with only [9] false alarms out of [222] healthy X-rays. We tuned it so the most dangerous mistake, missing a bacterial case, dropped from [58] to [23]. Federated learning reached [X]% balanced accuracy on both children and adults, close to pooling all the data, without sharing any."

## 1:45–2:00 · Honest limits and next steps (show: shortcut test or failure examples)
> "It's not perfect. Bacterial versus viral is hard even for radiologists, and we caught our first model failing on lower-quality X-rays, so we trained a second, more robust one. We also tested whether it was cheating by blacking out the lungs. Next, we'd test it at a real clinic with more hospitals in the federation. This is a prototype, not medical advice."

---

### Recording tips
- Do one practice run, then record. Don't edit; one clean take is plenty.
- Zoom the browser to 125% so the judges can read the numbers.
- Upload to YouTube as **Unlisted**, then open the link in an incognito window to check it works.
