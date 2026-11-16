# PneumoType: 2-minute demo video script

Record your screen with your voice over it. All numbers are from the final Kaggle run (section 21 of the notebook).

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
4. Click the **3rd example** X-ray (a bacterial case). The model leans viral, so it says **UNSURE**.
   > "And when it isn't confident, it doesn't guess. It says: refer to a doctor. That's on purpose."

## 1:15–1:45 · How it works and how well (show: the results table, then the federated chart)
> "Everything is our own neural network, trained from scratch. On 799 X-rays it had never seen, it caught 95% of pneumonia cases, with only 8 false alarms out of 222 healthy children. A simple baseline missed 92 bacterial cases; with our safety tuning we miss 22. When it's at least 90% confident, it's right 98% of the time. And for children and adults together, federated learning gave the best worst-case result of any setup that keeps X-rays private: 83% balanced accuracy on both."

## 1:45–2:00 · The twist: we caught a model cheating (show: figures/shortcut_tb.png)
> "We also tried tuberculosis. It looked 91% accurate, but when we blacked out the lungs it still 'found' TB 100% of the time. It was recognising which hospital the X-ray came from, not the disease. Our pneumonia models pass that test, so we left TB out of the demo. Next: TB data from many hospitals, and testing at a real clinic. This is a prototype, not medical advice."

---

### Recording tips
- Do one practice run, then record. Don't edit; one clean take is plenty.
- Zoom the browser to 125% so the judges can read the numbers.
- Upload to YouTube as **Unlisted**, then open the link in an incognito window to check it works.
