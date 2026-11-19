# Brain Tumor MRI Classifier (practice project)

**Live demo:** https://brain-mri-classifier-va8vyaxo6vu45zyxzsihdj.streamlit.app/

> **Educational demo only. Not a medical device. Do not use it for diagnosis.**

A ResNet18 fine-tuned to classify brain MRI images into four classes:
**glioma, meningioma, no tumor, pituitary**. Built as a learning project to practice the full
machine learning pipeline: data checking, training, honest evaluation and deployment.

## Result

| Evaluation | Accuracy |
|---|---|
| Locked half of the Testing folder (800 images, run once) | **90.2%** (722 / 800) |

| Class | Recall (locked half) |
|---|---|
| glioma | 0.835 |
| meningioma | 0.850 |
| no tumor | 0.965 |
| pituitary | 0.960 |

The score is probably a point or two optimistic (see limitations). The uncertainty on 800 images
is roughly +/- 2 points.

## How it was built

1. **Data:** a public Kaggle brain MRI dataset (about 7,200 images, four balanced classes).
2. **Data cleaning:** a duplicate check found that about **26% of the test images (411 of 1,600)** had a
   near-identical copy in the training set. I removed those copies from training (and duplicates inside
   training), keeping 4,607 of 5,600 training images, so the test score is not inflated.
3. **First model:** ResNet18 (pretrained), last block unfrozen, class weights. It scored 89.7% on the cleaned
   test set but called many real tumors "no tumor".
4. **Diagnosis:** the missed tumors were often scans with a different style (brighter, tighter crop). The model
   was likely relying on scan style, not only on the tumor (likely explanation, not proven).
5. **Fix:** stronger augmentation (random zoom and crop, larger brightness and contrast changes) and no class
   weights. Across three seeds, tumors called "no tumor" dropped from 59 to 16-18 on a 400-image tuning half.
6. **Honest evaluation:** the Testing folder was split into a tuning half and a locked half. The locked half was
   used once, on the final model.

## How reliable is the confidence number?

Checked on the 800 locked images (dataset-style scans only):

| Confidence | Images | Accuracy |
|---|---|---|
| under 0.60 | 14 | 28.6% |
| 0.60 - 0.80 | 34 | 44.1% |
| 0.80 - 0.95 | 51 | 64.7% |
| 0.95 - 0.99 | 64 | 82.8% |
| 0.99 and above | 637 | 96.9% |

Higher confidence goes with higher accuracy on these images, but the model can still be confidently wrong
(about 20 errors at 0.99+), mostly tumors called "no tumor". The app warns whenever confidence is below 0.99.
That cutoff was chosen after looking at this table, so it is mildly tuned on the locked half.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model file `mri_C_both_half.pth` stores the weights in 16-bit to save space. The app converts them back to
full precision on load (verified: 0 different predictions out of 800 images against the full-size file).

## Files

- `app.py`: the Streamlit app
- `requirements.txt`: dependencies
- `mri_C_both_half.pth`: trained model weights

## Limitations

- **No patient IDs** in the dataset, so scans from the same patient may appear in both training and test.
- The duplicate check is crude (64x64 pixel distance), so some copies may remain.
- The Testing folder was split after I had seen the errors on the full test set, so the locked-half score is mildly optimistic.
- Glioma and meningioma are often confused with each other, and some tumors are still called "no tumor".
- The model always picks one of four classes, even for images that are not brain MRIs. Example: a cat photo was
  classified as **meningioma with about 100% confidence**. Confidence is not the same as being correct, and there
  is no check for out-of-scope inputs.
- **Not externally validated:** I could not find labeled brain MRIs from another source to test on, so all results
  come from one dataset.
- It may fail on scans that look different from its training data.
- Not validated on clinical data and not for medical use.

## Data

Trained on the Brain Tumor MRI Dataset by Masoud Nickparvar on Kaggle. Check the dataset page for its license
and original sources.
