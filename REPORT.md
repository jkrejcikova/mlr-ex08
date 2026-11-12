# REPORT.md – Task 4

## 1. Setup
I ran the experiments using our group's dataset (107 train, 29 val) and last year's larger dataset (700 train, 120 val). I used my Task 3 LeafNet as the baseline. The agent wrote the code based on my Task 3 pipeline, and I made sure it stayed simple. I measured three things for each variant: (a) accuracy on our data/leaves/val, (b) accuracy on a camera-like validation set (the validation images randomly cropped, rotated and with changed colors), and (c) accuracy on last year's validation photos (unseen data).

## 2. Results
Accuracy in %, mean over 3 seeds.

| Variant | (a) | (b) | (c) |
| :--- | :--- | :--- | :--- |
| Baseline (LeafNet, 15 epochs) | 67.8 | 35.6 | 36.7 |
| 1. geometric augmentation | 58.6 | 37.9 | 33.6 |
| 2. geometric aug. + ColorJitter | 66.7 | 56.3 | 37.2 |
| 3. batch normalization | 72.4 | 42.5 | 40.0 |
| 4. dropout | 60.9 | 39.1 | 37.5 |
| 5. global average pooling | 47.1 | 27.6 | 34.4 |
| 6. AdamW | 66.7 | 37.9 | 37.2 |
| 7. MobileNet, only last layer trained | 66.7 | 64.4 | 43.1 |
| 8. MobileNet, whole net fine-tuned | 55.2 | 62.1 | 39.7 |
| 9a. 8 + geometric augmentation | 63.2 | 64.4 | 46.1 |
| 9b. 9a + cosine LR, 30 epochs | 74.7 | 67.8 | 52.2 |
| 11. label smoothing | 67.8 | 36.8 | 36.1 |
| 12. weights of the best epoch | 70.1 | 36.8 | 36.7 |
| 10. 9b + last year's photos for training | 78.2 | 72.4 | 91.4* |
| A. frozen MobileNet + aug + ColorJitter + cosine | 77.0 | 71.3 | 41.1 |
| B. 8 + geometric aug., 30 epochs, constant LR | 71.3 | 67.8 | 48.1 |
| C. LeafNetBN + aug + ColorJitter + cosine | 67.8 | 70.1 | 47.2 |
| D. 8 + aug + ColorJitter + cosine, 30 epochs | 71.3 | 71.3 | 62.5 |

_Note (*): variant 10 was trained on last year's photos too, so (c) is not a fair test for it. I do not compare it directly with the other rows._

## 3. Noise
The validation set has only 29 images, so one image is about 3.4 points. This causes a lot of noise. The baseline alone changed between seeds by up to 14 points on (a). I only count a difference as real when the ranges of the 3 seeds do not overlap. Set (c) has 120 images and is much more reliable.

## 4. What helped and what did not
Dropout, AdamW, label smoothing, and picking the best epoch stayed within the noise of the baseline. Global average pooling made the model worse. Batch normalization reached 100% training accuracy (overfitting), and its small gains on validation were just noise.

ColorJitter raised (b) from 37.9% to 56.3%, but since (b) contains color changes, it just tests exactly what this augmentation trains. On (c), there was no clear gain. The pretrained MobileNet was the biggest real gain, pushing (b) to 64.4% and (c) to 43.1%. Augmentation with a cosine scheduler and 30 epochs (Variant 9b) reached 52.2% on (c).

## 5. The model for the phone
I chose variant D (fine-tuned MobileNet with geometric augmentation, ColorJitter, and cosine scheduler, 30 epochs) because it reached 62.5% on last year's photos (c), which is clearly better than 9b. I picked D using set (c), so its 62.5% is a bit optimistic.

Variant 10 used more training data and got slightly higher numbers on (a) and (b), but the ranges overlap with the other models, so I cannot prove it is actually better. Its 91.4% on (c) cannot be used because it was trained on photos from that same session. So, more data might help, but this experiment doesn't prove it.

## 6. Test of the exported model
I exported the seed 1 model of variant D to ONNX. The agent checked it in ONNX Runtime: the classes, mean, and std match the training code, and the ONNX and PyTorch probabilities differ by at most 0.000005, which means the exported file is correct. Because seed 1 was explicitly picked as the best of the three runs, its own accuracy numbers are optimistic; the fair estimate is the mean in the table.

The agent also tested 8 Google photos (2 per species) in ONNX Runtime, and 5 of 8 were correct (apples were predicted as cherry, and a maple as chestnut). When I tested the model myself in the course camera app, I saw the exact same thing: apple was mostly classified as cherry.

The model mixes up apple and cherry in both directions (on last year's photos, 8 of 30 apples were predicted as cherry, and 6 of 30 cherries as apple). I checked the dataset folders by eye and found some leaves in the wrong folders. Since apple and cherry only have 28 training photos each, even a few mislabeled images add a lot of noise. The two species look very similar anyway. To improve the model, we mainly need a cleaner and larger dataset.