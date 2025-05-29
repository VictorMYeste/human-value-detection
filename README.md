# human-value-detection

## Prediction

```bash
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline
```

## Evaluation

```bash
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline
```