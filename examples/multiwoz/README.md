## Multiwoz experiments

*Generate synthetic data*
```python
# All single-domain data in data/standard
# Under data_augment folder generate synthetic data
python3 augmented_restaurant_train5.py
# Under multiwoz folder delexicalize synthetic data
python3 create_delex_data_forconstructed.py
# Under multiwoz folder convert synthetic data to soloist format
python3 create_soloist_data.py
```
*Evaluating*
```bash
# under example/multiwoz folder
sh evaluate_singledomainmodels.sh
```
<code>MODEL_PATH </code>: Path of multiple checkpoints containing decoded file.
<code>MODE </code>: valid or test