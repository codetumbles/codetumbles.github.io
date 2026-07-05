---
title: "Unit Tests in Machine Learning Code"
date: 2025-03-17
last_updated: 2025-03-17
categories: [ml, unit_test]
tags:
  - machine-learning
  - unit-tests
  - statistics
toc: true
toc_levels: 2..3
---

Unlike traditional software, ML code has a higher likelihood of being non-deterministic (running the same code multiple times **with the same inputs and configuration** can produce different outputs) due to factors like random weight initialization, data shuffling, random batch sampling in stochastic algorithms, dropout layers, multi-threading / multi-GPU training, etc. Non-deterministic nature of machine learning process makes it hard to follow conventional unit testing practice. 

## Best Practices for Unit Testing ML Code

### 1. Avoid Testing End-to-End Training

Unit test should be focused on small, isolated functions. Testing the entire training process makes the test slow, hard to debug, and non-deterministic.

```python
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# wrong example
def test_model_output_shape():
    model = SimpleModel(5, 2)
    x = torch.randn(10, 5)  # Batch of 10 samples
    output = model(x)
```





### 2. Avoid self-fulfilling Tests 

It's stupid but we still end up doing it. Don't use the same function you are trying to test in the unit test itself

```python
import numpy as np

def normalize(data, mean, std):
    return (data - mean) / std

def test_normalize():
    data = np.array([10, 20, 30])
    mean, std = 20, 10
    
    # wrong example
    expected = (data - mean) / std
    
    # correct example
    expected = np.array([-1, 0, 1])  # Manually computed
    
    assert np.allclose(normalize(data, mean, std), expected)
```





### 3. Use Approx Comparison for Float Point Numbers

Directly comparing two float-point values can lead to false negatives, due to [precision error](https://learn.microsoft.com/en-us/cpp/build/why-floating-point-numbers-may-lose-precision?view=msvc-170).

```python
import numpy as np

def test_mse_loss():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    loss = np.mean((y_true - y_pred) ** 2)
    expected_loss = 0.0133

    # wrong example
    # assert loss == expected_loss
    
    # correct example 
    assert np.isclose(loss, expected_loss, atol=1e-3), "Loss values differ!"
```




### 4. Mock External Dependencies

Replace any code dependencies on external resources like DB, API, or cloud storage with mocks. Unit test should be testing your code, and shouldn't need to account for network failure, missing files on storage or data issues. 

```python
# Function that calls the external model API
def get_model_prediction(data):
    response = requests.post("https://ml-model.com/predict", json=data)
    return response.json()

class TestModelAPI(unittest.TestCase):
    @patch("requests.post")  # Mocking requests.post
    def test_get_model_prediction(self, mock_post):
        mock_post.return_value.json.return_value = {"prediction": 0.92}  # Mock response

        result = get_model_prediction({"feature1": 24, "feature2": 999})
        self.assertEqual(result, {"prediction": 0.92})
```





### 5. **Use Minimal & Synthetic Data**

Unit test are meant to be quick proxy for real world execution. Introducing huge datasets will make it slow, and resource intensive (*CI/CD servers are not usually built for resource-intensive tasks*)

```python
small_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```



### 6. Use Seed For Random Data
This ensures that your result are reproducible, and tests don't randomly fail.

```python
# wrong example
def test_random_data():
  	torch.manual_seed(42)
    x = torch.randn(10, 5)
		assert x.mean() < 1  # This may randomly fail

# correct example
def test_random_data():
  	torch.manual_seed(42)
    x = torch.randn(10, 5)
		assert x.mean() < 1
		
```



### 7. Include Edge Cases

It's convenient to include just the expected input and output in your unit test, and forgot to account for any edge cases but that makes your test non-robust.   

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Handle empty dataset
    if data.shape[0] == 0:
        return data
    # Handle missing values
    if np.isnan(data).any():
        raise ValueError("Input contains NaNs")
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def test_preprocess_data_edge_cases():
    # Edge Case 1: Empty Input
    empty_data = np.array([]).reshape(0, 2)
    processed_empty = preprocess_data(empty_data)
    assert processed_empty.shape == (0, 2), "Failed empty dataset test"

    # Edge Case 2: Single Feature
    single_feature = np.array([[1], [2], [3]])
    processed_single_feature = preprocess_data(single_feature)
    assert np.isclose(processed_single_feature.mean(), 0, atol=1e-6)
    assert np.isclose(processed_single_feature.std(), 1, atol=1e-6)

    # Edge Case 3: Single Sample
    single_sample = np.array([[5, 10]])
    processed_single_sample = preprocess_data(single_sample)
    assert processed_single_sample.shape == (1, 2)

    # Edge Case 4: Identical Rows (Zero Variance)
    identical_rows = np.array([[7, 7], [7, 7], [7, 7]])
    processed_identical = preprocess_data(identical_rows)
    assert np.allclose(processed_identical, 0), "Zero-variance feature should be standardized to zero"

    # Edge Case 5: Extreme Values
    extreme_values = np.array([[1e10, 1e-10], [-1e10, -1e-10], [0, 0]])
    processed_extreme = preprocess_data(extreme_values)
    assert processed_extreme.shape == (3, 2)

    # Edge Case 6: Missing Values (NaNs)
    nan_data = np.array([[1, 2], [np.nan, 3]])
    try:
        preprocess_data(nan_data)
        assert False, "Should raise ValueError on NaNs"
    except ValueError:
        pass

    print("All edge case tests passed!")

# Run tests
test_preprocess_data_edge_cases()
```
