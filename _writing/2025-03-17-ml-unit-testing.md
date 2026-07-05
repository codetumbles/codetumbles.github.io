---
title: "Unit Tests in Machine Learning Code"
date: 2025-03-17
last_updated: 2026-07-05
categories: [ml, unit_test]
tags:
  - machine-learning
  - unit-tests
  - statistics
toc: true
toc_levels: 2..3
---

Unlike traditional software, ML code has a higher likelihood of being non-deterministic (running the same code multiple times **with the same inputs and configuration** can produce different outputs) due to factors like random weight initialization, data shuffling, random batch sampling in stochastic algorithms, dropout layers, multi-threading / multi-GPU training, etc. Non-deterministic nature of machine learning process makes it hard to follow conventional unit testing practice.

Another problem: ML bugs often don't crash. A shape mismatch might broadcast silently. A disconnected layer might still return a valid-looking loss. You find out hours later. You can't unit test *learning*, but you can unit test the plumbing — shapes, gradients, loss going down on a tiny batch, that kind of thing.

## Best Practices for Unit Testing ML Code

### 1. Test Shapes and Dtypes, Not End-to-End Training

Unit tests should be focused on small, isolated functions. Testing the entire training process makes the test slow, hard to debug, and non-deterministic. What you actually want is to verify that data flows through your model correctly.

Most ML bugs I've hit were shape or dtype mismatches, not fancy statistical failures. Assert `output.shape` and `output.dtype` for a known input. Use a batch size > 1 — some bugs only show up when a dimension isn't accidentally squeezed away.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# wrong example — runs forward but asserts nothing
def test_model_output_shape():
    model = SimpleModel(5, 2)
    x = torch.randn(10, 5)
    output = model(x)

# correct example
def test_model_output_shape():
    model = SimpleModel(5, 2)
    x = torch.testing.make_tensor((10, 5), dtype=torch.float32, device="cpu")
    output = model(x)
    assert output.shape == (10, 2)
    assert output.dtype == torch.float32
```

### 2. Check That Gradients Actually Flow

A layer can pass a shape test and still not train. Maybe you detached a tensor, or a branch skipped a parameter. After `loss.backward()`, every trainable parameter should have a non-None, finite gradient. This has no analogue in regular app code, and it's one of the most useful ML-specific tests you can write.

```python
def test_model_gradients_flow():
    model = SimpleModel(5, 2)
    x = torch.randn(4, 5, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
```

### 3. Overfit a Single Batch

This is the sanity check between a unit test and an integration test. Take 2–4 fixed examples, turn off dropout and augmentation, train for a hundred steps or so, and check that loss goes near zero. If the model can't memorize a handful of examples, something is wrong in your loss, labels, optimizer, or gradient path — not model capacity. [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) calls this out explicitly.

```python
def test_model_can_overfit_single_batch():
    torch.manual_seed(0)
    model = SimpleModel(4, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
    y = torch.tensor([0, 1])

    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 1e-2
```

Keep the batch tiny and cap steps so it stays fast in CI. Passing doesn't mean your model generalizes — it means the training stack is probably wired correctly.

### 4. Avoid self-fulfilling Tests

It's stupid but we still end up doing it. Don't use the same function you are trying to test in the unit test itself.

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

### 5. Use Approx Comparison for Floating-Point Numbers

Directly comparing two floating-point values can lead to false negatives, due to [precision error](https://learn.microsoft.com/en-us/cpp/build/why-floating-point-numbers-may-lose-precision?view=msvc-170). For PyTorch tensors, `torch.testing.assert_close` is better than `==` — it handles tolerances and checks dtype.

```python
import numpy as np
import torch

def test_mse_loss():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    loss = np.mean((y_true - y_pred) ** 2)
    expected_loss = 0.0133

    # wrong example
    # assert loss == expected_loss

    # correct example (NumPy)
    assert np.isclose(loss, expected_loss, atol=1e-3), "Loss values differ!"

    # correct example (PyTorch)
    torch.testing.assert_close(
        torch.tensor(loss), torch.tensor(expected_loss), atol=1e-3, rtol=0
    )
```

### 6. Mock External Dependencies

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

### 7. Use Minimal & Synthetic Data

Unit tests are meant to be quick proxy for real world execution. Introducing huge datasets will make it slow, and resource intensive (*CI/CD servers are not usually built for resource-intensive tasks*).

```python
# wrong example — loads real data, slow in CI
# train_df = pd.read_parquet("s3://bucket/50gb/train.parquet")

# correct example
small_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# or with explicit shape/dtype
x = torch.testing.make_tensor((2, 4), dtype=torch.float32, device="cpu")
```

### 8. Use Seed For Random Data

This ensures that your results are reproducible, and tests don't randomly fail. If you don't need randomness at all, test a deterministic property (like shape) instead.

```python
# wrong example — no seed, assertion depends on the draw
def test_random_data():
    x = torch.randn(10, 5)
    assert x.mean() < 1  # This may randomly fail

# correct example — test something deterministic
def test_random_data_shape():
    x = torch.randn(10, 5)
    assert x.shape == (10, 5)

# correct example — seed when you need a fixed draw
def test_seeded_data_is_reproducible():
    torch.manual_seed(42)
    x1 = torch.randn(3, 3)
    torch.manual_seed(42)
    x2 = torch.randn(3, 3)
    torch.testing.assert_close(x1, x2)
```

Note: seeding fixes reproducibility in one process. Full GPU determinism needs extra flags and isn't always worth it for unit tests.

### 9. Include Edge Cases

It's convenient to include just the expected input and output in your unit test, and forget to account for any edge cases but that makes your test non-robust. In ML, the painful ones are often silent broadcasting and NaNs poisoning your loss.

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
```

Watch out for silent broadcasting too — PyTorch won't always error when shapes are compatible but not what you intended:

```python
def test_broadcasting_trap():
    # (batch, features) + (features,) is usually fine
    x = torch.ones(4, 10)
    bias = torch.zeros(10)
    assert (x + bias).shape == (4, 10)

    # (batch, seq, features) + (seq, features) broadcasts — is that what you want?
    x = torch.ones(4, 8, 10)
    bias = torch.zeros(8, 10)
    assert (x + bias).shape == (4, 8, 10)
```

## Wrapping Up

Unit tests in ML aren't about proving your model generalizes. They're about catching broken plumbing before you burn GPU hours — wrong shapes, dead gradients, a training loop that can't overfit two examples. Get those right, then worry about the science.
