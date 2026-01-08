# Examples

Practical examples in Jupyter Notebooks to learn DataFrameIt.

## Available Notebooks

Examples are organized from basic to advanced. We recommend following in order.

### 1. Basic
**[01_basic.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/01_basic.ipynb)**

Introduction to DataFrameIt with sentiment analysis.

- Create simple Pydantic model
- Process basic DataFrame
- Understand the output

```python
from pydantic import BaseModel
from typing import Literal

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
```

---

### 2. Error Handling
**[02_error_handling.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/02_error_handling.ipynb)**

How to handle errors and configure retry.

- Configure `max_retries`, `base_delay`, `max_delay`
- Check `_dataframeit_status` column
- Analyze `_error_details`

---

### 3. Incremental Processing
**[03_resume.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/03_resume.ipynb)**

Continue processing from where it stopped.

- Use `resume=True`
- Save and load partial results
- Reprocess only error rows

---

### 4. Custom Placeholder
**[04_custom_placeholder.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/04_custom_placeholder.ipynb)**

Control where text appears in the prompt.

- Use `{texto}` in template
- Create complex multi-part prompts

---

### 5. Advanced Case: Legal Analysis
**[05_advanced_legal.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/05_advanced_legal.ipynb)**

Real example with complex Pydantic model.

- Nested models
- Optional fields
- Lists of objects
- Multiple entity extraction

```python
class Party(BaseModel):
    name: str
    type: Literal['plaintiff', 'defendant']

class Decision(BaseModel):
    parties: List[Party]
    outcome: Literal['granted', 'denied']
```

---

### 6. Polars
**[06_polars.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/06_polars.ipynb)**

Use DataFrameIt with Polars instead of Pandas.

- Input with `polars.DataFrame`
- Output preserves Polars type

---

### 7. Multiple Data Types
**[07_multiple_data_types.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/07_multiple_data_types.ipynb)**

Process different input types.

- Lists
- Dictionaries
- Series

---

### 8. Rate Limiting
**[08_rate_limiting.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/08_rate_limiting.ipynb)**

Control request rate.

- Configure `rate_limit_delay`
- Use `parallel_requests`
- Combine for maximum efficiency

---

## Running the Examples

### 1. Clone the Repository

```bash
git clone https://github.com/bdcdo/dataframeit.git
cd dataframeit
```

### 2. Install Dependencies

```bash
pip install dataframeit[google]
pip install jupyter
```

### 3. Configure your API Key

```bash
export GOOGLE_API_KEY="your-key"
```

### 4. Run Jupyter

```bash
jupyter notebook example/
```

---

## Contributing Examples

If you created an interesting example, consider contributing! Open an issue or pull request on [GitHub](https://github.com/bdcdo/dataframeit).
