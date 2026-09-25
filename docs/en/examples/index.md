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
- Check `_dataframeit_status` column, which only appears when some row fails
- Analyze `_error_details`

---

### 3. Incremental Processing
**[03_resume.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/03_resume.ipynb)**

Continue processing from where it stopped.

- Save checkpoints with `batch_size` and `checkpoint_path`
- Reload with `read_df` and continue with `resume=True`
- Reprocess only error rows

---

### 4. Prompt and Text Column
**[04_custom_placeholder.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/04_custom_placeholder.ipynb)**

Control where text appears in the prompt and which column it comes from.

- Place `{texto}` in the template, or let the text go at the end
- Inference of `text_column` and when to set it
- Rows with empty text

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

### 9. Web Search
**[example_09_web_search.py](https://github.com/bdcdo/dataframeit/blob/main/example/example_09_web_search.py)**

Script with agent-based web search.

- `use_search`, `search_per_field` and `save_trace`
- Tavily or Exa as the search provider

---

## Running the Examples

### 1. Clone the Repository

```bash
git clone https://github.com/bdcdo/dataframeit.git
cd dataframeit
```

### 2. Install Dependencies

```bash
pip install dataframeit[openai]
pip install jupyter
```

### 3. Configure your API Key

```bash
export OPENAI_API_KEY="your-key"
```

### 4. Run Jupyter

```bash
jupyter notebook example/
```

---

## Contributing Examples

If you created an interesting example, consider contributing! Open an issue or pull request on [GitHub](https://github.com/bdcdo/dataframeit).
