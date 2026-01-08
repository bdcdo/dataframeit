# Structured Output

Learn to create advanced Pydantic models for complex extractions.

## Field Types

### Required Fields

```python
from pydantic import BaseModel, Field

class Model(BaseModel):
    # Required without default
    title: str = Field(description="Document title")

    # Required with validation
    score: int = Field(ge=1, le=10, description="Score from 1 to 10")
```

### Optional Fields

```python
from typing import Optional

class Model(BaseModel):
    # Optional - can be None
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes, if any"
    )
```

### Fixed Value Fields (Literal)

```python
from typing import Literal

class Model(BaseModel):
    # Only accepts these values
    priority: Literal['low', 'medium', 'high', 'critical']

    # Multiple options
    status: Literal['pending', 'in_progress', 'completed', 'cancelled']
```

!!! tip "When to use Literal"
    Use `Literal` whenever possible values are known and finite. This:

    - Forces the LLM to choose from valid options
    - Avoids unwanted variations (e.g., "High" vs "high" vs "HIGH")
    - Makes subsequent analysis easier

### Lists

```python
from typing import List

class Model(BaseModel):
    # List of strings
    tags: List[str] = Field(description="Relevant tags")

    # List of objects
    items: List[Item] = Field(description="List of extracted items")
```

## Nested Models

For complex structures, use nested models:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: Optional[str] = None

class Contact(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Address] = None

class Company(BaseModel):
    name: str
    tax_id: Optional[str] = None
    contacts: List[Contact] = Field(description="List of company contacts")
    sector: Literal['technology', 'health', 'finance', 'retail', 'other']
```

## Real Example: Legal Analysis

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Party(BaseModel):
    """Party involved in the case."""
    name: str = Field(description="Full name of the party")
    type: Literal['plaintiff', 'defendant', 'third_party'] = Field(description="Party type")
    tax_id: Optional[str] = Field(default=None, description="Tax ID")

class Claim(BaseModel):
    """Claim made in the case."""
    description: str = Field(description="Claim description")
    amount: Optional[float] = Field(default=None, description="Amount in USD")
    granted: Optional[bool] = Field(default=None, description="Whether it was granted")

class CourtDecision(BaseModel):
    """Complete analysis of a court decision."""

    # Identification
    case_number: str = Field(description="Case number")
    court: str = Field(description="Court (e.g., Supreme Court, District Court)")
    decision_date: str = Field(description="Decision date (YYYY-MM-DD)")

    # Parties
    parties: List[Party] = Field(description="Parties involved")

    # Merit
    decision_type: Literal['judgment', 'ruling', 'order', 'interlocutory']
    outcome: Literal['granted', 'denied', 'partially_granted', 'dismissed']

    # Claims
    claims: List[Claim] = Field(description="Claims analyzed")

    # Summary
    summary: str = Field(description="Decision summary in up to 100 words")
    legal_grounds: List[str] = Field(description="Main legal grounds")

PROMPT = """
Analyze the court decision below and extract all relevant information.
Be precise with dates, amounts, and names.
If information is not available, use null.
"""

result = dataframeit(df_decisions, CourtDecision, PROMPT, text_column='text')
```

## Custom Validations

```python
from pydantic import BaseModel, Field, field_validator

class Document(BaseModel):
    ssn: str = Field(description="SSN in format XXX-XX-XXXX")
    email: str = Field(description="Valid email")

    @field_validator('ssn')
    @classmethod
    def validate_ssn(cls, v):
        # Remove non-numeric characters
        numbers = ''.join(filter(str.isdigit, v))
        if len(numbers) != 9:
            raise ValueError('SSN must have 9 digits')
        return v
```

!!! warning "Be careful with validations"
    Very restrictive validations can cause frequent errors. Use sparingly.

## Best Practices

1. **Use clear descriptions**: The LLM uses descriptions to understand what to extract

2. **Prefer Literal over str**: When values are known, use `Literal`

3. **Use Optional for uncertain fields**: If information may not exist, mark as `Optional`

4. **Break down complex models**: Use nested models for better organization

5. **Test with examples**: Validate your model with real texts before processing everything
