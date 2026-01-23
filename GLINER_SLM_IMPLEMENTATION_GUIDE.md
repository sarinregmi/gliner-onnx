# GLiNER + SLM PII/PHI Detection Integration - Claude Code Implementation Guide

## Overview

This document provides a complete implementation specification for integrating GLiNER-based NER detection and SLM context analysis into an existing FastAPI + Presidio PII/PHI detection system. Import this file into Claude Code to execute the implementation.

**Architecture**: Three-layer hybrid detection:
- **Layer 1 (Presidio)**: Default Presidio recognizers ONLY (no custom recognizers)
- **Layer 2 (GLiNER)**: Context-aware NER for entities Presidio handles poorly
- **Layer 3 (SLM)**: Semantic analysis for complex detections (attorney-client, trade secrets)

---

## Critical Design Decision: Detector-to-Layer Mapping

### Layer 1: Presidio (DEFAULT Recognizers Only)

Presidio handles ONLY its built-in recognizers with checksum validation and high-precision patterns. **Do NOT add custom recognizers for healthcare codes (ICD-10, NPI, etc.)** - those go to GLiNER.

| Presidio Entity | Description | Why Presidio |
|-----------------|-------------|--------------|
| `CREDIT_CARD` | Credit card numbers | Luhn checksum validation |
| `US_SSN` | Social Security Numbers | Format + checksum |
| `US_ITIN` | Individual Tax ID | Checksum validated |
| `US_PASSPORT` | US Passport number | Specific format |
| `US_DRIVER_LICENSE` | US Driver's License | State-specific patterns |
| `US_BANK_NUMBER` | US Bank account | Validated format |
| `IBAN_CODE` | International Bank Account | Checksum validated |
| `CRYPTO` | Cryptocurrency addresses | Format-specific |
| `EMAIL_ADDRESS` | Email addresses | Regex + validation |
| `PHONE_NUMBER` | Phone numbers | Format patterns |
| `IP_ADDRESS` | IP addresses | IPv4/IPv6 format |
| `URL` | URLs | Regex validation |
| `DATE_TIME` | Dates and times | Pattern matching |

**Presidio SpaCy NER entities** (lower accuracy, supplementary):
- `PERSON` - Names (GLiNER is better)
- `LOCATION` - Places (GLiNER is better)
- `NRP` - Nationality/Religious/Political groups

### Layer 2: GLiNER (Context-Aware NER)

GLiNER handles entities that need **semantic understanding** to avoid false positives. This includes ALL healthcare codes.

| Your Detector | GLiNER Labels | Why GLiNER |
|---------------|---------------|------------|
| `full-name` / `person` | `person`, `name`, `full name` | Context distinguishes names from companies |
| `first-name` | `first name`, `name` | Part of name context |
| `last-name` | `last name`, `name` | Part of name context |
| `address` | `address`, `street address` | Full address vs partial location |
| `organization` / `company` | `organization`, `company` | Distinguish from person names |
| `date-of-birth` | `date of birth`, `dob` | Contextual date classification |
| `medical-record-number` | `medical record number`, `mrn` | Healthcare context |
| `diagnosis` | `diagnosis`, `medical condition` | Medical terminology |
| `medication` | `medication`, `drug`, `prescription` | Drug names in context |
| `icd10-code` | `icd10 code`, `diagnosis code` | **Medical codes (NOT Presidio)** |
| `npi` | `npi`, `national provider identifier` | **Healthcare provider ID** |
| `health-insurance-id` | `health insurance id`, `insurance number` | Insurance identification |
| `passport` | `passport number`, `passport` | Various country formats |
| `driver-license` | `driver license`, `driving license` | Non-US formats |
| `bank-account` | `bank account`, `account number` | Non-US formats |
| `national-id` | `national id`, `government id` | International IDs |

### Layer 3: SLM (Semantic Understanding)

The SLM (Phi-4-mini 3.8B or Gemma 3-4B) handles detections requiring **semantic reasoning**.

| Detector | Why SLM Required |
|----------|-----------------|
| `attorney-client-privilege` | Requires understanding legal relationship, communication intent |
| `trade-secret` | Needs business context, proprietary nature assessment |
| `confidential-business-info` | Semantic classification beyond keywords |
| `work-product-doctrine` | Legal document classification |
| `custom-sensitive-patterns` | User-defined semantic patterns |
| `source-code-detection` | Language detection, proprietary vs open source |
| `database-schema` | Structural understanding of data patterns |

---

## Target File Structure

```
app/detections/
├── __init__.py
├── base.py                        # DetectionService ABC (existing)
├── models.py                      # Detection models (MODIFY - add source tracking)
├── config.py                      # Settings (MODIFY - add GLiNER + SLM config)
│
├── presidio_service.py            # Existing Presidio (DEFAULT recognizers only)
├── gliner_service.py              # NEW: GLiNER NER detection
├── slm_service.py                 # NEW: SLM semantic detection
├── hybrid_service.py              # NEW: Orchestrates all three layers
│
├── entity_merger.py               # NEW: Entity deduplication/merging
├── model_manager.py               # NEW: GPU model lifecycle management
│
├── dependencies.py                # MODIFY: Update DI for hybrid service
├── service_loader.py              # MODIFY: Initialize HybridDetectionService
│
└── tests/
    ├── conftest.py                # NEW: Shared fixtures
    ├── test_gliner_service.py     # NEW
    ├── test_slm_service.py        # NEW
    ├── test_hybrid_service.py     # NEW
    └── test_entity_merger.py      # NEW
```

---

## Implementation Tasks

### TASK 1: Add Dependencies

**File**: `requirements.txt`

**Action**: Add these dependencies:

```text
# GLiNER for Layer 2 NER detection
gliner==0.2.7
torch>=2.0.0,<2.3.0
transformers>=4.38.0,<4.46.0

# SLM inference for Layer 3
vllm>=0.4.0  # For production SLM inference
# OR for lighter deployments:
# llama-cpp-python>=0.2.50  # CPU/Metal inference

# Optional: Better GPU memory management
accelerate>=0.27.0
```

**Note**: Choose between `vllm` (GPU, high throughput) or `llama-cpp-python` (CPU/edge) based on deployment.

---

### TASK 2: Create Model Manager

**File**: `app/detections/model_manager.py`

**Purpose**: Singleton for GPU model lifecycle management. Handles both GLiNER and SLM models.

**Key Components**:
- Lazy loading (load on first use)
- Device detection (CUDA/CPU fallback)
- Memory management with `torch.cuda.empty_cache()`
- Thread-safe singleton pattern
- Health check endpoints
- Support for multiple model types (GLiNER, SLM)

**Implementation Requirements**:
```python
class ModelManager:
    """
    Manages GLiNER and SLM model lifecycle.
    
    Key methods:
    - get_gliner_model(model_name: str) -> GLiNER
    - get_slm_model(model_name: str) -> LLM
    - unload_model(model_name: str)
    - get_gpu_memory_usage() -> dict
    - health_check() -> dict
    
    Configuration via environment:
    - GLINER_MODEL_NAME: default "nvidia/gliner-PII"
    - SLM_MODEL_NAME: default "microsoft/Phi-4-mini-instruct"
    - DETECTION_DEVICE: "cuda" or "cpu"
    - ENABLE_SLM: "true" or "false"
    """
```

---

### TASK 3: Create GLiNER Service (Layer 2)

**File**: `app/detections/gliner_service.py`

**Purpose**: NER-based entity detection for context-dependent entities.

**Model**: Use `nvidia/gliner-PII` (your tested model) initially. Can swap to `knowledgator/gliner-pii-base-v1.0` if accuracy testing supports it.

**Key Implementation Details**:

```python
# Detector to GLiNER label mapping
DETECTOR_TO_GLINER_LABELS: dict[str, list[str]] = {
    # Personal identifiers (GLiNER excels here)
    "person": ["person", "name", "full name"],
    "full-name": ["person", "name", "full name"],
    "first-name": ["first name", "name"],
    "last-name": ["last name", "name"],
    
    # Contact (GLiNER for context-aware detection)
    "address": ["address", "street address", "location"],
    
    # Organizations
    "organization": ["organization", "company", "institution"],
    "company": ["company", "organization", "business"],
    
    # Dates with context
    "date-of-birth": ["date of birth", "birthdate", "dob"],
    
    # Healthcare (ALL healthcare goes to GLiNER, NOT Presidio)
    "medical-record-number": ["medical record number", "mrn", "patient id"],
    "diagnosis": ["diagnosis", "medical condition", "disease", "disorder"],
    "medication": ["medication", "drug", "prescription", "medicine"],
    "icd10-code": ["icd10 code", "icd-10", "diagnosis code", "medical code"],
    "npi": ["npi", "national provider identifier", "provider number"],
    "health-insurance-id": ["health insurance id", "insurance number", "member id"],
    
    # Financial (international/non-checksummed)
    "passport": ["passport number", "passport"],
    "driver-license": ["driver license", "driving license"],
    "national-id": ["national id", "government id", "national identification"],
    "bank-account": ["bank account", "account number", "iban"],
    
    # Technical
    "username": ["username", "user id", "login"],
    "password": ["password", "credential"],
}

# Confidence threshold per entity type
GLINER_THRESHOLDS: dict[str, float] = {
    "person": 0.4,
    "address": 0.5,
    "organization": 0.4,
    "medical-record-number": 0.5,
    "icd10-code": 0.5,
    "diagnosis": 0.4,
    "medication": 0.4,
    # Default for unlisted: 0.35
}
```

**Class Interface**:
```python
class GlinerService(DetectionService):
    """
    Key methods:
    - async detect(text: str, guardrails: List[Guardrail]) -> DetectionResponse
    - _get_labels_for_guardrails(guardrails) -> List[str]
    - _map_predictions_to_entities(predictions, text) -> List[DetectedEntity]
    - _chunk_text(text: str, max_length: int = 512) -> List[TextChunk]
    
    Important: Handle texts > 512 tokens by chunking with overlap.
    """
```

---

### TASK 4: Create SLM Service (Layer 3)

**File**: `app/detections/slm_service.py`

**Purpose**: Semantic analysis for complex, context-dependent detections.

**Model Options**:
- Primary: `microsoft/Phi-4-mini-instruct` (3.8B, best reasoning)
- Alternative: `google/gemma-3-4b-it` (better quantization)
- Quantization: AWQ 4-bit for production

**Key Implementation Details**:

```python
# Detectors that require SLM (semantic understanding)
SLM_REQUIRED_DETECTORS: set[str] = {
    "attorney-client-privilege",
    "trade-secret",
    "confidential-business-info",
    "work-product-doctrine",
    "source-code-proprietary",
    "database-schema",
    "custom-sensitive-pattern",
}

# Prompt templates for each detector type
SLM_PROMPTS: dict[str, str] = {
    "attorney-client-privilege": '''Analyze the following text for attorney-client privileged communication.

Criteria for attorney-client privilege:
1. Communication between attorney and client
2. Made for purpose of legal advice
3. Intended to be confidential
4. Not shared with third parties

Text to analyze:
"""
{text}
"""

Respond in JSON format:
{{
    "is_privileged": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "spans": [
        {{"start": int, "end": int, "text": "matched text"}}
    ]
}}''',

    "trade-secret": '''Analyze the following text for potential trade secrets.

Criteria for trade secrets:
1. Information that derives economic value from being secret
2. Subject to reasonable efforts to maintain secrecy
3. Not generally known or readily ascertainable
4. Examples: formulas, processes, designs, customer lists, pricing strategies

Text to analyze:
"""
{text}
"""

Respond in JSON format:
{{
    "contains_trade_secret": true/false,
    "confidence": 0.0-1.0,
    "category": "formula|process|design|customer_data|pricing|other",
    "reasoning": "brief explanation",
    "spans": [
        {{"start": int, "end": int, "text": "matched text"}}
    ]
}}''',

    "confidential-business-info": '''Analyze the following text for confidential business information.

Consider:
1. Internal financial data or projections
2. Strategic plans or initiatives
3. Unreleased product information
4. Internal communications marked confidential
5. Merger/acquisition discussions

Text to analyze:
"""
{text}
"""

Respond in JSON format:
{{
    "is_confidential": true/false,
    "confidence": 0.0-1.0,
    "category": "financial|strategic|product|internal|m_and_a|other",
    "reasoning": "brief explanation"
}}''',
}
```

**Class Interface**:
```python
class SlmService:
    """
    SLM-based semantic detection service.
    
    Key methods:
    - async analyze(text: str, detector_type: str) -> SlmAnalysisResult
    - async batch_analyze(texts: List[str], detector_type: str) -> List[SlmAnalysisResult]
    - _build_prompt(text: str, detector_type: str) -> str
    - _parse_response(response: str, detector_type: str) -> SlmAnalysisResult
    
    Configuration:
    - SLM_MODEL_NAME: model identifier
    - SLM_MAX_TOKENS: default 512
    - SLM_TEMPERATURE: default 0.1 (low for classification)
    - SLM_TIMEOUT_SECONDS: default 5.0
    """
```

**vLLM Integration Pattern**:
```python
from vllm import LLM, SamplingParams

class SlmService:
    def __init__(self, model_name: str = "microsoft/Phi-4-mini-instruct"):
        self.llm = LLM(
            model=model_name,
            quantization="awq",  # 4-bit quantization
            gpu_memory_utilization=0.3,  # Reserve memory for GLiNER
            max_model_len=4096,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            stop=["```", "\n\n\n"],
        )
    
    async def analyze(self, text: str, detector_type: str) -> SlmAnalysisResult:
        prompt = self._build_prompt(text, detector_type)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return self._parse_response(outputs[0].outputs[0].text, detector_type)
```

**Alternative: llama-cpp for CPU/Edge**:
```python
from llama_cpp import Llama

class SlmService:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # Use all GPU layers if available
            verbose=False,
        )
```

---

### TASK 5: Create Entity Merger

**File**: `app/detections/entity_merger.py`

**Purpose**: Deduplicate and merge entities from all three layers.

**Merge Strategy**:
```python
class MergeStrategy(Enum):
    SMART = "smart"           # Layer-aware: GLiNER for NER, Presidio for patterns
    PREFER_GLINER = "prefer_gliner"
    PREFER_PRESIDIO = "prefer_presidio"
    HIGHEST_CONFIDENCE = "highest_confidence"
    UNION = "union"           # Keep all (no deduplication)

# Smart strategy entity routing
PRESIDIO_PREFERRED_ENTITIES: set[str] = {
    "credit-card", "ssn", "us-ssn", "itin", "iban", 
    "crypto", "us-passport", "us-driver-license", "us-bank-number"
}

GLINER_PREFERRED_ENTITIES: set[str] = {
    "person", "full-name", "first-name", "last-name",
    "organization", "company", "address", "location",
    "diagnosis", "medication", "icd10-code", "npi",
    "medical-record-number", "health-insurance-id",
    "date-of-birth"
}

SLM_ONLY_ENTITIES: set[str] = {
    "attorney-client-privilege", "trade-secret",
    "confidential-business-info", "work-product-doctrine"
}
```

**Class Interface**:
```python
class EntityMerger:
    """
    Key methods:
    - merge(
        presidio_entities: List[DetectedEntity],
        gliner_entities: List[DetectedEntity],
        slm_entities: List[DetectedEntity],
        strategy: MergeStrategy
      ) -> List[DetectedEntity]
    
    - _has_overlap(entity1, entity2) -> bool
    - _resolve_overlap(entity1, entity2, strategy) -> DetectedEntity
    - _calculate_overlap_ratio(entity1, entity2) -> float
    """
```

---

### TASK 6: Create Hybrid Service (Orchestrator)

**File**: `app/detections/hybrid_service.py`

**Purpose**: Orchestrates all three detection layers with parallel execution.

**Key Implementation**:
```python
class HybridDetectionService(DetectionService):
    """
    Orchestrates Layer 1 (Presidio) + Layer 2 (GLiNER) + Layer 3 (SLM).
    
    Execution strategy:
    1. Run Layer 1 and Layer 2 in PARALLEL
    2. Merge results
    3. Conditionally invoke Layer 3 for specific detector types
    4. Return unified detection response
    """
    
    def __init__(
        self,
        presidio_service: PresidioService,
        gliner_service: GlinerService,
        slm_service: Optional[SlmService],
        merger: EntityMerger,
        config: HybridConfig,
    ):
        self.presidio = presidio_service
        self.gliner = gliner_service
        self.slm = slm_service  # Optional, can be None
        self.merger = merger
        self.config = config
    
    async def detect(
        self,
        text: str,
        guardrails: List[Guardrail],
    ) -> DetectionResponse:
        start_time = time.perf_counter()
        
        # Separate guardrails by layer
        presidio_guardrails = self._filter_presidio_guardrails(guardrails)
        gliner_guardrails = self._filter_gliner_guardrails(guardrails)
        slm_guardrails = self._filter_slm_guardrails(guardrails)
        
        # Layer 1 + Layer 2: Run in parallel
        layer1_task = asyncio.create_task(
            self.presidio.detect(text, presidio_guardrails)
        )
        layer2_task = asyncio.create_task(
            self.gliner.detect(text, gliner_guardrails)
        )
        
        # Wait for both with timeout
        try:
            presidio_result, gliner_result = await asyncio.wait_for(
                asyncio.gather(layer1_task, layer2_task, return_exceptions=True),
                timeout=self.config.layer_timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning("Layer 1/2 timeout, using partial results")
            presidio_result = layer1_task.result() if layer1_task.done() else DetectionResponse()
            gliner_result = layer2_task.result() if layer2_task.done() else DetectionResponse()
        
        # Merge Layer 1 + Layer 2 results
        merged_entities = self.merger.merge(
            presidio_entities=presidio_result.entities if not isinstance(presidio_result, Exception) else [],
            gliner_entities=gliner_result.entities if not isinstance(gliner_result, Exception) else [],
            slm_entities=[],
            strategy=self.config.merge_strategy,
        )
        
        # Layer 3: Conditional SLM analysis
        if self.slm and slm_guardrails:
            slm_entities = await self._run_slm_analysis(text, slm_guardrails)
            merged_entities.extend(slm_entities)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return DetectionResponse(
            entities=merged_entities,
            processing_time_ms=processing_time,
            layers_used=self._get_layers_used(presidio_result, gliner_result, slm_guardrails),
        )
    
    def _filter_presidio_guardrails(self, guardrails: List[Guardrail]) -> List[Guardrail]:
        """Return guardrails that Presidio should handle (default recognizers only)."""
        presidio_detectors = {
            "credit-card", "ssn", "us-ssn", "itin", "us-itin",
            "us-passport", "us-driver-license", "us-bank-number",
            "iban", "crypto", "email", "email-address",
            "phone", "phone-number", "ip-address", "url"
        }
        return [g for g in guardrails if g.detector_id.lower() in presidio_detectors]
    
    def _filter_gliner_guardrails(self, guardrails: List[Guardrail]) -> List[Guardrail]:
        """Return guardrails that GLiNER should handle."""
        gliner_detectors = {
            "person", "full-name", "first-name", "last-name",
            "organization", "company", "address", "location",
            "date-of-birth", "medical-record-number", "mrn",
            "diagnosis", "medication", "icd10-code", "npi",
            "health-insurance-id", "passport", "driver-license",
            "national-id", "bank-account", "username"
        }
        return [g for g in guardrails if g.detector_id.lower() in gliner_detectors]
    
    def _filter_slm_guardrails(self, guardrails: List[Guardrail]) -> List[Guardrail]:
        """Return guardrails requiring SLM semantic analysis."""
        slm_detectors = {
            "attorney-client-privilege", "trade-secret",
            "confidential-business-info", "work-product-doctrine",
            "source-code-proprietary", "database-schema"
        }
        return [g for g in guardrails if g.detector_id.lower() in slm_detectors]
```

---

### TASK 7: Update Configuration

**File**: `app/detections/config.py`

**Add these configuration options**:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class DetectionSettings(BaseSettings):
    # Existing Presidio settings...
    
    # GLiNER Configuration (Layer 2)
    GLINER_MODEL_NAME: str = "nvidia/gliner-PII"
    GLINER_DEVICE: str = "cuda"  # or "cpu"
    GLINER_DEFAULT_THRESHOLD: float = 0.35
    GLINER_MAX_TEXT_LENGTH: int = 512
    GLINER_CHUNK_OVERLAP: int = 50
    
    # SLM Configuration (Layer 3)
    ENABLE_SLM: bool = True
    SLM_MODEL_NAME: str = "microsoft/Phi-4-mini-instruct"
    SLM_QUANTIZATION: str = "awq"  # awq, gptq, or none
    SLM_MAX_TOKENS: int = 512
    SLM_TEMPERATURE: float = 0.1
    SLM_TIMEOUT_SECONDS: float = 5.0
    SLM_GPU_MEMORY_FRACTION: float = 0.3  # Reserve for GLiNER
    
    # Hybrid Service Configuration
    DETECTION_MODE: str = "hybrid"  # "hybrid", "presidio", "gliner"
    MERGE_STRATEGY: str = "smart"  # "smart", "prefer_gliner", "highest_confidence"
    LAYER_TIMEOUT_SECONDS: float = 2.0
    ENABLE_PARALLEL_EXECUTION: bool = True
    MODEL_WARMUP_ON_STARTUP: bool = True
    
    class Config:
        env_prefix = "DETECTION_"
```

---

### TASK 8: Update Service Loader

**File**: `app/detections/service_loader.py`

**Update to initialize HybridDetectionService**:

```python
from .config import get_detection_settings
from .presidio_service import PresidioService
from .gliner_service import GlinerService
from .slm_service import SlmService
from .hybrid_service import HybridDetectionService
from .entity_merger import EntityMerger

def create_detection_service() -> DetectionService:
    """Factory function to create the appropriate detection service."""
    settings = get_detection_settings()
    
    if settings.DETECTION_MODE == "presidio":
        return PresidioService()
    
    if settings.DETECTION_MODE == "gliner":
        return GlinerService(
            model_name=settings.GLINER_MODEL_NAME,
            device=settings.GLINER_DEVICE,
        )
    
    # Default: Hybrid mode
    presidio = PresidioService()
    gliner = GlinerService(
        model_name=settings.GLINER_MODEL_NAME,
        device=settings.GLINER_DEVICE,
    )
    
    slm = None
    if settings.ENABLE_SLM:
        try:
            slm = SlmService(
                model_name=settings.SLM_MODEL_NAME,
                quantization=settings.SLM_QUANTIZATION,
            )
        except Exception as e:
            logger.warning(f"SLM initialization failed, disabling Layer 3: {e}")
    
    merger = EntityMerger(strategy=settings.MERGE_STRATEGY)
    
    return HybridDetectionService(
        presidio_service=presidio,
        gliner_service=gliner,
        slm_service=slm,
        merger=merger,
        config=HybridConfig(
            merge_strategy=settings.MERGE_STRATEGY,
            layer_timeout_seconds=settings.LAYER_TIMEOUT_SECONDS,
            enable_parallel=settings.ENABLE_PARALLEL_EXECUTION,
        ),
    )
```

---

### TASK 9: Update Dependencies (FastAPI DI)

**File**: `app/detections/dependencies.py`

**Update dependency injection**:

```python
from functools import lru_cache
from .service_loader import create_detection_service
from .model_manager import ModelManager

@lru_cache()
def get_detection_service() -> DetectionService:
    """Get singleton detection service instance."""
    return create_detection_service()

@lru_cache()
def get_model_manager() -> ModelManager:
    """Get singleton model manager instance."""
    return ModelManager.get_instance()
```

---

### TASK 10: Update Models

**File**: `app/detections/models.py`

**Add source tracking and SLM response models**:

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class DetectionSource(str, Enum):
    PRESIDIO = "presidio"
    GLINER = "gliner"
    SLM = "slm"

class DetectedEntity(BaseModel):
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)
    source: DetectionSource = DetectionSource.PRESIDIO
    model: Optional[str] = None  # e.g., "nvidia/gliner-PII"
    metadata: Optional[dict] = None  # For SLM reasoning, etc.

class SlmAnalysisResult(BaseModel):
    detected: bool
    confidence: float
    category: Optional[str] = None
    reasoning: Optional[str] = None
    spans: List[dict] = []  # [{"start": int, "end": int, "text": str}]

class DetectionResponse(BaseModel):
    entities: List[DetectedEntity] = []
    processing_time_ms: float = 0.0
    layers_used: List[str] = []
    error: Optional[str] = None
```

---

### TASK 11: Update main.py (Lifespan Events)

**File**: `app/main.py`

**Add model warmup on startup**:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .detections.dependencies import get_detection_service, get_model_manager
from .detections.config import get_detection_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_detection_settings()
    
    # Startup: Warm up models
    if settings.MODEL_WARMUP_ON_STARTUP:
        logger.info("Warming up detection models...")
        try:
            service = get_detection_service()
            # Trigger model loading with dummy text
            await service.detect(
                "Test warmup text for model loading.",
                guardrails=[]
            )
            logger.info("Model warmup complete")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
    
    yield
    
    # Shutdown: Clean up GPU memory
    logger.info("Shutting down, cleaning up models...")
    model_manager = get_model_manager()
    model_manager.cleanup()
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
```

---

### TASK 12: Create Tests

**File**: `app/detections/tests/conftest.py`

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_gliner_model():
    """Mock GLiNER model for testing without GPU."""
    model = MagicMock()
    model.predict_entities.return_value = [
        {"text": "John Doe", "label": "person", "start": 0, "end": 8, "score": 0.95}
    ]
    return model

@pytest.fixture
def mock_slm():
    """Mock SLM for testing without GPU."""
    slm = AsyncMock()
    slm.analyze.return_value = SlmAnalysisResult(
        detected=True,
        confidence=0.85,
        reasoning="Contains legal advice discussion"
    )
    return slm

@pytest.fixture
def sample_text_pii():
    return "John Doe's SSN is 123-45-6789 and email is john@example.com"

@pytest.fixture
def sample_text_phi():
    return "Patient MRN-123456 diagnosed with E11.9 (Type 2 diabetes), prescribed Metformin 500mg"

@pytest.fixture
def sample_text_legal():
    return "Dear Client, regarding your case, I advise that we proceed with the settlement offer..."
```

**File**: `app/detections/tests/test_gliner_service.py`

```python
import pytest
from ..gliner_service import GlinerService, DETECTOR_TO_GLINER_LABELS

class TestGlinerService:
    def test_label_mapping_exists_for_healthcare(self):
        """Ensure healthcare detectors have GLiNER label mappings."""
        healthcare_detectors = ["icd10-code", "npi", "diagnosis", "medication"]
        for detector in healthcare_detectors:
            assert detector in DETECTOR_TO_GLINER_LABELS
    
    def test_icd10_not_in_presidio_mapping(self):
        """ICD-10 should NOT be handled by Presidio."""
        from ..hybrid_service import HybridDetectionService
        # This test ensures ICD-10 is routed to GLiNER
        pass
    
    @pytest.mark.asyncio
    async def test_detect_person_name(self, mock_gliner_model, sample_text_pii):
        """Test person name detection."""
        service = GlinerService(model=mock_gliner_model)
        result = await service.detect(sample_text_pii, [Guardrail(detector_id="person")])
        assert len(result.entities) > 0
        assert result.entities[0].entity_type == "person"
    
    @pytest.mark.asyncio
    async def test_detect_healthcare_codes(self, mock_gliner_model, sample_text_phi):
        """Test ICD-10 code detection via GLiNER (not Presidio)."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "E11.9", "label": "icd10 code", "start": 35, "end": 40, "score": 0.92}
        ]
        service = GlinerService(model=mock_gliner_model)
        result = await service.detect(sample_text_phi, [Guardrail(detector_id="icd10-code")])
        assert len(result.entities) > 0
        assert result.entities[0].source == DetectionSource.GLINER
```

**File**: `app/detections/tests/test_hybrid_service.py`

```python
import pytest
from ..hybrid_service import HybridDetectionService

class TestHybridService:
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Verify Layer 1 and Layer 2 run in parallel."""
        # Implementation: check that total time < sum of individual times
        pass
    
    @pytest.mark.asyncio
    async def test_slm_conditional_invocation(self, sample_text_legal):
        """SLM only invoked for attorney-client-privilege detector."""
        pass
    
    @pytest.mark.asyncio
    async def test_merge_strategy_smart(self):
        """Smart merge: GLiNER for names, Presidio for SSN."""
        pass
    
    def test_guardrail_routing(self):
        """Verify detectors route to correct layers."""
        service = HybridDetectionService(...)
        
        # Presidio-only detectors
        presidio_guards = [Guardrail(detector_id="credit-card")]
        assert len(service._filter_presidio_guardrails(presidio_guards)) == 1
        assert len(service._filter_gliner_guardrails(presidio_guards)) == 0
        
        # GLiNER-only detectors
        gliner_guards = [Guardrail(detector_id="icd10-code")]
        assert len(service._filter_presidio_guardrails(gliner_guards)) == 0
        assert len(service._filter_gliner_guardrails(gliner_guards)) == 1
        
        # SLM-only detectors
        slm_guards = [Guardrail(detector_id="attorney-client-privilege")]
        assert len(service._filter_slm_guardrails(slm_guards)) == 1
```

---

## Infrastructure Requirements

### GPU Instance Recommendations

| Use Case | Instance | GPU | VRAM | vCPU | RAM | Cost/hr |
|----------|----------|-----|------|------|-----|---------|
| **Production** | g5.2xlarge | A10G | 24GB | 8 | 32GB | ~$1.21 |
| **MVP/Dev** | g5.xlarge | A10G | 24GB | 4 | 16GB | ~$1.00 |
| **Budget Split** | g4dn.xlarge + g5.xlarge | T4 + A10G | 16GB + 24GB | 4 + 4 | 16GB + 16GB | ~$0.55 + $1.00 |

### Memory Budget (g5.xlarge, 24GB VRAM)

| Component | Memory |
|-----------|--------|
| GLiNER (nvidia/gliner-PII, FP16) | ~1.2GB |
| Phi-4-mini (4-bit AWQ) | ~2.5GB |
| CUDA/PyTorch overhead | ~2.0GB |
| Batch inference buffers | ~1.0GB |
| **Total Used** | ~6.7GB |
| **Available Headroom** | ~17GB |

### Docker Configuration

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install PyTorch with CUDA
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install detection dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download models at build time (optional, for faster startup)
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('nvidia/gliner-PII')"

COPY app/ /app/
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  detection-api:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DETECTION_GLINER_MODEL_NAME=nvidia/gliner-PII
      - DETECTION_SLM_MODEL_NAME=microsoft/Phi-4-mini-instruct
      - DETECTION_ENABLE_SLM=true
      - DETECTION_DEVICE=cuda
    ports:
      - "8000:8000"
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Layer 1 (Presidio) | <10ms | Regex + checksums only |
| Layer 2 (GLiNER) | 30-50ms | Batch processing, FP16 |
| Layer 3 (SLM) | 50-100ms | 4-bit quantized, ~5% of requests |
| End-to-End P95 | <100ms | Parallel L1+L2, conditional L3 |
| Throughput | 100+ req/s | Per GPU, with batching |

---

## Monitoring & Observability

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    model_manager = get_model_manager()
    return {
        "status": "healthy",
        "models": model_manager.health_check(),
        "gpu_memory": model_manager.get_gpu_memory_usage(),
    }
```

### Metrics to Track

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

detection_requests = Counter(
    "detection_requests_total",
    "Total detection requests",
    ["layer", "detector_type"]
)

detection_latency = Histogram(
    "detection_latency_seconds",
    "Detection latency by layer",
    ["layer"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

entities_detected = Counter(
    "entities_detected_total",
    "Total entities detected",
    ["entity_type", "source"]
)
```

---

## Migration Checklist

### Phase 1: GLiNER Integration (Layer 2)
- [ ] Add GLiNER dependencies to requirements.txt
- [ ] Create model_manager.py
- [ ] Create gliner_service.py with entity mapping
- [ ] Create entity_merger.py
- [ ] Create hybrid_service.py (L1 + L2 only)
- [ ] Update service_loader.py
- [ ] Update dependencies.py
- [ ] Add model warmup to lifespan
- [ ] Write unit tests
- [ ] Deploy to staging, validate accuracy

### Phase 2: SLM Integration (Layer 3)
- [ ] Add vLLM/llama-cpp dependency
- [ ] Create slm_service.py with prompt templates
- [ ] Update hybrid_service.py for conditional L3
- [ ] Update config.py with SLM settings
- [ ] Write SLM-specific tests
- [ ] Deploy to staging, validate latency
- [ ] A/B test against production

### Phase 3: Production Rollout
- [ ] Configure GPU autoscaling
- [ ] Set up monitoring dashboards
- [ ] Configure alerting for latency/errors
- [ ] Gradual traffic migration (10% → 50% → 100%)
- [ ] Document runbooks for common issues

---

## Quick Reference: Detector Routing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DETECTOR ROUTING MATRIX                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 1 (PRESIDIO) - Checksum/Regex validated                         │
│  ─────────────────────────────────────────────                         │
│  credit-card, ssn, us-ssn, itin, us-passport, us-driver-license,       │
│  us-bank-number, iban, crypto, email, phone, ip-address, url           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 2 (GLINER) - Context-aware NER                                  │
│  ────────────────────────────────────                                  │
│  person, full-name, first-name, last-name, organization, company,      │
│  address, location, date-of-birth, medical-record-number, mrn,         │
│  diagnosis, medication, icd10-code, npi, health-insurance-id,          │
│  passport (non-US), driver-license (non-US), national-id, bank-account │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 3 (SLM) - Semantic understanding                                │
│  ───────────────────────────────────────                               │
│  attorney-client-privilege, trade-secret, confidential-business-info,  │
│  work-product-doctrine, source-code-proprietary, database-schema,      │
│  custom-sensitive-pattern                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: GLiNER Model Comparison

| Model | F1 Score | Parameters | ONNX | Notes |
|-------|----------|------------|------|-------|
| nvidia/gliner-PII | ~70% | 570M | Manual | Your tested model |
| knowledgator/gliner-pii-base-v1.0 | 80.99% | ~300M | Built-in | Higher accuracy, smaller |
| urchade/gliner_multi_pii-v1 | ~68% | 300M | Manual | Multilingual |

**Recommendation**: Start with `nvidia/gliner-PII` (your validated model), then A/B test `knowledgator/gliner-pii-base-v1.0` for potential 11% F1 improvement.

---

## Appendix B: SLM Model Comparison

| Model | Params | MMLU | GSM8K | VRAM (4-bit) | Best For |
|-------|--------|------|-------|--------------|----------|
| Phi-4-mini | 3.8B | 67.3% | 88.6% | ~4GB | Structured reasoning |
| Gemma 3-4B | 4B | ~64% | ~85% | 2.6GB | Cost efficiency |
| Qwen3-4B | 4B | ~65% | ~80% | ~4GB | Dynamic complexity |
| Llama 3.2-3B | 3B | 61.8% | 75.6% | ~4GB | Community support |

**Recommendation**: Phi-4-mini for best structured output and instruction following.

---

## Appendix C: SpaCy Model Selection

| Model | Size | Memory | F1 (NER) | Speed | Use Case |
|-------|------|--------|----------|-------|----------|
| en_core_web_sm | 12MB | ~100MB | 85.5% | 18k WPS | Minimal footprint |
| en_core_web_md | 40MB | ~200MB | 85.5% | 15k WPS | **Recommended** |
| en_core_web_lg | 560MB | ~800MB | 85.5% | 10k WPS | Overkill for Presidio |
| en_core_web_trf | 400MB | 4-8GB | 90.2% | 684 WPS | Too slow for <100ms |

**Decision**: Use `en_core_web_md` for Presidio tokenization. GLiNER handles heavy NER.

---

*Document Version: 2.0*
*Last Updated: Based on implementation discussions*
*Previous Version: GLINER_IMPLEMENTATION_CLAUDE_CODE.md*