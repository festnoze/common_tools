# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**common_tools** is a Python package providing reusable utilities for AI applications and general development needs. It ranges from low-level helpers (file, JSON, string management) to sophisticated AI tools including LLM querying, RAG pipelines, and workflow execution systems powered by LangChain.

This is a library package meant to be installed as a dependency in other projects.

## Python Environment

- **Python Version**: `>=3.11.10`
- **Virtual Environment**: The project uses a virtual environment located at `senv/`
- **IMPORTANT**: Always activate the virtual environment before running tests, scripts, or any commands from apps within this package:
  ```bash
  # Windows
  senv\Scripts\activate

  # Linux/Mac
  source senv/bin/activate
  ```

## Installation & Building

### Installing the Package

The package supports flexible installation profiles to minimize dependencies:

```bash
# Basic installation (includes ChromaDB as default vector database)
pip install -e .

# Installation with optional features
pip install -e .[pinecone]      # Pinecone vector DB (requires C++ redistributable)
pip install -e .[qdrant]        # Qdrant vector DB
pip install -e .[database]      # SQLite + PostgreSQL support
pip install -e .[ml]            # scikit-learn, scipy, pandas
pip install -e .[advanced]      # langgraph, langsmith
pip install -e .[ragas]         # ragas
pip install -e .[full]          # All optional dependencies

# Multiple features
pip install -e .[pinecone,database,ml,advanced]

# Environment variable-based installation (for CI/CD)
# Windows:
set COMMON_TOOLS_INSTALL_MODE=full
pip install -e .

# Linux/Mac:
COMMON_TOOLS_INSTALL_MODE=full pip install -e .
```

### Building the Package

To build a distributable wheel package:

```bash
# Run the build script (auto-increments version)
libs_build.bat

# The built package will be in the dist/ folder as common_tools-<version>.whl
```

The build process:
- Automatically calculates the next version based on existing builds in `dist/`
- Uses `python -m build` to create the wheel
- Reads version from `BUILD_VERSION` environment variable (set by `get_next_version_of_lib.bat`)

## Running Tests

Tests are located in the `tests/` directory and use pytest:

```bash
# Activate venv first!
senv\Scripts\activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/workflow_executor_test.py

# Run with verbose output
pytest tests/ -v

# Run specific test method
pytest tests/workflow_executor_test.py::TestWorkflowExecutor::test_execute_workflow_single_step
```

## Code Architecture

### Async Method Naming Convention

**CRITICAL**: Async methods MUST be prefixed with "a" (e.g., `async def acreate_user(...)`, `async def aget_user_by_id(...)`). DO NOT use "_async" suffix.

### Key Components

#### 1. Workflow Executor (`common_tools/workflows/`)

The WorkflowExecutor is a generic engine that interprets YAML-based workflow definitions to execute complex pipelines with:
- Sequential step execution
- Parallel async step execution (`parallel_async` keyword)
- Sub-workflow nesting
- Dynamic argument passing between steps
- Retry mechanisms on exceptions

**Workflow Configuration**: Steps are defined in YAML files (see `common_tools/RAG/configs/*.yaml` for examples). Each step references a static method in format `ClassName.method_name`.

**Key Features**:
- Steps can be strings (function references), lists (sub-workflows), or dicts (parallel execution)
- The `@workflow_output('name1', 'name2')` decorator allows functions to inject named outputs into the workflow's kwargs
- Automatic argument resolution: arguments are filled from `kwargs_values` first, then from `previous_results`
- Type checking validates that values from previous results match expected parameter types

**Usage Pattern**:
```python
from common_tools.workflows.workflow_executor import WorkflowExecutor

executor = WorkflowExecutor(
    config_or_config_file_path='path/to/workflow.yaml',
    available_classes={'ClassName': ClassName}
)
results = await executor.execute_workflow_async(kwargs_values={'param': value})
```

#### 2. RAG System (`common_tools/RAG/`)

A complete Retrieval-Augmented Generation system with:

**Ingestion Pipeline** (`rag_ingestion_pipeline/`):
- Document chunking and metadata handling
- Embedding generation
- Vector database creation (Chroma/Pinecone/Qdrant)

**Inference Pipeline** (`rag_inference_pipeline/`):
- Pre-treatment: query translation, metadata extraction, multi-querying
- Hybrid search: BM25 + vector search
- Post-treatment: response verification and formatting
- Configurable via YAML workflow files

**RagService** (`rag_service.py`):
- Central service managing LLMs, embeddings, and vector stores
- Supports multiple vector databases (ChromaDB, Qdrant, Pinecone)
- Factory pattern: `RagServiceFactory.build_from_env_config()`
- Handles both cloud-hosted and local vector databases

**Pipeline Exceptions**: Custom exceptions like `EndPipelineException`, `GreetingsEndsPipelineException` allow early termination of workflows.

#### 3. Database Layer (`common_tools/database/`)

**GenericDataContext** provides unified async database access for both SQLite and PostgreSQL:

**Key Methods**:
- `aget_entity_by_id_async()`, `aget_all_entities_async()`: Read operations
- `aadd_entity_async()`, `aupdate_entity_async()`: Write operations
- `new_transaction_async()`: Context manager for transactions
- `create_database_async()`: Async table creation (important for SQLite async operations)

**Database Type Detection**: Auto-detects from connection string or explicit `DatabaseType.SQLITE`/`DatabaseType.POSTGRESQL`

**Important**: For async SQLite testing, use `create_database_async()` to ensure tables are visible to async sessions immediately.

#### 4. LangChain Helpers (`common_tools/langchains/`)

- **LangChainFactory**: Creates LLM instances from `LlmInfo` configurations
- **LangSmithClient**: Integration with LangSmith for tracing and monitoring
- Supports multiple providers: OpenAI, Anthropic, Groq, Ollama

#### 5. Helper Modules (`common_tools/helpers/`)

Low-level utilities organized by function:
- `file_helper`: File I/O, YAML/JSON reading
- `json_helper`: JSON parsing and manipulation
- `llm_helper`: LLM querying with fallbacks, batching, parallel execution
- `batch_helper`: Batch processing utilities
- `reflexion_helper`: Reflection/introspection utilities for dynamic method execution
- `test_helper`: Testing utilities
- `env_helper`: Environment variable management
- `txt_helper`: Text/console output utilities
- `duration_helper`: Timing and duration formatting
- `matching_helper`: String matching and fuzzy search
- `rag_filtering_metadata_helper`: RAG metadata filtering logic

#### 6. Models (`common_tools/models/`)

Data models and enums:
- `LlmInfo`: LLM configuration (model, temperature, API keys)
- `EmbeddingModel`, `EmbeddingModelFactory`: Embedding model management
- `VectorDbType`: Enum for vector database types (ChromaDB, Pinecone, Qdrant)
- `Message`, `User`: Data structures for conversations
- `LanggraphAgentState`: State management for LangGraph agents
- `QuestionAnalysisBase`, `QuestionRewritting`: RAG query processing models

## Project Structure

```
common_tools/
├── database/           # Generic async data context (SQLite/PostgreSQL)
├── helpers/            # Low-level utilities (file, JSON, LLM, etc.)
├── langchains/         # LangChain factory and adapters
├── models/             # Data models and enums
├── prompts/            # LLM prompts (packaged as data files)
├── RAG/
│   ├── configs/        # YAML workflow configurations
│   ├── rag_ingestion_pipeline/
│   ├── rag_inference_pipeline/
│   └── rag_service.py  # Main RAG service
├── workflows/          # Workflow executor engine
│   ├── workflow_executor.py
│   ├── workflow_output_decorator.py
│   └── end_workflow_exception.py
└── project/            # Project-level utilities

tests/                  # Pytest tests
├── database/
├── workflow_executor_test.py
├── rag_inference_pipeline_test.py
└── ...
```

## Important Notes

### Package Data Files

The package includes non-Python resource files:
- `common_tools.prompts`: LLM prompt templates
- `common_tools.RAG.configs`: YAML workflow configurations

These are specified in `setup.py` under `package_data` to be included in the built package.

### Environment Variables

Many features are configured via environment variables (see `env_helper.py`):
- LLM API keys (OpenAI, Anthropic, Groq, etc.)
- Vector database configuration
- RAG pipeline settings (BM25 storage, hybrid search, summarization)
- Model selection (embedding models, LLM models)

### Git & Version Control

- Main branch: `master` (for PRs)
- Development branch: `develop`
- Build artifacts in `dist/` are version-controlled for distribution tracking
- The `build/` directory contains compiled package artifacts (typically gitignored)

### Testing Database Operations

When testing async database operations with SQLite:
1. Always use `create_database_async()` instead of `create_database()` for async contexts
2. This ensures tables are visible to async sessions immediately
3. PostgreSQL users should ensure the database exists before running tests
