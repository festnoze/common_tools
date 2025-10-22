<H1>______ Common tools ______</H1>

<H3>Presentation</H3>
**Common tools** contains a bunch of helpers methods of common usage, which are embeded into this package to keep the code factorized and ease their reusability.

<H3>Includes</H3>
The provided tools range from:

- Low-level helpers, like for: string, json or file management.

- To higher level AI tools (which will be in a separate library in the future), including:
  
  - **LLM querying** tools (including: output parsing, fallbacks, paralellization and batching) powered by Langchain,
  
  - **RAG** toolbox, including:
    
    - A complete and modulable **injection pipeline** with: metadata handling, chunking, embedding, vector database creation, and querying.
    
    - A complete and modulable **inference pipeline** with: pre-treatment (query translation, multi-querying,metadata extraction & pre-filtering), hybrid search (BM25 & vector search), and post-treatment. This tool is itself based on:
    
    - A generic **Workflow Executor** capable of interpreting a scripted workflow defined into a YAML file which specify the workflow steps and structure, as well as parallele steps execution (as separate thread or async methods).
  
  - **Agents & tools**.
    
    - To be completed.

<H3>Installation</H3>

The package supports flexible installation profiles to minimize dependencies based on your needs.

**Basic installation (core features only - includes ChromaDB as default vector database):**
```bash
pip install -e <CommonToolsPath>
```

**Installation with optional features:**
```bash
# Install with Pinecone vector database support (requires C++ redistributable for pinecone-text)
pip install -e <CommonToolsPath>[pinecone]

# Install with Qdrant vector database support
pip install -e <CommonToolsPath>[qdrant]

# Install with database support (SQLite + PostgreSQL)
pip install -e <CommonToolsPath>[database]

# Install with ML/scientific computing dependencies (scikit-learn, scipy, pandas)
pip install -e <CommonToolsPath>[ml]

# Install with advanced features (langgraph, langsmith, ragas)
pip install -e <CommonToolsPath>[advanced]

# Install with multiple optional features
pip install -e <CommonToolsPath>[pinecone,qdrant,database,ml,advanced]

# Install everything (all optional dependencies)
pip install -e <CommonToolsPath>[full]
```

**Environment variable-based installation (for CI/CD):**
```bash
# Windows
set COMMON_TOOLS_INSTALL_MODE=full
pip install -e <CommonToolsPath>

# Linux/Mac
COMMON_TOOLS_INSTALL_MODE=full pip install -e <CommonToolsPath>

# Install specific profiles (comma-separated)
set COMMON_TOOLS_INSTALL_MODE=pinecone,database,ml
pip install -e <CommonToolsPath>
```

**Available installation profiles:**
- `pinecone` - Pinecone vector database (requires C++ redistributable)
- `qdrant` - Qdrant vector database
- `database` - Database support (SQLite + PostgreSQL)
- `ml` - ML/scientific computing (scikit-learn, scipy, pandas)
- `advanced` - Advanced AI features (langgraph, langsmith, ragas)
- `vectordb` - Both Pinecone and Qdrant
- `full` - All optional dependencies

<u>*Tips:*</u>

- Look into `setup.py` file, the dependencies are organized into core and optional extras.
- Replace `<CommonToolsPath>` with the actual path of your local "common tools" project root folder.
- To build the package 'common_tools', simply run the command: "**libs_build.bat**", from within the "common tools" root folder. The built package will be found in the "dist" folder with an auto-incremented version. Copy the '*.whl' file where you want this library to be referenced.