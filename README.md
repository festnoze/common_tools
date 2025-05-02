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

<u>*Tips:*</u>

- Look into `setup.py` file, the required packages are listed in: `install_requires` section.
- To install locally this package from another project, execute from the terminal: `pip install -e <CommonToolsPath>` where 'CommonToolsPath' has to be replaced by the actual path of your local "common tools" project root folder.
- To build the package 'common_tools', simply run the command: "**libs_build.bat**", from within the "common tools" root folder. The built package will be found in the "dist" folder with an auto-incremented version. Copy the '*.whl' file where you want this library to be referenced.