from setuptools import setup, find_packages
import os

######################################################################
## To install/update the below listed dependencies globally,
#  run the following command as admin in a python terminal:
# senv\Scripts\activate
# pip install -e . --upgrade
#
## To create the package wheel, run the following command:
# python -m build .
# or: 
# python setup.py sdist bdist_wheel
######################################################################

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Core dependencies - always installed
core_requires = [
    'typer==0.10.0',  # instructor (v0.5.2) requires this upper version
    'python-dotenv',
    'requests',
    'pydantic',
    'rank-bm25',
    'langchain>=0.3.3',
    'langchain-core>=0.3.12',
    'langchain-community>=0.3.2',
    'langchain-experimental>=0.3.2',
    'langchain-openai>=0.2.2',
    'langchain-groq>=0.2.0',
    'protobuf==4.25.1',
    'langchain-chroma>=0.1.4',  # Chroma as default/lightweight vector DB
    'langchain-ollama>=0.1.0',
    'langchain-anthropic>=0.2.4',
    'openai>=1.52.0',
    'ollama',
    'groq',
    'pyyaml',
    'lark',  # needed for langchain 'self-querying' retriever
    'fuzzywuzzy',
    'python-Levenshtein',
]

# Optional dependency groups
extras_require = {
    # Pinecone vector database (requires C++ redistributable for pinecone-text)
    'pinecone': [
        'pinecone>=7.3.0',
        'pinecone-text>=0.9.0',
        'langchain-pinecone>=0.1.0',
    ],
    # Qdrant vector database
    'qdrant': [
        'qdrant-client',
        'langchain-qdrant>=0.1.0',
    ],
    # Database support (SQLite + PostgreSQL)
    'database': [
        'sqlalchemy>=2.0.0',  # For all database SGBD
        'aiosqlite>=0.19.0',  # For SQLite async
        'asyncpg>=0.29.0',    # For PostgreSQL async
        'psycopg2-binary>=2.9.0',  # For PostgreSQL sync (has binary deps)
    ],
    # ML/Scientific computing (large dependencies)
    'ml': [
        'scikit-learn',
        'scipy',
        'pandas',
    ],
    # Advanced features
    'advanced': [
        'langgraph>=0.2.38',
        'langsmith>=0.1.136',
        'ragas>=0.2.5',
    ],
}

# Convenience groups
extras_require['vectordb'] = extras_require['pinecone'] + extras_require['qdrant']
extras_require['full'] = (
    core_requires +
    extras_require['pinecone'] +
    extras_require['qdrant'] +
    extras_require['database'] +
    extras_require['ml'] +
    extras_require['advanced']
)

# Support environment variable for backward compatibility and CI/CD
# Usage: set COMMON_TOOLS_INSTALL_MODE=full (or pinecone,qdrant,database,ml,advanced)
install_mode = os.environ.get('COMMON_TOOLS_INSTALL_MODE', '').lower()
install_requires = core_requires.copy()

if install_mode:
    if install_mode == 'full':
        install_requires = extras_require['full']
    else:
        # Support comma-separated list: e.g., "pinecone,database,ml"
        for mode in install_mode.split(','):
            mode = mode.strip()
            if mode in extras_require:
                install_requires.extend(extras_require[mode])

setup(
    name='common_tools',
    # HOW TO USE: in cmd, do: 'set BUILD_VERSION=0.4.14' to override the version to be built.
    version=os.environ.get('BUILD_VERSION', '1.0.0'),
    description='Common tools for AI and generic needs on file, console, json, ...',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Etienne Millerioux',
    author_email='etienne.emillerioux@strudi.fr',
    url='https://studi-ai@dev.azure.com/studi-ai/Skillforge/_git/ai-commun-tools',
    packages=find_packages(where="."),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.10',
    include_package_data=True,  # Permet d'inclure les fichiers spécifiés dans MANIFEST.in
    package_data={        
        # If any folder contains ressources files (which are not python files, like *.txt files), reference those folders here to include them into the 'common_tools' package:
        'common_tools.prompts': ['**/*'],
        'common_tools.RAG.configs': ['**/*']
    },
    package_dir={"": "."},  # Ensure root directory is taken as package directory
    zip_safe=False
)