import pytest
from unittest.mock import patch, MagicMock
from langchain_chroma import Chroma
from langchain_core.documents import Document
from common_tools.RAG.rag_inference_pipeline.rag_inference_pipeline import RagInferencePipeline
from common_tools.RAG.rag_service import RagService
from common_tools.models.langchain_adapter_type import LangChainAdapterType
from common_tools.models.llm_info import LlmInfo
from common_tools.models.embedding_model import EmbeddingModel
from common_tools.models.user import User
from common_tools.models.conversation import Conversation
from common_tools.models.message import Message
from common_tools.models.device_info import DeviceInfo
from common_tools.models.vector_db_type import VectorDbType
from common_tools.helpers.env_helper import EnvHelper

class TestRagInferencePipelineIntegration:

    def setup_method(self):
        # Set up the necessary LLM information for the RAGService
        llms_infos = []
        #llms_infos.append(LlmInfo(type= LangChainAdapterType.Ollama, model= "phi3", timeout=80, temperature=0))
        llms_infos.append(LlmInfo(type=LangChainAdapterType.Ollama, model="llama3.2:1b", timeout=80, temperature=0))
        
        docs: list[Document] = [
            Document(page_content="Choupicity is the capital of Choupiland.", metadata={"source": "Wikipedia", "id": "doc1"}),
            Document(page_content="The Eiffel Tower is a famous landmark in Paris.", metadata={"source": "Wikipedia", "id": "doc2"}),
            Document(page_content="The Louvre is a famous museum in Paris.", metadata={"source": "Wikipedia", "id": "doc3"}),
            Document(page_content="CCIAPF is the simulation of octopus intelligence in trees.", metadata={"source": "Wikipedia", "id": "doc4"}),
        ]

        # Patch DB related env. values from EnvHelper
        self.env_helper_patchers = [
            patch.object(EnvHelper, 'get_BM25_storage_as_db_sparse_vectors', return_value=False),
            patch.object(EnvHelper, 'get_is_common_db_for_sparse_and_dense_vectors', return_value=False),
            patch.object(EnvHelper, 'get_is_summarized_data', return_value=False),
            patch.object(EnvHelper, 'get_is_questions_created_from_data', return_value=False),
        ]
        self.env_helper_mocks = [patcher.start() for patcher in self.env_helper_patchers]

        with patch.object(RagService, '__init__', return_value=None):
            self.rag_service = RagService()
            self.rag_service.instanciate_embedding(EmbeddingModel.Ollama_AllMiniLM)
            self.rag_service.instanciate_llms(llms_infos, test_llms_inference=False)
            self.rag_service.langchain_documents = docs
            self.rag_service.vectorstore = Chroma.from_documents(documents= docs, embedding = self.rag_service.embedding)
            self.rag_service.vector_db_type = VectorDbType.ChromaDB
            #
            self.inference = RagInferencePipeline(self.rag_service)
    
    def teardown_method(self):
        # Stop the patchers when the test is done
        for patcher in self.env_helper_patchers:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_inference_pipeline_run_dynamic_with_bm25_retrieval(self):
        query = "Quelle est la capitale de la Choupiland ?"
        user = User('John Doe', device_info=DeviceInfo("0.1.2.3", "test_agent", "test", "v1", "win10", "", False))
        conv = Conversation(user, [Message(role="user", content=query)])
        
        response = await self.inference.run_pipeline_dynamic_no_streaming_async(
            conv, 
            include_bm25_retrieval=True, 
            give_score=True, 
            format_retrieved_docs_function=None
        )

        assert isinstance(response, str), "The response should be a string"
        # assert isinstance(sources, list), "The sources should be a list"
        # assert len(sources) > 0, "There should be at least one source retrieved"
        assert "Choupicity" in response, f"The response should mention the fake capital of Choupiland from the data: 'Choupicity', but was: '{response}'"

    @pytest.mark.asyncio
    async def test_inference_pipeline_run_dynamic_without_bm25_retrieval(self):
        query = "Explain the concept of CCIAPF."
        user = User('John Doe', device_info=DeviceInfo("1.2.3.4", "test_agent", "test", "v1", "win10", "", False))
        conv = Conversation(user, [Message(role="user", content=query)])        

        response = await self.inference.run_pipeline_dynamic_no_streaming_async(
            conv,
            include_bm25_retrieval=False, 
            give_score=True, 
            format_retrieved_docs_function=TestRagInferencePipelineIntegration.format_retrieved_docs_function
        )

        # Assertions to verify that the response and sources are valid
        assert isinstance(response, str), "The response should be a string"
        # assert isinstance(sources, list), "The sources should be a list"
        # assert len(sources) > 0, "There should be at least one source retrieved"
        #assert [ "I found! " source for source in sources], f"The response should mention 'I found! ' added by the formatting function, but was: '{response}'"
        assert response.lower().__contains__("octopus") or response.lower().__contains__("pieuvre") or response.lower().__contains__("poulpe"), f"The response should mention 'octopus', but was: '{response}'"

    @pytest.mark.asyncio
    async def test_inference_pipeline_run_static_with_bm25_retrieval(self):
        query = "Quelle est la capitale de la Choupiland ?"
        user = User('John Doe', device_info=DeviceInfo("1.2.3.4", "test_agent", "test", "v1", "win10", "", False))
        conv = Conversation(user, [Message(role="user", content=query)])  

        response = await self.inference.run_pipeline_static_no_streaming_async(
            conv, 
            include_bm25_retrieval=True, 
            give_score=True, 
            format_retrieved_docs_function=None
        )

        assert isinstance(response, str), "The response should be a string"
        # assert isinstance(sources, list), "The sources should be a list"
        # assert len(sources) > 0, "There should be at least one source retrieved"
        assert "Choupicity" in response, f"The response should mention the fake capital of Choupiland from the data: 'Choupicity', but was: '{response}'"

    @staticmethod
    def format_retrieved_docs_function(retrieved_docs:list):
        if not any(retrieved_docs):
            return 'not a single information were found. Don\'t answer the question.'
        add_txt = "I found! "
        for doc in retrieved_docs:
            if isinstance(doc, tuple):
                doc[0].page_content = add_txt + doc[0].page_content
            elif isinstance(doc, Document):
                doc.page_content = add_txt + doc.page_content
            elif isinstance(doc, str):
                doc = add_txt + doc
            else:
                raise ValueError("Invalid document type")
        
        return retrieved_docs

    # def test_inference_pipeline_custom_format_function(self):
    #     # Define the query for the test
    #     query = "What is the importance of quantum computing?"

    #     # Define a custom formatting function
    #     def custom_format_function(docs):
    #         return f"Custom Format: {docs}"

    #     # Run the inference pipeline with a custom formatting function
    #     response, sources = self.inference.run_pipeline_dynamic(
    #         query, 
    #         include_bm25_retrieval=True, 
    #         give_score=False, 
    #         format_retrieved_docs_function=custom_format_function
    #     )

    #     # Assertions to verify that the response and sources are valid and custom formatted
    #     assert isinstance(response, str), "The response should be a string"
    #     assert isinstance(sources, list), "The sources should be a list"
    #     assert len(sources) > 0, "There should be at least one source retrieved"
    #     assert [ "I found! " source for source in sources], f"The response should mention 'I found! ' added by the formatting function, but was: '{response}'"
    #     assert "Custom Format" in response, "The response should contain the custom formatted output"
