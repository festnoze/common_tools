import os
import logging
from langchain_core.documents import Document
from common_tools.helpers.file_helper import file
from common_tools.helpers.env_helper import EnvHelper
from common_tools.models.doc_w_summary_chunks_questions import DocWithSummaryChunksAndQuestions
from common_tools.RAG.rag_ingestion_pipeline.summary_and_questions.summary_and_questions_creation import SummaryAndQuestionsChunksCreation

class SummaryAndQuestionsChunksService:
    logger = logging.getLogger(__name__)
    
    async def build_summaries_and_chunks_by_questions_objects_from_docs_async(
             documents: list[Document], llm_and_fallback: list = None, load_existing_summaries_and_questions_from_file: bool = True, file_path: str = './outputs', existing_summaries_and_questions_filename: str = 'summaries_chunks_and_questions_objects') -> list[DocWithSummaryChunksAndQuestions]:
        
        summaries_and_chunks_by_questions_objects: list[DocWithSummaryChunksAndQuestions] = None
        
        docs_with_summary_chunks_and_questions_file_path = os.path.join(file_path, existing_summaries_and_questions_filename)        
        if load_existing_summaries_and_questions_from_file:
            summaries_and_chunks_by_questions_objects = await SummaryAndQuestionsChunksService._load_existing_summaries_and_questions_objects_from_file_async(docs_with_summary_chunks_and_questions_file_path)
            
        if not summaries_and_chunks_by_questions_objects:
            summaries_and_chunks_by_questions_objects = await SummaryAndQuestionsChunksCreation.generate_summaries_and_chunks_by_questions_objects_from_docs_async(
                documents, llm_and_fallback, docs_with_summary_chunks_and_questions_file_path, load_batch_files_if_exists=load_existing_summaries_and_questions_from_file
            )

        # Verify that each loaded/generated summaries and chunks by questions objects correspond to each provided documents
        if documents:
            if len(documents) != len(summaries_and_chunks_by_questions_objects):
                raise ValueError(f"Number of summaries and chunks by questions objects ({len(summaries_and_chunks_by_questions_objects)}) != number of documents ({len(documents)})")
            for summary_and_chunks_by_questions_object, doc in zip(summaries_and_chunks_by_questions_objects, documents):
                if summary_and_chunks_by_questions_object.doc_content != doc.page_content:
                    raise ValueError(f"Content mismatch in: {doc.metadata['name']}")
                for key, value in summary_and_chunks_by_questions_object.metadata.items():
                    if key != 'id' and (key not in doc.metadata or doc.metadata[key] != value):
                        raise ValueError(f"Metadata mismatch in: {doc.metadata['name']} for key: {key}")
        return summaries_and_chunks_by_questions_objects                

    async def _load_existing_summaries_and_questions_objects_from_file_async(docs_with_summary_chunks_and_questions_file_path: str) -> list[DocWithSummaryChunksAndQuestions]:
        if not docs_with_summary_chunks_and_questions_file_path.endswith('.json'):
            docs_with_summary_chunks_and_questions_file_path += '.json'
        if file.exists(docs_with_summary_chunks_and_questions_file_path):
            docs_with_summary_chunks_and_questions_json = file.get_as_json(docs_with_summary_chunks_and_questions_file_path)
            summaries_and_chunks_by_questions_objects = [DocWithSummaryChunksAndQuestions(**doc) for doc in docs_with_summary_chunks_and_questions_json]
            SummaryAndQuestionsChunksService.logger.info(f">>> Loaded existing {len(summaries_and_chunks_by_questions_objects)} docs with: summary, chunks and questions from file: {docs_with_summary_chunks_and_questions_file_path}")
            summaries_and_chunks_by_questions_objects = SummaryAndQuestionsChunksCreation._replace_metadata_id_by_doc_id(summaries_and_chunks_by_questions_objects)
            return summaries_and_chunks_by_questions_objects
        return None
    
    def build_chunks_splitted_by_questions_from_summaries_and_chunks_by_questions_objects(summaries_and_chunks_by_questions_objects: list[DocWithSummaryChunksAndQuestions]) -> list[Document]:
        chunks_and_questions_documents: list[Document] = []
        create_questions_from_data: bool = EnvHelper.get_is_questions_created_from_data()
        merge_questions_with_data: bool = EnvHelper.get_is_mixed_questions_and_data()
        
        for summary_and_chunks_object in summaries_and_chunks_by_questions_objects:
            if not create_questions_from_data:
                chunks_and_questions_documents.append(Document(page_content=summary_and_chunks_object.text, metadata=summary_and_chunks_object.metadata))
            else:
                chunks_and_questions_documents.extend(summary_and_chunks_object.to_langchain_documents_chunked_summary_and_questions(merge_questions_with_data))

        return chunks_and_questions_documents