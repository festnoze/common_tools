import logging
from typing import Union, TYPE_CHECKING
from pydantic import BaseModel, Field
from langchain_core.documents import Document

if TYPE_CHECKING:
    import pandas as pd

def _import_pandas():
    """Lazy import for pandas."""
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise ImportError(
            "Pandas not found. Install with: pip install common_tools[ml]"
        ) from e

#
from common_tools.helpers.txt_helper import txt
from common_tools.models.document_with_text import DocumentWithText

class Question:
    def __init__(self, text: str = ''):
        self.text = text

    def __repr__(self) -> str:
        return f"Question: {self.text})"
    
    def to_dict(self) -> dict:
        return {"text": self.text}

class DocChunk:
    def __init__(self, text: str = '', questions:list[Question] = []):
        self.text = text
        self.questions = questions
    
    def __repr__(self) -> str:
        return f"DocChunk: {self.text})"
    
    def to_dict(self) -> dict:
        return {"text": self.text, "questions": [question.to_dict() for question in self.questions]}
    
class DocWithSummaryChunksAndQuestions:
    def __init__(self, doc_content: str = None, doc_summary: str = None, doc_chunks_with_questions: Union[dict, list[DocChunk]] = None, metadata: dict = None, **kwargs):
        self.doc_content: str = doc_content if doc_content else kwargs.get('doc_content')
        self.doc_summary: str = doc_summary if doc_summary else kwargs.get('doc_summary')
        self.doc_chunks: list[DocChunk] = self.get_typed_chunks_with_their_questions(
                    doc_chunks_with_questions if doc_chunks_with_questions else kwargs.get('doc_chunks')
        )
        self.metadata: dict = metadata if metadata else kwargs.get('metadata', {})
        self.logger = logging.getLogger(__name__)

    def __repr__(self) -> str:
        summary_words_count = len(self.doc_summary.replace('\n', ' ').split(' '))
        chunk_max_words_count = max([len(chunk.text.split(' ')) for chunk in self.doc_chunks])
        max_chunk_sum_questions_words_count = max([sum([len(question.text.split(' ')) for question in chunk.questions]) for chunk in self.doc_chunks])
        return f"Summary: {summary_words_count} words, {len(self.doc_chunks)} chunks (max. {chunk_max_words_count} words), and {sum([len(chunk.questions) for chunk in self.doc_chunks])} questions (max. words in all questions in chunk: {max_chunk_sum_questions_words_count})."
    
    def to_dict(self, include_full_doc=True, include_metadata=True) -> dict:
        doc_dict = {
            'doc_content': self.doc_content if include_full_doc else None,
            'doc_summary': self.doc_summary,
            'doc_chunks': [chunk.to_dict() for chunk in self.doc_chunks],
            'metadata': self.metadata if include_metadata else None
        }
        return doc_dict
    
    def to_langchain_document(self, include_full_training_text=False, include_summary=True) -> Document:
        page_content = ''
        if include_summary:
            page_content += self.doc_summary + '\n'
        if include_full_training_text:
            page_content += self.doc_content + '\n'
        return Document(page_content=page_content, metadata=self.metadata)
    
    questions_title = '### Questions ###\n'
    answers_title = '### Réponses ###\n'
    
    def to_langchain_documents_chunked_summary_and_questions(self, mixed_questions_and_summary: bool = False) -> list[Document]:
        docs = []
        for chunk in self.doc_chunks:
            chunk_content = ''
            if mixed_questions_and_summary:
                chunk_content += DocWithSummaryChunksAndQuestions.questions_title
                for question in chunk.questions:
                    chunk_content += question.text + '\n'
                chunk_content += DocWithSummaryChunksAndQuestions.answers_title
                chunk_content += chunk.text + '\n'
                docs.append(Document(page_content=chunk_content, metadata=self.metadata))
            else:
                for question in chunk.questions:
                    question_chunk = DocumentWithText(content_to_return=chunk.text, page_content=question.text, metadata=self.metadata)
                    docs.append(question_chunk)
        return docs
    
    def get_typed_chunks_with_their_questions(self, chunks_and_questions_dict: Union[dict, list[DocChunk]]) -> list[DocChunk]:
        chunks_with_questions_typed = [] 
        if isinstance(chunks_and_questions_dict, list) and any(chunks_and_questions_dict) and isinstance(chunks_and_questions_dict[0], DocChunk):
            return chunks_and_questions_dict
        for chunk_with_questions in chunks_and_questions_dict:
            questions = chunk_with_questions['questions']
            question_list = []
            if any(questions) and isinstance(questions[0], Question):
                question_list = questions
            else:
                for question in questions:
                    if isinstance(question, str):
                        question_list.append(Question(question))
                    elif isinstance(question, dict) and 'text' in question:
                        question_list.append(Question(question['text']))
                    else:
                        raise ValueError(f'Invalid question format: {question}')
                
            doc_chunk = DocChunk(chunk_with_questions['text'], question_list)
            chunks_with_questions_typed.append(doc_chunk)      
        return chunks_with_questions_typed
    
    def display_to_terminal_pandas(self):
        pd = _import_pandas()
        data = []
        max_questions = 0
        data.append({'Summary: ': self.doc_summary})
        for i, chunk in enumerate(self.doc_chunks, start=1):
            row = [chunk.text] + [q.text for q in chunk.questions]
            data.append(row)
            max_questions = max(max_questions, len(chunk.questions))
        columns = ['Chunk Text'] + [f'Question n°{i+1}' for i in range(max_questions)]
        df = pd.DataFrame(data, columns=columns)
        df = df.fillna('')
        self.logger.info(df.to_string(index=False))

    def display_to_terminal(self, display_questions: bool = True):
        i = 1
        total_questions = 0
        max_questions = 0
        total_chunks = len(self.doc_chunks)
        for chunk in self.doc_chunks:
            self.logger.info(f'\n>>> Chunk n°{i}:\n' + chunk.text)
            if len(chunk.questions) > max_questions: 
                max_questions = len(chunk.questions)
            i += 1
            j = 1
            for question in chunk.questions:
                if display_questions:
                    self.logger.info(f'>> Question n°{str(j)}: {question.text}')
                total_questions += 1
                j += 1
        self.logger.info(f'Total: {total_chunks} chunks')
        self.logger.info(f'Total: {total_questions} questions')
        self.logger.info(f'Max.: {max_questions} questions')
        self.logger.info(f'Average: {total_questions/total_chunks:.1f} questions by chunk')
        self.logger.info(f'All chunks size: {sum([len(chunk.text.split(' ')) for chunk in self.doc_chunks])} words.')


###################
# Pydantic models #
###################

class QuestionPydantic(BaseModel):
    text: str = Field(description="Une question atomique et complète portant sur une partie du chunk du document.")

class DocChunkPydantic(BaseModel):
    text: str = Field(description="Texte d'une partie (chunk) du document.")
    questions: list[QuestionPydantic] = Field(description="Liste des questions correspondant à ce chunk.")
    
class DocWithSummaryChunksAndQuestionsPydantic(BaseModel):
    doc_summary: str = Field(description="Résumé complet et structuré du contenu du document")
    doc_chunks: list[DocChunkPydantic] = Field(description="Liste des chunks du document.")

class DocQuestionsByChunkPydantic(BaseModel):
    doc_summary: str = Field(description="Résumé complet et structuré du contenu du document")
    doc_chunks: list[DocChunkPydantic] = Field(description="Liste des chunks du document.")

