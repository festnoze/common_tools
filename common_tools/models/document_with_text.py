from typing import Any, Literal
from typing import Optional

class DocumentWithText():
    id: Optional[str] = None
    metadata: dict = {}
    content_to_return: str
    page_content: str
    type: Literal["Document"] = "Document"

    def __init__(self, content_to_return: str, page_content: str, metadata: dict = {}) -> None:
        """Pass page_content in as positional or named arg."""
        self.page_content: str = page_content
        self.content_to_return: str = content_to_return
        self.metadata: dict = metadata
        self.type: Literal["Document"] = "Document"

    def __str__(self) -> str:
        return f"page_content='{self.page_content}', content_to_return={self.content_to_return}, metadata={self.metadata}"