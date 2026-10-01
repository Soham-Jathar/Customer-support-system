"""Safe text extraction for customer-supplied complaint attachments.

Files are processed in memory and never persisted. The result is capped before
it enters the normal NLP pipeline, so an attachment cannot bypass message-size
or storage protections.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

from fastapi import HTTPException, UploadFile


MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 5000
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@dataclass(frozen=True)
class ExtractedAttachment:
    filename: str
    extension: str
    text: str


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:MAX_EXTRACTED_CHARACTERS]


async def extract_attachment_text(upload: UploadFile) -> ExtractedAttachment:
    filename = Path(upload.filename or "attachment").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Attach a TXT, PDF, or DOCX file.")

    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Attachment must be 2 MB or smaller.")

    try:
        if extension == ".txt":
            extracted = content.decode("utf-8", errors="replace")
        elif extension == ".pdf":
            from pypdf import PdfReader
            extracted = " ".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
        else:
            from docx import Document
            extracted = " ".join(paragraph.text for paragraph in Document(BytesIO(content)).paragraphs)
    except Exception as error:
        raise HTTPException(status_code=422, detail="The attachment could not be read as text.") from error

    text = _normalise(extracted)
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="No readable complaint text was found in the attachment.")
    return ExtractedAttachment(filename=filename, extension=extension.removeprefix("."), text=text)
