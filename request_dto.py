from pydantic import BaseModel
class SyncRequest(BaseModel):
    title: str
    referenceNumber: str
    issuingBody: str
    documentType: str
    documentField: str
    issuanceDate: str
    effectiveDate: str
    isRegulatory: str
    validityStatus: str
    invalidDate: str
    fileUrl: str

class QuestionRequest(BaseModel):
    question: str
    conversation_id: int
    
class UnsyncRequest(BaseModel):
    doc_id: str
    
class RemoveAndSyncRequest(SyncRequest):
    doc_id: str
    
class ResyncRequest(SyncRequest):
    doc_id: str
