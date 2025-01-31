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
    key: str
    fileUrl: str

class QuestionRequest(BaseModel):
    question: str
    conversation_id: int
    
class UnsyncRequest(BaseModel):
    doc_id: str
    key: str    

class RemoveAndSyncRequest(SyncRequest):
    doc_id: str
    old_key: str

class ResyncRequest(SyncRequest):
    doc_id: str