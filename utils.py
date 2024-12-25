from llama_parse import LlamaParse
from llama_index.core import Document
from request_dto import SyncRequest
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core import ChatPromptTemplate


async def getDocumentFromDigitalOcean(parser: LlamaParse, fileUrl: str):
    documents = await parser.aload_data(file_path=fileUrl)
    return documents[0]


def get_prompt():
   return (
       "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the question: {query_str}\n"
        "Follow instructions below.\n"
        "1. Respond in Vietnamese.\n"
        "2. If the context isn't helpful, inform that no results are available.\n"
        "3. Format the response using markdown for better readability.\n"
        "4. Include HTML links at the end of response to the source documents based on metadata {file_url} "
        "with a title derived from the relevant context and an introducing label for the links.\n"
   )


def set_metadata_from_request(request: SyncRequest, document: Document):
    document.metadata["document_title"] = request.title
    document.metadata["document_reference_number"] = request.referenceNumber
    document.metadata["document_issuing_body"] = request.issuingBody
    document.metadata["document_type"] = request.documentType
    document.metadata["document_field"] = request.documentField
    document.metadata["issuance_date"] = request.issuanceDate
    document.metadata["date_of_effect"] = request.effectiveDate
    document.metadata["regulatory"] = request.isRegulatory
    document.metadata["validity_status"] = request.validityStatus
    document.metadata["invalid_date"] = request.invalidDate
    document.metadata["file_url"] = request.fileUrl


def get_vector_store_info():
    return VectorStoreInfo(
        content_info="Documents of Vietnamese University",
        metadata_info=[
            MetadataInfo(
                name="document_title",
                type="str",
                description=(
                    "The name or title of the document manually entered by administrator or manager "
                    "that briefly decribe its content."
                ),
            ),
            MetadataInfo(
                name="document_reference_number",
                type="str",
                description=(
                    "A unique identifier or code assigned to the document for tracking and reference."
                ),
            ),
            MetadataInfo(
                name="document_issuing_body",
                type="str",
                description=(
                    "The organization or authority responsible for creating and issuing the document."
                ),
            ),
            MetadataInfo(
                name="document_type",
                type="str",
                description=(
                    "The category of the document based on its purpose or nature, such as: "
                    "directive, decision, notification, plan, template, etc."
                ),
            ),
            MetadataInfo(
                name="document_field",
                type="str",
                description=(
                    "The area or domain the document applies to, such as: education, student affairs, "
                    "quality assurance, science and technology, etc."
                ),
            ),
            MetadataInfo(
                name="issuance_date",
                type="str",
                description=(
                    "The date when the document was officially published or issued [by its issuing body], "
                    "represented as a date string, for example: 'Sat Dec 20 2024'."
                ),
            ),
            MetadataInfo(
                name="date_of_effect",
                type="str",
                description=(
                    "The date when the document's provisions or regulations become applicable or enforceable, "
                    "represented as a date string, for example: 'Sat Dec 20 2024'."
                ),
            ),
            MetadataInfo(
                name="regulatory",
                type="str",
                description=(
                    "Indicates whether the document has legal or regulatory implications, "
                    "one of: ['regulatory document', 'non-regulatory document']"
                ),
            ),
            MetadataInfo(
                name="validity_status",
                type="str",
                description=(
                    "The current status of the document's applicability, one of ['valid', 'expired']."
                ),
            ),
            MetadataInfo(
                name="invalid_date",
                type="str",
                description=(
                    "The date when the document loses its validity or is no longer effective, "
                    "represented as a date string, for example: 'Sat Dec 20 2024'."
                ),
            ),
            MetadataInfo(
                name="file_url",
                type="str",
                description=(
                    "The link to access or download the digital version of the document."
                ),
            ),
        ],
    )
