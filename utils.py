from llama_parse import LlamaParse
from llama_index.core import Document
from request_dto import SyncRequest
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core import ChatPromptTemplate


async def get_doc_from_digital_ocean(parser: LlamaParse, fileUrl: str):
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
            "2. Format the response using markdown for better readability.\n"
            "3. If the context isn't helpful, inform that no results are available.\n"
            "4. Include HTML <a></a> tag format link at the end of response to the first source document based on metadata {file_url} "
            "with a title derived from the relevant context and an introducing label for the link.\n"
    )


def set_metadata_from_request(request: SyncRequest, document: Document):
    document.metadata["title"] = request.title
    document.metadata["reference number"] = request.referenceNumber
    document.metadata["issuing body"] = request.issuingBody
    document.metadata["document type"] = request.documentType
    document.metadata["document field"] = request.documentField
    document.metadata["issuance date"] = request.issuanceDate
    document.metadata["date of effect"] = request.effectiveDate
    document.metadata["regulatory"] = request.isRegulatory
    document.metadata["validity status"] = request.validityStatus
    document.metadata["invalid date"] = request.invalidDate
    document.metadata["file_url"] = request.fileUrl


def get_vector_store_info():
    return VectorStoreInfo(
        content_info="Documents of Vietnamese University",
        metadata_info=[
            MetadataInfo(
                name="title",
                type="str",
                description=(
                    "The title of the document that briefly decribe its content and its purpose."
                ),
            ),
            MetadataInfo(
                name="reference number",
                type="str",
                description=(
                    "A unique identifier or code assigned to the document for tracking and reference."
                ),
            ),
            MetadataInfo(
                name="issuing body",
                type="str",
                description=(
                    "The organization or authority responsible for creating and issuing the document."
                ),
            ),
            MetadataInfo(
                name="document type",
                type="str",
                description=(
                    "The category of the document based on its purpose or nature, for example: "
                    "directive, decision, notification, plan, template, etc."
                ),
            ),
            MetadataInfo(
                name="document field",
                type="str",
                description=(
                    "The area or domain the document applies to, such as: education, student affairs, "
                    "quality assurance, science and technology, etc."
                ),
            ),
            MetadataInfo(
                name="issuance date",
                type="str",
                description=(
                    "The date when the document was officially published or issued [by its issuing body], "
                    "represented as a date string, for example: 'Sat Dec 20 2024'."
                ),
            ),
            MetadataInfo(
                name="date of effect",
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
                name="validity status",
                type="str",
                description=(
                    "The current status of the document's applicability, one of ['valid', 'expired']."
                ),
            ),
            MetadataInfo(
                name="invalid date",
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
