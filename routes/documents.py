from flask import Blueprint, jsonify, request
from database import chat_db
import pdfplumber
import openai
import pinecone
import langchain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings

documents_route = Blueprint("documents_route", __name__)


# Endoint for sending message
@documents_route.route("/sendMessage", methods=["POST"])
def send_message():
    data = request.get_json()
    message = data["message"]
    document_id = data["document_id"]
    pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index("langchainvector")
    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    vectors = embeddings.embed_query(message)
    response = index.query(
        namespace=document_id,
        vector=vectors,
        top_k=2,
        include_metadata=True,
    )
    prompt_template = """
    Context: {context}\n\n
    Question: {question}\n\n
    """
    context = ""
    for match in response["matches"]:
        metadata = match["metadata"]
        text = metadata["text"]
        context += text
    prompt = prompt_template.format(context=context, question=message)
    openai.api_key = settings.OPENAI_API_KEY
    client = openai.Client(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Answer the question as detailed as possible from the provided context and make sure to provide all the details, if the answer is not present in the context, please say 'I don't know', don't provide the wrong answer and give to the point answers only no extra information.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return jsonify({"response": response.choices[0].message.content}), 200


# Endpoint for getting documents
@documents_route.route("/getDocuments", methods=["GET"])
def get_documents():
    documents = chat_db["documents"].find()
    documents_list = []
    for document in documents:
        documents_list.append(
            {"id": str(document["_id"]), "filename": document["filename"]}
        )
    return jsonify(documents_list), 200


# Endpoint for uploading a document
@documents_route.route("/uploadDocument", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        pdf = pdfplumber.open(file)
        text = ""
        document_id = (
            chat_db["documents"].insert_one({"filename": file.filename}).inserted_id
        )

        pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index("langchainvector")
        for page in pdf.pages:
            text += page.extract_text()
        pdf.close()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

        counter = 1
        for chunk in chunks:
            vectors = embeddings.embed_query(chunk)
            vectors_obj = {
                "id": str(document_id) + "_chunk_" + str(counter),
                "values": vectors,
                "metadata": {"text": chunk},
            }
            vectors_arr = [vectors_obj]
            index.upsert(vectors_arr, namespace=str(document_id))
            counter += 1

        return jsonify({"message": "Document uploaded successfully"}), 200
