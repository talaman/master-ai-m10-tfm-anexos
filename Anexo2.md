# Anexo 2: Pipeline de ingestión de datos

## Variable de entorno para OpenAI en archivo .env: 

```markdown
OPENAI_API_KEY=<openai-key>
```

## Librerias usadas definidas en archivo pipeline/requirements.txt:

```markdown
langchain
langchain_openai
langchain_community
faiss-cpu
openai
pypdf
dotenv
cryptography>=3.1 
```

## Código del pipeline de ingestión de datos, desde carga de archivos hasta generación de archivo de vectores en FAISS:

```python
from dotenv import load_dotenv
load_dotenv()  # Carga las variables de entorno desde un archivo .env
 
def load_data(path):
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    print(":::Getting documents:::")
    directory_loader = PyPDFDirectoryLoader(path)  # Crea un cargador de directorio para archivos PDF
    documents = directory_loader.load()  # Carga los documentos desde el directorio especificado
    return documents  # Retorna los documentos cargados
 
def split_data(documents, chunk_size=1000, chunk_overlap=100):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print(":::Splitting data:::")  # Imprime un mensaje indicando que se están dividiendo los datos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    # Crea un divisor de texto con los parámetros especificados
    split_documents = text_splitter.split_documents(documents)  # Divide los documentos en fragmentos
    return split_documents  # Retorna los documentos divididos
 
def openai_generate_embeddings(documents, name):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    print(":::Generating OpenAI embeddings:::")
    embeddings = OpenAIEmbeddings()  # Crea un objeto de embeddings de OpenAI
    vector_store = FAISS.from_documents(documents, embeddings)  # Crea un almacén vectorial a partir de los documentos y los embeddings
    print("vector_store.index.ntotal", vector_store.index.ntotal)  # Imprime el número total de vectores en el almacén
    vector_store.save_local(name)  # Guarda el almacén vectorial localmente con el nombre especificado
 
def test_vector_store(name):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    print(":::Testing vector store:::") 
    embeddings = OpenAIEmbeddings()  # Crea un objeto de embeddings de OpenAI
    vector_store = FAISS.load_local(name, embeddings, allow_dangerous_deserialization=True)  # Carga el almacén vectorial localmente con el nombre especificado
    query = "Como es Libra con ascendente en Aries?"  # Define una consulta de ejemplo
    retriever = vector_store.as_retriever()  # Crea un recuperador a partir del almacén vectorial
    docs = retriever.invoke(query)  # Invoca la consulta en el recuperador
    print(f"Query: {query}")  # Imprime la consulta
    print(f"Retrieved {len(docs)} documents")  # Imprime el número de documentos recuperados
    print(f"Top document: {docs[0].page_content}")  # Imprime el contenido del documento principal recuperado
    print(":::Testing done:::")  
 
def data_ingestion(data_path="data/reviewed", faiss_path="api/faiss/openai"):
    documents = load_data(data_path)  # Carga los datos desde el directorio especificado
    documents = split_data(documents, chunk_size=2000, chunk_overlap=200)  # Divide los datos en fragmentos
    openai_generate_embeddings(documents, faiss_path)  # Genera embeddings de OpenAI y guarda el almacén vectorial
    test_vector_store(faiss_path)  # Prueba el almacén vectorial
 
if __name__ == "__main__":
    data_ingestion()  # Ejecuta la función principal de ingestión de datos si el script se ejecuta directamente
```

## Output ejemplo del pipeline:
```markdown
:::Getting documents:::
Object 207 0 not defined.
Object 208 0 not defined.
Object 209 0 not defined.
:::Splitting data:::
:::Generating OpenAI embeddings:::
vector_store.index.ntotal 361
:::Testing vector store:::
Query: Como es Libra con ascendente en Aries?
Retrieved 4 documents
Top document: entonces esta orientación de los equinoccios basado s en el hemisferio norte, debido al polo 
norte dominante en la Tierra, debería ser reconsiderado. Si el polo sur ahora, resulta ser el 
dominante, entonces necesitaríamos usar Libra más q ue Aries en el comienzo de la primavera 
del zodiaco, y Leo más que Acuario como nuestra equ inoccial edad constelacional. Todo 
interesante y yo supongo tendremos que esperar hast a que el polo cambie y ver que sucede.
:::Testing done::: 
```	


 

