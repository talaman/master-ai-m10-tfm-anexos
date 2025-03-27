# Anexo 3: API con FastAPI, LangChain y FAISS


## Variable de entorno para OpenAI en archivo .env: 

```markdown
OPENAI_API_KEY=<openai-key>
```	

## Librerias usadas definidas en archivo pipeline/requirements.txt:

```markdown
langchain 
langchain-openai
faiss-cpu
langchain-community
langchain-cli
fastapi
uvicorn
python-multipart
```

## Código de la API con sistema RAG utilizando LangChain, FAISS vector database y FastAPI:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
import uvicorn
from pydantic import BaseModel
 
load_dotenv()  # Carga las variables de entorno desde un archivo .env
 
print(":::Loading vector database:::")
embeddings = OpenAIEmbeddings()  # Inicializa las incrustaciones de OpenAI
vector_store = FAISS.load_local(
    "faiss/openai", embeddings, allow_dangerous_deserialization=True)  # Carga la base de datos vectorial FAISS
retriever = vector_store.as_retriever()  # Configura el recuperador de documentos
 
print(":::Loading OpenAI model:::")
model = ChatOpenAI(model="o3-mini",temperature=1)  # Inicializa el modelo de chat de OpenAI
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an astrologer, tasked with answering questions about astrology. Please respond in the same language as the question without mentioning the language, it is most likely that you will use Spanish. 
            Follow these specific guidelines based on the type of question:
            1. If the question is a general greeting (e.g., "hello"): Respond with a friendly greeting only. Do not reference the provided context or add any additional information.
            2. If the question is unrelated to astrology: Disregard the context and state that you can only answer astrology-related questions. Provide a polite and brief explanation.
            3. If the question is about astrology and the context is relevant: Use the provided context to inform your answer, supplemented by your prior knowledge. Do not copy the text verbatim.
            4. If the question is about astrology but the context is not relevant: Disregard the context and answer based on your prior knowledge only.
            If you don't know the answer, simply state that you don't know. Your answer should reflect the persona of an astrologer and be kind. Aim for a response that is at least 200 words long, simple, and clear, but it can be shorter or longer when it makes sense.
 
            For example:
            Question: What is the significance of the moon in astrology?
            Answer: The moon is a significant celestial body in astrology, representing emotions, intuition, and the subconscious mind. It influences our emotional responses, instincts, and habits, shaping our inner world and emotional well-being. The moon's placement in the natal chart reveals our emotional needs, how we nurture ourselves and others, and our instinctual reactions to life events. It also governs our relationship with our mother and the feminine aspects of our personality. The moon's phase and sign placement further refine its influence on our emotional landscape and personal growth.
            
            <example>
            Question: Hello!
            Answer: Hello! How can I assist you today?
 
            Question: What is the capital of France?
            Answer: I can only answer questions related to astrology. Please ask an astrology-related question for more information.
 
            Question: How does the position of the planets influence our daily lives?
            Answer: The position of the planets in the zodiac at the time of our birth influences our personality traits, behaviors, and life events. Each planet governs specific aspects of our lives and personality, such as Mercury for communication, Venus for love and relationships, and Mars for energy and motivation. The planets' placements in the natal chart create unique patterns that shape our individuality and experiences. By understanding the planetary influences in our birth chart, we can gain insight into our strengths, challenges, and life purpose.
 
            Question: What is the significance of a sun-moon conjunction in astrology?
            Answer: The sun-moon conjunction is a powerful aspect in astrology, symbolizing the blending of our conscious and unconscious selves. This alignment represents a harmonious integration of our ego (sun) and emotions (moon), creating a sense of wholeness and self-awareness. Individuals with a sun-moon conjunction often possess a strong sense of identity and emotional intelligence, allowing them to navigate life with confidence and authenticity. This aspect enhances creativity, intuition, and self-expression, enabling individuals to align their inner desires with their external actions. The sun-moon conjunction is a potent force for personal growth and self-realization.
 
            Question: Thanks
            Answer: You're welcome! Feel free to ask any astrology-related questions you may have.
            
            Question: Good bye
            Answer: Good bye! Have a wonderful day!
            </example>

            <context>
            {context}
            </context>
            
            """
        ),
        ("human", "{input}"),
    ]
)
 
def format_docs(docs): # Formatea los documentos recuperados en una cadena para el contexto
    """Formats retrieved documents into a string for context."""
    return "\n\n".join(doc.page_content for doc in docs)
 
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)  # Configura la cadena RAG (Retrieve and Generate) con el recuperador, el prompt, el modelo y el parser de salida
 
app = FastAPI( title="AstralGuru API Server",
    version="1.0",
    description="An API for AstralGuru LangChain")  # Inicializa la aplicación FastAPI con título, versión y descripción
 
class HistoryItem(BaseModel): # Define un modelo de datos para un elemento de historial
    content: str # Define un campo de contenido de texto
    type: str # Define un campo de tipo de texto
 
class ChatRequest(BaseModel): # Define un modelo de datos para una solicitud de chat
    input: str # Define un campo de entrada de texto
    history: list[HistoryItem] = [] # Define un campo de historial de texto
 
@app.post("/chat") # Define una ruta de chat
async def chat(request: ChatRequest): 
    input_text = request.input 
    history = request.history
    history_text = "".join(f"<{item.type}>{item.content}</{item.type}>" for item in history) if history else "" # Formatea el historial de texto
    full_input = f"{history_text}<human>{input_text}</human>" # Combina el historial y la entrada de texto
    response = rag_chain.invoke(full_input) # Invoca la cadena RAG con la entrada completa
    return {"output": response} 
 
```	