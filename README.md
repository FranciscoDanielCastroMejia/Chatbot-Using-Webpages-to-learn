# Chatbot with Memory and Web Page Learning Using LangChain and Streamlit
In this project, a chatbot was developed using langchain, where the chatbot is provided with a web page and it learns what is on that page, and it can answer questions related to the web page. It is also worth mentioning that the chatbot has memory of what has been previously mentioned in the conversation in order to give more specific answers. The graphical interface of the chatbot was developed in streamlit.

---
## Code
### Importing libraries 
```python
import streamlit as st
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.llm import LLMChain 
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
```
---
### Importing the API Keys 
You will have to get your API Key from OpenAI, create a file ".env", and put the next:
```python
OPENAI_API_KEY=YOUR_OPENAI_KEY
```
---
### Rest of the code
```python
#___________________________Title of the page in streamlit_________________________
st.title("ChatBot Inteligente with memory and web")

# Función para eliminar el directorio de persistencia, en caché para que se ejecute solo una vez
@st.cache_data
def eliminar_directorio(persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        return 
    return 

#________Seleccionar entre subir archivo de texto o pagina web_____________
        
webpage_url = st.text_input('Ingrese la URL de la PAgina Web', type="default")


#_____________Here we transform the website to a vecto_______________________
       
if webpage_url:

    # Ruta del directorio de persistencia
    persist_directory = "./chroma_db"
    eliminar_directorio(persist_directory)

    #laod the webpage
    loader = WebBaseLoader(webpage_url)
    pagina = loader.load()

    #splotter document
    c_splitter = CharacterTextSplitter(chunk_size = 500,chunk_overlap = 75, separator = ' ')
    docs = c_splitter.split_documents(pagina)
    
    #defining the embedder
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Función para inicializar o cargar la base de datos de vectores
    def cargar_o_inicializar_chroma(docs, embedding, persist_directory=persist_directory):
        if os.path.exists(persist_directory):
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            print("Base de datos de vectores cargada.")
        else:
            # Crear y persistir la base de datos si no existe
            vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
            vectordb.persist()
            print("Base de datos de vectores creada y guardada.")

        return vectordb
    
    vectordb = cargar_o_inicializar_chroma(docs, embedding, persist_directory)


    st.success(f"Cargada la URL {webpage_url} exitosamente!!")
    st.markdown("---")


    #Importing the model
    llm1 = ChatOpenAI(model="gpt-4o", temperature=0)
    llm2 = Ollama(model='llama3', temperature=0)

    # Crear el selectbox para seleccionar el modelo
    modelo_seleccionado = st.selectbox(
        'Select the model you want to use:',
        ['No model','GPT-4', 'Llama3']  # Opciones legibles para el usuario
    )

    llm = modelo_seleccionado



    if modelo_seleccionado != 'No model':

        # Asignar el modelo correspondiente según la selección
        if modelo_seleccionado == 'GPT-4':
            llm = llm1
        elif modelo_seleccionado == 'Llama3':
            llm = llm2

        #___________________________Prompt Template______________________________

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """Eres un asistente virtual llamado "RealtyBot" especializado en ayudar a los clientes interesados en propiedades inmobiliarias. Tu objetivo es ofrecer un servicio rápido, amigable y eficiente, respondiendo preguntas sobre propiedades disponibles, precios, ubicaciones, características y procesos de compra o renta. Toda la información relevante sobre las propiedades está disponible en la página {context}. Mantén un tono profesional y cercano, adaptando tu lenguaje a un español neutro.

                    No hagas suposiciones ni proporciones información sobre propiedades que no esté incluida en {context}.

                    **Estilo y tono:**
                    - Usa un lenguaje profesional y amigable, con frases como "Con gusto te ayudo" o "Déjame verificar esa información para ti" para generar confianza.
                    - Mantén las respuestas claras y concisas, pero incluye detalles relevantes cuando sea necesario.

                    **Personalidad:**
                    - Actúa como un experto inmobiliario, apasionado por encontrar la propiedad ideal para cada cliente y con una actitud resolutiva.
                    - Sé paciente, servicial y proactivo, ofreciendo siempre información relevante y transmitiendo entusiasmo por ayudar al cliente.

                    **Conocimiento:**
                    - Responde únicamente utilizando la información disponible en {context}.
                    - Si no tienes información sobre una propiedad o tema, indica que no puedes ayudar directamente y sugiere consultar con un asesor inmobiliario.
                    - Sé específico y detallado al describir las propiedades, incluyendo características como tamaño, ubicación, precios, y servicios cercanos, siempre basándote en {context}.

                    **Contexto:**
                    - Toma en cuenta el historial de la conversación para dar respuestas coherentes. Si el cliente hace una pregunta ambigua, solicita más detalles para aclarar.
                    - Si el cliente pregunta sobre una propiedad específica, ofrece opciones como "¿Podrías proporcionarme el ID de la propiedad o la ubicación para buscarla?"

                    **Formato de respuesta:**
                    - Usa una introducción breve, seguida de una respuesta clara y bien estructurada.
                    - Organiza la información en listas o tablas cuando sea necesario (por ejemplo, características de una propiedad o rangos de precios).
                    - Incluye pasos detallados si el cliente solicita ayuda con el proceso de compra, renta o agendar una visita.

                    **Manejo de incertidumbre:**
                    - Si no estás seguro o no tienes información, responde con frases como "Lo siento, no tengo esa información en este momento, pero puedo ayudarte con otra consulta."

                    **Temas sensibles o restricciones:**
                    - Evita dar asesoramiento financiero o legal. Si el cliente lo solicita, redirige al asesor correspondiente.
                    - No especules sobre precios futuros, valores de mercado o temas fuera de la información disponible en {context}.

                    **Capacidades adicionales:**
                    - Ofrece sugerencias proactivas como: "¿Te gustaría conocer propiedades similares en esta área?" o "Puedo ayudarte a filtrar propiedades según tus preferencias, ¿te interesa?"
                    - Ayuda con cálculos básicos como estimaciones de costos mensuales de renta o financiamiento.

                    **Lenguaje y localización:**
                    - Responde en español neutro para clientes de cualquier región, evitando regionalismos. Si el cliente usa términos locales (como "departamento" o "piso"), adapta tu respuesta a ese lenguaje.

                    **Aprendizaje continuo:**
                    - Si el cliente indica que no encontró útil tu respuesta, pide disculpas, solicita más detalles y ofrece una nueva solución.

                    **Ejemplo de respuesta:**
                    Cliente: "¿Cuánto cuesta la casa en el centro de la ciudad?"
                    RealtyBot: "¡Claro! Según la información en nuestra página {context}, la casa ubicada en el centro tiene un precio de $2,500,000 MXN. Tiene 3 recámaras, 2 baños y estacionamiento para 2 autos. ¿Te gustaría agendar una visita o recibir más detalles sobre esta propiedad?""",
                ),

                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )   



        #___________________________Chain______________________________

        chain = prompt_template | llm
        #chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

        #___________________________QA Chain______________________________

        retriever = vectordb.as_retriever()
        
        #___________funciones para juntar el promp template con el archivo________
        
        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def rag_chain(pregunta):
            retriever_docs = retriever.invoke(pregunta)
            formatted_context = combine_docs(retriever_docs)
            response = chain.invoke({"input":pregunta, "context":formatted_context, "chat_history":st.session_state.chat_history})

            if modelo_seleccionado == 'GPT-4':
                return response.content
            
            return response



        #______________ Inicializar el historial del chat desplegado_____________________
        if "messages" not in st.session_state:
            st.session_state.messages = []

        #___________________________Chat history______________________________
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []


        #________________ Mostrar en pantalla el historial de la conversacion________________

        for message in st.session_state.messages: #se itera sobre cada mensaje
            with st.chat_message(message["role"]): #message tiene la forma de 
                st.markdown(message["content"])


        
        if prompt := st.chat_input("Pregunta lo que necesites..."):
            
            # Monstrar mensaje del usuario en el contenedor de mensajes del chat
            st.chat_message('user').markdown(prompt)
            # Agregar mensaje del usuario al historial del chat
            st.session_state.messages.append({'role':"user", "content":prompt}) #guardar en el historial del chat


            #response = qa_chain.invoke({"input":prompt, "chat_history":st.session_state.chat_history})
            response = rag_chain(prompt)

            # Agregar la entrada del usuario al historial y la respuesta del Bot al historial
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Monstrar respuesta del asistente en el contenedor de mensjaes del chat
            with st.chat_message("assistant"):
                st.markdown(response)
            # Agregar mensaje del asistente al historial del chat
            st.session_state.messages.append({'role':"assistant", "content":response})
```
## Result
For the example we that is showen in the video result ,we used a webpage that is from a real estate company. The webpage is shown in the next image:
![WebPage](assets/webpage_image.png)
