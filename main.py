import os
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext, load_index_from_storage
import streamlit as st

os.environ['OPENAI_API_KEY'] = ''
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1024)
Settings.context_window = 15000
# d = 1536
# faiss_index = faiss.IndexFlatL2(d)
file_path = "quy_dinh.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
# vector_store = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=70)
doc = Document(text=content)
documents = []
documents.append(doc)
nodes = splitter.get_nodes_from_documents(documents)
# index = VectorStoreIndex(
#     nodes
# )
# index.storage_context.persist()
storage_context = StorageContext.from_defaults(persist_dir="storage")
# load index
index = load_index_from_storage(storage_context)
context_str = '{context_str}'
query_str = '{query_str}'
final_prompt_tmpl_str = f"""
Chỉ được sử dụng thông tin dưới đây để trả lời câu hỏi của nhân viên. Trả lời bằng tiếng việt và nêu rõ điều luật tài liệu liên quan. Nếu không có trong tài liệu, chỉ cần trả lời không có thông tin trong quy định
Context: {context_str}
Query: {query_str}
Answer:
"""
# additional_args = {"context_str": "{context_str}", "query_str": "{query_str}", "few_shot_examples": few_shot_examples}
final_prompt_tmpl = PromptTemplate(
    final_prompt_tmpl_str,
    # format =additional_args,
)
retriever = index.as_retriever(similarity_top_k=4)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()
query_engine_retriever = RetrieverQueryEngine.from_args(
    retriever
    , response_mode='compact'
)
query_engine_retriever.update_prompts(
    {"response_synthesizer:text_qa_template": final_prompt_tmpl}
)

# retriever = index.as_retriever(similarity_top_k=5)
# retriever.retrieve("1 tháng được đi muộn bao nhiêu lần")
# answer = query_engine_retriever.query('30/4 được nghỉ có lương không')
# print(answer)
# được nghỉ có lương trong những trường hợp nào
# làm hỏng laptop công ty cấp thì xử lý thế nào
# hồ sơ cần nộp những gì
# làm ngoài giờ được thêm lương không
# được ngủ qua đêm tại công ty không

original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Example</h1>' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Được nghỉ có lương trong những trường hợp nào</h2>' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Làm hỏng laptop công ty cấp thì xử lý thế nào</h2>' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Hồ sơ cần nộp những gì</h2>' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Được ngủ qua đêm tại công ty không</h2>'
st.markdown(original_title, unsafe_allow_html=True)
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://img.lovepik.com/background/20211021/medium/lovepik-dark-background-image_400109825.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .reportview-container .main .block-container div[data-baseweb="toast"] {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title('Hỏi đáp nội quy')

with st.form('my_form'):
    text = st.text_area('Input')
    submitted = st.form_submit_button('Run')
    if submitted:
        st.info(query_engine_retriever.query(text).response)
# được nghỉ có lương trong những trường hợp nào
# làm hỏng laptop công ty cấp thì xử lý thế nào
# hồ sơ cần nộp những gì
# làm ngoài giờ được thêm lương không
# được ngủ qua đêm tại công ty không
