from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Embeddings:
    
    def __init__(self, device):
        self.device = device
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={'device': str(device)+':0'}
        )
        
        
    def text_splitter(self, text):
        document = [Document(page_content=text)]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200)
        splitted_texts = text_splitter.split_documents(document)
        
        return splitted_texts
    
    
    def text_splitter2(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=20, 
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],)
        splitted_texts = text_splitter.split_documents(documents)
        
        return splitted_texts
    

    def embedding(self, index_path, text):
        splitted_text = self.text_splitter2(text)
        new_vec = FAISS.from_documents(splitted_text, self.embeddings)
        try:
            vector = FAISS.load_local(
                folder_path="../database/vector_store/"+index_path, 
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True)
            vector.merge_from(new_vec)
        except:
            vector = new_vec
        vector.save_local("../database/vector_store/"+index_path)
        
        return vector
    

    def load_vector(self, index_path):
        vector = FAISS.load_local(
            folder_path="../database/vector_store/"+index_path, 
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True)
        
        return vector