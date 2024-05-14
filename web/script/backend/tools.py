from tkinter import Tk, filedialog
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
import os
# ==============================WANT TO DO================================= 
# 使用したPDFを保存して利用できるようにしたい
# =========================================================================

# PDFファイルを取得
def get_pdf():
    
    # PDFファイル選択GUIの表示
    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilename(
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]  # file形式をPDF形式のみに限定
        )
    # files = "../../database/pdf/bitcoin.pdf"
    # files = ""
    
    text = ""  # 抽出したテキストの格納する変数を初期化
    if files:
        reader = PdfReader(files)
        pages_num = len(reader.pages)
        for i in range(pages_num):  # ページごとに読み込む
            page = reader.pages[i]
            text += page.extract_text()
    root.destroy()
    
    # text = "Bitcoin is a dog. Bitcoin is popular in Japan. Bitcoin likes to be named BIBI."

    return text  # 抽出したテキストを出力

def get_pdf2():
    # PDFファイル選択GUIの表示
    # root = Tk()
    # root.withdraw()
    # files = filedialog.askopenfilename(
    #     filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]  # file形式をPDF形式のみに限定
    #     )
    # files = str(os.getenv("WORKSPACE_PATH"))+"/web/database/pdf/bitcoin.pdf"
    file = "https://bitcoin.org/bitcoin.pdf"
    
    documents = []  # 抽出したテキストの格納する変数を初期化
    if file:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        
        documents.extend(pages)
    # root.destroy()
    
    return documents