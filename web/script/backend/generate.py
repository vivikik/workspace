from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
# from mlx_lm import load, generate
import torch
from accelerate import Accelerator
import os
import time
from embedding import Embeddings
from tools import get_pdf, get_pdf2

# try:
#     from vllm import LLM, SamplingParams
#     from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
# except:
#     print("\n===================================================================================")
#     print(f"DEBUG: failed to install vllm")
#     print("===================================================================================\n")


class Generate:
    
    def __init__(self):
        self.max_new_token = 512
        self.model_id = (
            "microsoft/Phi-3-mini-4k-instruct"
            # "microsoft/Phi-3-mini-128k-instruct"
            # "inu-ai/dolly-japanese-gpt-1b"
            # "andreaskoepf/pythia-1.4b-gpt4all-pretrain"
            # "elyza/ELYZA-japanese-Llama-2-7b-instruct"
            )
        self.model_name = (
            "Phi-3-mini-4k-instruct"
            # "Phi-3-mini-128k-instruct"
            # "dol_jp"
            # "gpt4all"
            # "elyza"
        )
        
        # モデルのダウンロード
        try:
            print("\n"+"="*100)
            print(f"DEBUG: start loading "+self.model_id)
            print("="*100+"\n")
            self.tokenizer = AutoTokenizer.from_pretrained("../model/"+self.model_name+"/"+self.model_name+"_tokenizer")
            self.model = AutoModelForCausalLM.from_pretrained(
                "../model/"+self.model_name+"/"+self.model_name+"_model",
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                use_cache=True,
                use_flash_attention_2=torch.cuda.is_available(),
                ).eval()
            print("\n"+"="*100)
            print(f"DEBUG: finish loading "+self.model_id)
            print("="*100+"\n")
        except:
            print("\n"+"="*100)
            print(f"DEBUG: start downloading "+self.model_id)
            print("="*100+"\n")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                use_cache=True,
                use_flash_attention_2=torch.cuda.is_available(),
                ).eval()
            
            if not os.path.exists("../model/"+self.model_name+"/"+self.model_name+"_tokenizer"):
                self.tokenizer.save_pretrained("../model/"+self.model_name+"/"+self.model_name+"_tokenizer")
            if not os.path.exists("../model/"+self.model_name+"/"+self.model_name+"_model"):
                self.model.save_pretrained("../model/"+self.model_name+"/"+self.model_name+"_model")
                
            print("\n"+"="*100)
            print(f"DEBUG: finish downloading "+self.model_id)
            print("="*100+"\n")
            
            self.template = """
            You are a movie recommender system that help users to find anime that match their preferences. 
            Use the following pieces of context to answer the question at the end. 
            For each question, suggest three anime, with a short description of the plot and the reason why the user migth like it.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Your response:"""

            self.prompt = PromptTemplate(
                template=self.template, 
                input_variables=["context", "question"]
                )
            
            self.chain_type_kwargs = {"prompt": self.prompt}

    def generate_text(self, question, use_rag, index_path):
        torch.random.manual_seed(0)

        accelerator = Accelerator()
        device = accelerator.device
        print("\n"+"="*100)
        print("DEBUG: current device is "+str(device))
        print("="*100+"\n")

        model = torch.compile(model=self.model).to(device)  # モデルを事前にコンパイルして高速化
        task = "text-generation"
        pipe = pipeline(
            task, 
            model=model,
            tokenizer=self.tokenizer,
            framework='pt',
            max_new_tokens=self.max_new_token,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        
        if use_rag:
            text = get_pdf2()
            
            start = time.time()  # 現在時刻（処理開始前）を取得
            if text:  
                vector = Embeddings(device=device).embedding(index_path=index_path, text=text)  #  テキストを使用してベクトルを更新
            else:
                vector = Embeddings(device=device).load_vector(index_path=index_path)  # DBから保存済みベクトルを取得
            end = time.time()  # 現在時刻（処理完了後）を取得

            time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
            print("\n"+"="*100)
            print("DEBUG: embedding text time is "+str(time_diff))  # 処理にかかった時間データを使用
            print("="*100+"\n")

            
            if vector:
                retriever = vector.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
                print("\n"+"="*100)
                print("DEBUG: retriever setup finished")
                print("="*100+"\n")
                
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    return_source_documents=True,
                    verbose=True,
                )
                print("\n"+"="*100)
                print("DEBUG: qa setup finished")
                print("="*100+"\n")

                start = time.time()  # 現在時刻（処理開始前）を取得
                try:
                    print("\n"+"="*100)
                    print("DEBUG: qa({'query':question}) style")
                    print("="*100+"\n")
                    result = qa({'query':question})
                    print("\n"+"="*100)
                    print('Answer:', result['result'])
                    print("="*100+"\n")
                    print("\n"+'='*100)
                    print('source:', result['source_documents'])
                    print("="*100+"\n")
                except:
                    print("\n"+"="*100)
                    print("ERROR: output answer errror")
                    print("="*100+"\n")
                
                end = time.time()  # 現在時刻（処理完了後）を取得

                time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
                print("\n"+"="*100)
                print("DEBUG: generate text finished")
                print("DEBUG: generate text time is "+str(time_diff))  # 処理にかかった時間データを使用
                print("="*100+"\n")
                
        else:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            DEFAULT_SYSTEM_PROMPT = "Answer the user's questions as accurately as possible based on the reference information."
            text = "{context}\nQuetion from user is following.{question}"
            template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=self.tokenizer.bos_token,
                b_inst=B_INST,
                system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
                prompt=text,
                e_inst=E_INST,
            )
            start = time.time()  # 現在時刻（処理開始前）を取得
            sum_start = time.time()
            inputs = template.format(context='', question=question)
            inputs = self.tokenizer(inputs, return_tensors='pt').to(model.device)
            end = time.time()  # 現在時刻（処理完了後）を取得

            time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
            print("\n"+"="*100)
            print("DEBUG: input setup finished")
            print("DEBUG: input setup time is "+str(time_diff))  # 処理にかかった時間データを使用
            print("="*100+"\n")

            start = time.time()  # 現在時刻（処理開始前）を取得
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            end = time.time()  # 現在時刻（処理完了後）を取得

            time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
            print("\n"+"="*100)
            print("DEBUG: output setup finished")
            print("DEBUG: output setup time is "+str(time_diff))  # 処理にかかった時間データを使用
            print("="*100+"\n")

                
            start = time.time()  # 現在時刻（処理開始前）を取得
                       
            output = self.tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
            print(output)
            
            end = time.time()  # 現在時刻（処理完了後）を取得

            time_diff = end - sum_start  # 処理完了後の時刻から処理開始前の時刻を減算する
            print("\n"+"="*100)
            print("DEBUG: generate text finished")
            print("DEBUG: generate text time is "+str(time_diff))  # 処理にかかった時間データを使用
            print("="*100+"\n")
            
        # start = time.time()  # 現在時刻（処理開始前）を取得

        # response = retrieval_chain.invoke({"input": question})
        # print(response["input"])
        # print(response["answer"])
        # end = time.time()  # 現在時刻（処理完了後）を取得

        # time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
        # print("\nDEBUG: generate text time is "+str(time_diff))  # 処理にかかった時間データを使用
       
        
        
if __name__ == "__main__":
    Generate().generate_text(question="What is bitcoin?", use_rag=True, index_path="bitcoin")