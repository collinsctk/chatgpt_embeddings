# 参考文章
# https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb


import openai
import pandas as pd
import time
from gpt_0_basic_info import api_key, excel_file_path, csv_file_path

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(text: str, open_ai_api_key: str, model: str = EMBEDDING_MODEL) -> list[float]:
    openai.api_key = open_ai_api_key
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame, open_ai_api_key: str):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a datafram with embedding 
    }
    """
    df['embeddings'] = ''
    df['embeddings'] = df['embeddings'].astype('object')

    for idx, r in df.iterrows():
        print(idx)
        df.at[idx, 'embeddings'] = get_embedding(r.QandA, open_ai_api_key)
        time.sleep(1)

    return df


def getembeddings(api_key, excelfilepath, csvfilepath):
    df = pd.read_excel(excelfilepath)
    df['prompt'] = df['prompt'].apply(lambda x: x.replace('\n', ''))
    df['prompt'] = df['prompt'].apply(lambda x: "当有人问：" + x + '')
    df['completion'] = df['completion'].apply(lambda x: "请回答：" + x)
    df['QandA'] = df['prompt'] + df['completion']
    df = compute_doc_embeddings(df, api_key)[['QandA', 'embeddings']]
    df.to_csv(csvfilepath, index=False, encoding='utf-8_sig')


if __name__ == '__main__':
    getembeddings(api_key, excel_file_path, csv_file_path)
