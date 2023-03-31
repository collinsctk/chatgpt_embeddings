import numpy as np
import openai
import pandas as pd
import ast
from gpt_0_basic_info import api_key, excel_file_path, csv_file_path

MAXCONTEXTLEN = 1500

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(text: str, open_ai_api_key: str, model: str=EMBEDDING_MODEL) -> list[float]:

    openai.api_key = open_ai_api_key
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame, open_ai_api_key: str) :
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a datafram with embedding 
    }
    """
    df['embeddings'] = ''
    df['embeddings'] = df['embeddings'].astype('object')
    
    for idx, r in df.iterrows():
        df.at[idx, 'embeddings'] = get_embedding(r.content, open_ai_api_key)
    
    return df


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def get_query_similarity(query: str, df: pd.DataFrame, open_ai_api_key: str):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    openai.api_key = open_ai_api_key

    query_embedding = get_embedding(query, open_ai_api_key)
    
    #df['similarities'] = 0

    df['similarities'] = df['embeddings'].apply(lambda x:vector_similarity(query_embedding, x))
    
    #print(df['similarities'])
    '''
    for idx, r in df.iterrows():
        df.loc[idx, 'similarities'] = vector_similarity(query_embedding, r.embeddings)
    '''
    
    two_largest = df['similarities'].nlargest(2).index.tolist()

    # print('get_query_similarity!!!!!!!!')

    context = '' if df.loc[two_largest[0]]['similarities'] < 0.8 else df.loc[two_largest[0]]['QandA'] if (df.loc[two_largest[1]]['similarities'] < 0.8 or (len(df.loc[two_largest[1]]['QandA'] + '\n' + df.loc[two_largest[0]]['QandA'])>=MAXCONTEXTLEN))  else (df.loc[two_largest[1]]['QandA'] + '\n' + df.loc[two_largest[0]]['QandA']) 
    # print(two_largest[0], df.loc[two_largest[0]]['similarities'], df.loc[two_largest[0]]['QandA'])
    # print(two_largest[1], df.loc[two_largest[1]]['similarities'], df.loc[two_largest[1]]['QandA'])
    # print(len(df.loc[two_largest[1]]['QandA'] + '\n' + df.loc[two_largest[0]]['QandA']))
    # print(context)

    return context


def _decorate_query(query: str, df: pd.DataFrame, open_ai_api_key: str)-> str:

    try:
        context = get_query_similarity(query, df, open_ai_api_key)
        if context != '':
            header = """请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。\n\n上下文：\n"""
            #header = "上下文：\n"
            query = header + context + "\n\n 问题: " + query + "\n 回答:？"
            # print(query)
        return query
    except:
        # print('ERROR 444444')

        return query


def decorate_query(query: str, open_ai_api_key, filename='foodsembeddings.csv')-> str:
    filepath = filename
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return query
        else:
            try:
                df['embeddings'] = df['embeddings'].apply(lambda x: ast.literal_eval(x))
                return _decorate_query(query, df, open_ai_api_key)
            except Exception as e:
                print(e)
                return query
    except Exception as e:
        print(e)
        return query


if __name__ == '__main__':
    query = '亁颐堂是做什么的'
    print(decorate_query(query, api_key, filename=csv_file_path))\
    # 产生问题
    """
    请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。

    上下文：
    当有人问：公司名称请回答：亁颐堂科技有限责任公司
    当有人问：亁颐堂是做什么的请回答：亁颐堂是一个网络培训公司
    
     问题: 亁颐堂是做什么的
     回答:？
    """


