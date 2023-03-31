import openai
from gpt_2_create_question import decorate_query
from gpt_0_basic_info import api_key, csv_file_path
openai.api_key = api_key


def question(query):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        max_tokens=100,
        n=1,
        temperature=0.5,
    )

    output_text = response.choices[0].message['content'].strip()
    return output_text


if __name__ == '__main__':
    query = '亁颐堂是做什么的'
    new_query = decorate_query(query, api_key, filename=csv_file_path)
    print(new_query)
    # 产生如下的问题：
    """
    请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。

    上下文：
    当有人问：公司名称请回答：亁颐堂科技有限责任公司
    当有人问：亁颐堂是做什么的请回答：亁颐堂是一个网络培训公司
    
     问题: 亁颐堂是做什么的
     回答:？
    """
    # 把上面的问题打给OPEN AI, 就能正确的回答问题
    # 亁颐堂是一个网络培训公司。
    print(question(new_query))
