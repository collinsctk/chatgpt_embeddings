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
    print(question(new_query))
