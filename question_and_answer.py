import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings


embedings_df = pd.read_csv('embeddings.csv', index_col=0)
#下面这句话的意思是把df['embeddings']中的每一个元素都转换成list，然后把list转换成np.array，然后把np.array赋值给df['embeddings']
embedings_df['embeddings'] = embedings_df['embeddings'].apply(eval).apply(np.array)

# embedings_df.head()
# print(embedings_df)
# 写将环境变量OPENAI_API_KEY的值赋值给openai.api_key
# openai.api_key = os.environ["OPENAI_API_KEY"]
# 下面这个函数的作用是计算两个向量之间的距离

def create_context(
        question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    # print(openai.Embedding.create(input=question, engine='text-embedding-ada-002'))
    # print(openai.Embedding.create(input=question, engine='text-embedding-ada-002'))
    # Get the distances from the embeddings
    # 下面这句话的意思是把df['embeddings']中的每一个元素都转换成list，然后把list转换成np.array，然后把np.array赋值给df['embeddings']
    # 其中函数distances_from_embeddings的三个参数分别是q_embeddings表示问题的向量，df['embeddings'].values表示文本的向量，
    # distance_metric表示距离的计算方式，cosine表示余弦距离
    # 在这里，将这些距离值添加到DataFrame的一个新列distances中，以便后续根据距离值排序并选择最相似的一段文本作为上下文。
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # 遍历DataFrame，根据距离值distances的值排序，选择最相似的一段文本作为上下文，i表示行号，row表示行
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length，+4是因为每一段文本后面都有一个换行符
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


# result = create_context("what is GPT-4", embedings_df)
# print(result)


def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            # 这个prompt是用来生成答案的，prompt是一个字符串，它包含了上下文和问题，以及一个分隔符，用来告诉模型什么是上下文，什么是问题。
            prompt=f"Answer the question based on the context below, \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# debug=True表示打印出上下文，debug=False表示不打印上下文
respond1 = answer_question(embedings_df, question="tell me the url about waitlist", debug=False)


print(respond1)
