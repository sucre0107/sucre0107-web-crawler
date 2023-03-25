import pandas as pd #pandas是一个开源的，BSD许可的库，提供高性能，易于使用的数据结构和数据分析工具，主要用于数据挖掘和数据分析，这里用于读取csv文件，处理数据，生成dataframe
import os
import openai
import tiktoken
import numpy as np
from openai.embeddings_utils import distances_from_embeddings

from main import domain
from remove_newlines import remove_newlines
from transformers import BertTokenizer
if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


    # Create a list to store the text files
    texts=[]

    # Get all the text files in the text directory
    for file_name in os.listdir("text/" + domain + "/"):

        # Open the file and read the text
        with open("text/" + domain + "/" + file_name, "r", encoding="UTF-8") as f:
            text = f.read()

            # 省略文件名的前11个和最后4个，这个例子中是"openai.com"和".txt"被省略，然后将文本中的字符“-”、“_”和“#update”替换为空格。texts是一个列表，列表中的每个元素都是一个元组，元组中的第一个元素是文件名，第二个元素是文件内容。
            texts.append((file_name[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))
    # 下面的代码将文本文件的内容转换为数据框（DataFrame）。这里的texts是一个列表，列表中的每个元素都是一个元组，元组中的第一个元素是文件名，第二个元素是文件内容。
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + "." + remove_newlines(df.text)
    #需要先新建scrapped.csv文件，然后再运行下面的代码
    df.to_csv('processed/scraped.csv')
    df.head()




    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model，这里是用于英文的
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer_for_chinese = BertTokenizer.from_pretrained('bert-base-chinese')

    #index_col是pandas库中read_csv函数的一个参数，用于指定哪一列作为数据框（DataFrame）的行索引。当您使用index_col=0时，表示将CSV文件中第一列的值作为数据框的行索引。
    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    print("原始的df数据")
    print(df.head())
    # 指使用直方图来可视化数据框中每一行中 token 数量的分布情况
    df.n_tokens.hist()




    max_tokens = 500


    # Function to split the text into chunks of a maximum number of tokens
    def split_into_many(text, max_tokens=max_tokens):
        # Split the text into sentences
        sentences = text.split('.')

        # Get the number of tokens for each sentence
        # n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]  -->这个是简写
        n_tokens = []
        for sentence in sentences:
            n_token=len(tokenizer.encode(" " + sentence))
            n_tokens.append(n_token) #这个是完整写法
        # 解释一下n_token=len(tokenizer.encode(" " + sentence)) 这个是把每个句子加上空格，
        # 然后用tokenizer.encode()编码，然后用len()计算编码后的长度，这个长度就是token的长度，
        # 为何要加空格呢，因为tokenizer.encode()是把句子编码成token，然后在token前面加上[CLS]，在token后面加上[SEP]，所以要加空格，这样就可以把[CLS]和[SEP]也算进去了
        chunks = []     #这个变量是用来存储分割后的句子
        tokens_so_far = 0   #这个变量是用来存储当前句子的token的长度，类型是int
        chunk = []      #这个变量是用来存储当前句子的，哪里知道chunk是用来存储当前句子的呢，因为在下面的for循环中，每次循环都会把当前句子加入到chunk中，然后把当前句子的token的长度加上tokens_so_far的长度赋值给tokens_so_far，所以chunk是用来存储当前句子的
        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):
            # zip()函数是python的内建函数，用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
            # 如果当前句子的token的长度加上tokens_so_far的长度大于max_tokens，那么就把当前的chunk加入到chunks中，然后把chunk清空，然后把tokens_so_far清空，然后把当前句子加入到chunk中，然后把当前句子的token的长度加上tokens_so_far的长度赋值给tokens_so_far
            if tokens_so_far + token > max_tokens:#
                chunks.append(". ".join(chunk) + ".")  #join()方法用于将序列中的元素以指定的字符连接生成一个新的字符串。这里是把chunk中的句子用“.”连接起来，然后加上“.”，然后把这个字符串加入到chunks中，打印出来就是一个完整的句子，举例来说，假如chunk中有两个句子，那么就是“句子1.句子2.”，然后把这个字符串加入到chunks中
                chunk = []
                tokens_so_far = 0
            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence) #把当前句子加入到chunk中，原来是这里定义的chunk是用来存储当前句子的

            tokens_so_far += token + 1 # 把当前句子的token的长度加上tokens_so_far的长度赋值给tokens_so_far，为何要加1呢，因为在上面的if tokens_so_far + token > max_tokens:中，tokens_so_far += token + 1，所以这里也要加1，这样才能保证tokens_so_far的值是正确的
            '''
            这段循环循环语句的执行过程 首先确定必定执行的是chunk.append(sentence)，因为这个是在for循环中的，所以必定执行，因此chunk中一定会有句子
            然后判断tokens_so_far + token > max_tokens，如果为真，那么就执行chunks.append(". ".join(chunk) + ".")，然后把chunk清空，然后把tokens_so_far清空，然后把当前句子加入到chunk中，然后把当前句子的token的长度加上tokens_so_far的长度赋值给tokens_so_far
            如果为假，那么就执行tokens_so_far += token + 1，然后把当前句子加入到chunk中，然后把当前句子的token的长度加上tokens_so_far的长度赋值给tokens_so_far
            token > max_tokens:这个判断语句是用来判断当前句子的token的长度是否大于max_tokens，如果大于，那么就跳过当前句子，然后执行tokens_so_far += token + 1，然后把当前句子加入到chunk中，然后把当前句子的token的长度加上tokens_so_far的长度赋值给tokens_so_far
            '''
            # print("chunks:",chunks)
        return chunks


    shortened = []  # 这个变量和chunks的区别是什么呢，chunks是用来存储分割后的句子，而shortened是用来存储分割后的句子和原来的句子，
                    # 举例来说，假如原来的句子是“我是中国人”，那么chunks中就只有“我是中国人”，而shortened中就有“我是中国人”，因为chunks是用来存储分割后的句子，而shortened是用来存储分割后的句子和原来的句子

    # Loop through the dataframe
    # print("row[1]df每一行的迭代数据")
    for row in df.iterrows():
        # print(row[1])
        # If the text is None, go to the next row
        # 为何要从1开始呢，因为iterrows()返回的是一个元组，第一个元素是索引，第二个元素是一个Series，所以要从1开始
        # Series对象是一种类似于一维数组的对象，它由一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引）组成。数组就是list，而Series就是dict，Series的索引就是dict的key
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        # 如果当前句子的token的长度大于max_tokens，那么就把当前句子分割成多个句子，然后把分割后的句子加入到shortened中
        if row[1]['n_tokens'] > max_tokens:
            # list和list相加，就是把两个list合并成一个list
            shortened += split_into_many(row[1]['text'])
            # shortened += split_into_many(row[1]['text'])这句话的意思是把split_into_many(row[1]['text'])返回的list加入到shortened中

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()
    print("使用max——tokens分割之后的df数据")
    print(df.head())
    # 制作直方图
    '''# 倒入一下plt
    from matplotlib import pyplot as plt
    # 绘制直方图
    plt.hist(df['n_tokens'])
    plt.xlabel('Number of Tokens')
    plt.ylabel('Row_numers')
    plt.title('Distribution of Number of Tokens per Row')
    
    # 显示图表
    plt.show()
    '''

    # 这句话的意思是把df['text']中的每一个元素都转换成list，然后把list转换成np.array，然后把np.array赋值给df['embeddings']
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

    df.to_csv('processed/embeddings.csv')
    df.head()



