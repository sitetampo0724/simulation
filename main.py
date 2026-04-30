# 模型仅限校内网络调用，数据不出校.
# Powered by hpc@sjtu.edu.cn, 2026.03.24
from query import query
from graph import photograph


if __name__ == "__main__":

    print("words or photo")
    choices = input("please input:")
    if(choices == "words"):
        question = ""
        question = input("hello, what you want to know about")
        print(f"问题: {question}")
        print("\n--- 模型答案（逐字输出）---")
        aianswer = query()
        aianswer.query_stream(question)
        print("\n--- 输出完成 ---")

    elif(choices == "photo"):
        photoai = photograph("gaoyunzhen.jpg")
        photoai.recognize_image_flexible()






