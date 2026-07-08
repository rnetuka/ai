from llm import Mistral


if __name__ == '__main__':
    llm = Mistral()
    answer = llm.invoke('Hi')
    print(answer)
