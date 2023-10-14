from cove import chain

if __name__ == "__main__":
    response = chain.invoke({"original_question": "Who are some politicians born in Boston?"})
    print(response)