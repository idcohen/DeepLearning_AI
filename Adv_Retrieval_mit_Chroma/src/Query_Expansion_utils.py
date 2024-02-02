


###############################################################################################
def augment_query_generated(openai_client, query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report. "
        },
        {"role": "user", "content": query}
    ] 

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

###############################################################################################
def Query_Expansion_Multiple_Queries(openai_client, original_query, model="gpt-3.5-turbo", verbose=False):
    hypothetical_answer = augment_query_generated(openai_client, original_query, model)
    joint_query = f"{original_query} {hypothetical_answer}"
    if verbose:
            print(word_wrap(joint_query))
    return joint_query

###############################################################################################
def augment_multiple_query(openai_client, query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about an annual report. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

###############################################################################################
def Query_Expansion_Multiple_Queries(openai_client, original_query, model="gpt-3.5-turbo", verbose=False):
    augmented_queries = augment_multiple_query(openai_client, original_query, model)
    queries = [original_query] + augmented_queries
    if verbose:
            print(word_wrap(queries))
    return queries
    
