import glob

prompt_templates = {
        "mistral-7b-instruct_rp": '''[INST] You are provided with a text of some essays, admist these essays is a sentence
that contains the answer to the user's question. I will now provide the text (delimited with XML tags) followed by the user question. 
            
[TEXT]
{content}
[/TEXT]


User: {prompt}[/INST]
        
Here is the most relevant sentence in the text:''',

    "mistral-7b-instruct": '''[INST] You are provided with a text of some essays, admist these essays is a sentence
that contains the answer to the user's question. I will now provide the text (delimited with XML tags) followed by the user question. 
            
[TEXT]
{content}
[/TEXT]


User: {prompt}[/INST]''',
    "llama-7b": "[INST]\nYou are provided with a text of some essays, admist these essays is a sentence that contains the answer to the user's question."
    + " I will now provide the text (delimited with XML tags) followed by the user question.\n\n[TEXT]\n{content}\n[/TEXT]\n\n" + "{prompt}\n[/INST]\n\n",
}   


def read_files(directory):
    context = ""
    for file in glob.glob(directory):
        with open(file, 'r', encoding="utf-8") as f:
            context += f.read()
    return context


def encode_and_trim(tokens_context, context_length):
    if len(tokens_context) > context_length:
        tokens_context = tokens_context[:context_length]
    return tokens_context


def insert_needle(tokens_needle, tokens_context, depth_percent, context_length, tokenizer):
    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= 180

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = tokenizer.encode('\n')
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    return tokens_new_context


def generate_context(tokenizer, needle, context, context_length, depth_percent):
    # Tokenize context and needle
    tokens_needle = tokenizer.encode(needle, add_special_tokens=False)
    tokens_context = tokenizer.encode(context, add_special_tokens=False)

    # Truncate the Paul Graham essays to the context length you desire
    tokens_context = encode_and_trim(tokens_context, context_length)

    # Insert your random statement according to your depth percent
    tokens_context = insert_needle(tokens_needle, tokens_context, depth_percent,
                            context_length, tokenizer)

    return tokens_context


def result_exists(results, context_length, depth_percent, version, model):
    """
    Checks to see if a result has already been evaluated or not
    """
    conditions_met = []
    for result in results:
        context_length_met = result['context_length'] == context_length
        depth_percent_met = result['depth_percent'] == depth_percent
        version_met = result.get('version', 1) == version
        model_met = result['model'] == model
        conditions_met.append(context_length_met and depth_percent_met and version_met)
    return any(conditions_met)