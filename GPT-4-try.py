from openai import OpenAI

client = OpenAI(
  api_key="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",
  #base_url = 'xxx'
)

def get_completion(prompt, model="gpt-4-0613"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content

prompt = "你是ChatGPT3.5还是4.0呢？"

print(get_completion(prompt))