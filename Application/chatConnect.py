from openai import OpenAI

class GPTChat:
    def __init__(self,first_input):
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key="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",
        )
        
        self.chat_prompt = {
            "role": "user",
            "content": first_input,
        }
        
        self.chat_completion = self.client.chat.completions.create(
            messages=[self.chat_prompt],
            model="gpt-3.5-turbo",
            max_tokens=150,
        )

    def get_reply(self, user_input):
        self.chat_prompt["content"] = user_input
        self.chat_completion = self.client.chat.completions.create(
            messages=[self.chat_prompt],
            model="gpt-3.5-turbo",
            max_tokens=150,
        )
        return self.chat_completion.choices[0].message.content

    
