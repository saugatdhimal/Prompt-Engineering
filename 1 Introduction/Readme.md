
# Welcome to this course on ChatGPT Prompt Engineering for Developers.
 
- A lot of that has been focused on the chatGPT web user interface, which many people are using to do specific and often one-off tasks. 

- But, the power of LLMs, large language models, as a developer tool, that is using API calls to LLMs to quickly build software applications is still very underappreciated.

- In this course, we'll share with you some of the possibilities for what you can do, as well as best practices for how you can do them. 

- First, you'll learn some prompting best practices for software development, then we'll cover some common use cases, summarizing, inferring, transforming, expanding, and then you'll build a chatbot using an LLM. We hope that this will spark your imagination about new applications that you can build. 

- In the development of large language models or LLMs, there have been broadly two types of LLMs, which I'm going to refer to as base LLMs and instruction-tuned LLMs. 

- Base LLM has been trained to predict the next word based on text training data, often trained on a large amount of data from the internet and other sources to figure out what's the next most likely word to follow. So, for example, if you were to prompt us once upon a time there was a unicorn, it may complete this, that is it may predict the next several words are that live in a magical forest with all unicorn friends. But if you were to prompt us with what is the capital of France, then based on what articles on the internet might have, it's quite possible that the base LLM will complete this with what is France's largest city, what is France's population and so on, because articles on the internet could quite plausibly be lists of quiz questions about the country of France. 

- Instruction-tuned LLM, which is where a lot of momentum of LLM research and practice has been going, an instruction-tuned LLM has been trained to follow instructions. So, if you were to ask it what is the capital of France, it's much more likely to output something like, the capital of France is Paris. So the way that instruction-tuned LLMs are typically trained is you start off with a base LLM that's been trained on a huge amount of text data and further train it, further fine-tune it with inputs and outputs that are instructions and good attempts to follow those instructions, and then often further refine using a technique called RLHF, reinforcement learning from human feedback, to make the system better able to be helpful and follow instructions. 

- Because instruction-tuned LLMs have been trained to be helpful, honest, and harmless, so for example, they are less likely to output problematic text such as toxic outputs compared to base LLM, a lot of the practical usage scenarios have been shifting toward instruction-tuned LLMs.Some of the best practices you find on the internet may be more suited for a base LLM, but for most practical applications today, we would recommend most people instead focus on instruction-tuned LLMs which are easier to use and also, because of the work of OpenAI and other LLM companies becoming safer and more aligned. 