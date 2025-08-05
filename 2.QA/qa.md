# Questions

Each question is answered directly below. References are provided when seem useful.

### 1. Differences between `completion` and `chat` models

In Langchain, completion models receive a prompt and output a completion. Is a 1 step process. On the other hand, chat models allow conversations and retain memory.
### 2. Differences exist between a `reasoning` and a `generalist` model.
Reasoning models are focused more on planning and complicated logic that require more than 1 step. They create chains of thought that guide them from step to step. Generalist models are more like "jack-of-all-trades" that can do many different tasks.
### 3. How to force a chatbot to answer `yes` or `no`.
This needs to be specified in the prompt given to the LLM. For critical tasks where Yes/No is COMPULSORY, consider another model checking the output of the first and proceed only when the second one verifies the output of the first.
### 4. RAG vs Fine-tuning
Fine-tuning modifies the model weights to learn how to respond. The data is INSIDE the model itself after fine-tuning. On ther other hand RAG adds knowledge at runtime with external docs, chunks of docs etc. The data is external and retrieved in each query.
### 5. What is an agent?
An assistant that receives an input and is able to answer questions, take actions calling APIs or other tools available through MCP, etc, decide what next steps to take, interact with other agents, etc.
### 6. How to evaluate the performance of a Q&A bot?
Evaluate the following:
- Relevance of retrieved documents
- Composition of output text from provided documentation
- Sensitivity of output vs input 
- Hallucinations and compression spike detection ([see reference](https://arxiv.org/pdf/2507.11768))
- f1 score to compare token overlap (predicted vs ground truth)
- BLEU / Rouge metric measuring n-gram overlap between generated and reference answer (I had to look this up, didnt know it before)
- qualitative evaluation by users (surveys)
- logging questions and answers 
-usual metrics for accuracy fallback rate and unaswered rate
- Consider also using the RAGAS library ot evaluate RAG pipelines (I had to also look this up)