#import transformers
#import torch
#from transformers import BitsAndBytesConfig
#from transformers import AutoModelForCausalLM, AutoTokenizer

import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, AIMessage


from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class CustomLlama3_405B(DeepEvalBaseLLM):
    def __init__(self):
        os.environ["NVIDIA_API_KEY"] = "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"
        model = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
        
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        return model.invoke(prompt).content#[HumanMessage(content=prompt)])

    async def a_generate(self, prompt: str) -> str:
        #return self.generate(prompt)
        model = self.load_model()
        res = await model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "llama-3.1-405b-instruct"
    

custom_llm = CustomLlama3_405B()

question = "Which is the largest country by population?"
testcase = LLMTestCase(
    input=question,
    actual_output=custom_llm.generate(question),
    expected_output="What the hell is happening?",
    context=["Just give the name of one country as an answer - do not include any other context in your response"]
)

relevancy_metric = AnswerRelevancyMetric(
    model=custom_llm,
    threshold=0.7
)
#relevancy_metric.measure(testcase)
#print(relevancy_metric.score, relevancy_metric.reason)

correctness_metric = GEval(
    model=custom_llm,
    name="Correctness",
    criteria="correctness - determine if the actual output exactly matches the expected output",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    strict_mode=True
)
#correctness_metric.measure(testcase)
#print(correctness_metric.score, correctness_metric.reason)


#run_async is False, so that the metrics scores can be used later in the code
evaluate([testcase], [relevancy_metric, correctness_metric], run_async=False) 
print("==> ")
print(relevancy_metric.score, relevancy_metric.reason)
print(correctness_metric.score, correctness_metric.reason)
