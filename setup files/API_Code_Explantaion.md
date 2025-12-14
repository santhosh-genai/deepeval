
*DeepEval API Explanation*

1)src/config/env.ts
Groq/Open API Key for LLM Evaluation
If Key is not present ,throws the error
Switch from one model to another

2)src/routes/evalRoute.ts
API Calls to be initiated from postman are found here

a)llm/eval
1) Sending Request(query) to the LLM 
2) Response(output) from LLM 
3) Deepeval Compare the request and response gives scoring as output 
4) Response in postman gives output with scoring

b)rag/eval
1) Retrieve document along with the prompt send to LLM for evaluation
3) Deepeval Compare the request and response gives scoring as output 
4) Response in postman gives output with scoring

c)/health
checking the status of health

d)eval-only
1)We have the input and output in the request
2)Send Request to Deep Eval (not generating output from LLM)
3)Validate the response

e)/metrics
All metrics available

3)src/services/llmClient.ts
Connecting llm model via api call

4)src/services/ragService.ts
Connecting to the Rag database

Reads env-evalRoutes.ts (API)-evalClient.ts(evalWithFields)

evalClient.ts calls via axios Deepeval(deepeval_server.py)
LLMTestCase




*Below files will have effect while adding /removing metrics*

evalRoutes.ts
usage example
available_metrics

evalclient.ts
64-condition part to be added same as other two metrics

deepeval_server.py
SUPPORTED_METRICS
Metric_requirements
def evaluate_faithfulness(self, test_case) -> tuple[float, str]:-Function to be added for other metrics same like faithfulness(Line no 258)  
Multi_metric_support  
 Line no 673 other metrics to be added








































