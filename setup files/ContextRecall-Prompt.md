Ask Mode:
Scan my project files tell me the files which are impacted with
implementation of metrics -Faithfulness and Answer relevancy
--Tell me the files where code impementation is done
--Also examples are added

Ask Mode:
Now i want to add the new metrics context_recall from deepeval
evaluation site is After Retrieval  
which is for retrieved context (retrieval_context) and expected output(expected_output) should be added in request parameters
use the same procedure for calling function like faithfulness and answer relevancy 
[STRICTLY]impacted api is -api/eval-only for implementation
[MANDATORY] Include all reasons strict mode false
[USE]https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/contextual_recall/contextual_recall.py FOR referenence
so my api request should be the below as an example ,for which 
{
  "metric": "contextual_recall",
  "query": "Salesforce login troubleshooting Steps",
  "expected_output": "Steps to resolve Salesforce login issues: verify username, reset password, check SSO/SAML, network/allowlist, lockout, MFA.",
  "retrieval_context": [
    "Salesforce login error codes and fixes (invalid username/password, lockout, SSO).",
    "Admin guide: Resetting user passwords and unlocking users in Salesforce.",
    "Troubleshooting MFA login failures for Salesforce.",
    "Network & allowlist: Salesforce trust domains, firewall/proxy, TLS/cipher requirements."
  ],
  "output": "N/A"
}