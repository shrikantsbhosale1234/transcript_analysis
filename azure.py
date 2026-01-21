import os
from openai import AzureOpenAI, azure_endpoint, api_key

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("azure_endpoint"),
    api_key=os.getenv("api_key")
)

def detect_sentiment_emotion(conversation_json: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a call center sentiment and emotion detection engine.\nAnalyze ONLY the customer utterances.\n\nYou MUST return all fields in the output schema.\n\nSentiment must be exactly one of: positive, neutral, negative, mixed.\n\nEmotions must be selected ONLY from this list:\nsatisfaction, happiness, trust, frustration, confusion, anger, disappointment, relief.\n\nConfidence must be exactly one of: high, medium, low.\n\nDo not add, rename, omit, or move any fields.\n\nReturn ONLY raw JSON in the specified schema.",
            },
            {
                "role": "user",
                "content": f"{conversation_json}",
            }
        ],
        max_completion_tokens=16384,
        model="gpt-5-nano",

    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

def call_summary_and_desposition (conversation_json: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a call center post-call analysis engine. Your responsibility is to generate a concise, factual call summary in two to four sentences and determine the most appropriate call disposition strictly from a provided list, using only the given agent and customer utterances. You must not invent information, make assumptions, or add details that are not explicitly present in the input, and you should avoid performing sentiment analysis except where it is strictly required to select the correct disposition. Your response must be returned as raw JSON only, without any markdown, code blocks, or additional explanatory text, and it must follow the exact output structure containing the fields call_summary, call_disposition, and confidence"
            },
            {
                "role": "user",
                "content": f"{conversation_json}",
            }
        ],
        max_completion_tokens=16384,
        model="gpt-5-nano",

    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

def script_adherence_and_compliance (conversation_json: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an integrated quality audit and preprocessing engine. Execute this complete workflow:\n\n(1) PREPROCESSING PHASE:\n- Treat all questions as agent utterances and all answers as customer utterances\n- Detect primary language using ISO codes (en, hi, hi-en, ta-en, etc.)\n- Lightly clean filler words without summarizing or inferring meaning\n- Do not analyze sentiment or content meaning beyond what is explicitly present\n\n(2) SCRIPT ADHERENCE PHASE:\n- Verify whether each mandatory question was asked by analyzing only agent utterances\n- Consider a question as asked if intent or semantic meaning matches even if wording differs\n- Support language variations (English, Hindi, Hinglish, mixed language)\n- Do not assume compliance unless intent is clearly present\n- For each mandatory question: determine if asked, identify closest matching agent utterance if yes, or provide actionable coaching tip if no\n\n(3) MANDATORY DISCLOSURES PHASE:\n- Verify all mandatory disclosures were made by analyzing agent utterances\n- Detect semantic intent of each disclosure statement\n- Support language variations and paraphrasing\n- Flag disclosures as made only if intent is clearly present\n- For each disclosure: identify matching utterance if made or provide coaching tip if not\n\n(4) FORBIDDEN PHRASES PHASE:\n- Detect zero-tolerance forbidden phrases in agent utterances\n- Identify exact phrase instances with line references and severity levels\n- Severity levels: Critical (compliance/legal violations), High (brand/policy violations), Medium (guideline violations)\n- Count all instances and flag detected utterances\n\n(5) SCORING & FINAL AUDIT PHASE:\n- Calculate compliance scores as percentage of compliant items rounded to nearest whole number\n- Script Adherence Score: percentage of mandatory questions asked\n- Disclosures Score: percentage of mandatory disclosures made\n- Forbidden Phrases Score: 100 if zero violations detected, 0 if any violations found\n- Overall Compliance Status: Set to \"Compliant\" ONLY if ALL three dimensions are Compliant (all questions asked AND all disclosures made AND no forbidden phrases detected), otherwise \"Non-Compliant\"\n- Overall Compliance Percentage: Average of all three scores (Script + Disclosures + Phrases) / 3\n- Provide comprehensive audit notes summarizing key findings, violations, and recommendations\n\nOUTPUT REQUIREMENTS:\n- Return ONLY valid raw JSON\n- No markdown formatting, code blocks, or backticks\n- Strictly follow the output schema without adding, renaming, omitting, or reordering fields\n- Include no explanations or extra text outside JSON\n\nREQUIRED MANDATORY QUESTIONS (verify each):\n1. PurchaseExperience: On a scale of 0 to 10, how would you rate the New Holland brand?\n2. RepairServiceExperience: On a scale of 0 to 10, how would you rate the dealer/service agency?\n3. OwnUseExperience: On a scale of 0 to 10, how would you rate the service quality?\n\nREQUIRED MANDATORY DISCLOSURES (verify each):\n1. Call Recording: Call is being recorded for quality and training purposes\n2. Data Privacy: Data will be used in compliance with privacy policies\n3. Service Terms: This is not a warranty or service guarantee\n\nFORBIDDEN PHRASES (detect all instances):\n1. guaranteed results (Severity: Critical)\n2. You have to buy (Severity: High)\n3. 100% satisfaction (Severity: Medium)\n4. risk-free (Severity: Medium)\n\nOUTPUT JSON SCHEMA:\n{\n  \"preprocessing\": {\n    \"language\": \"string (ISO code)\",\n    \"agent_utterances\": [\"string\"],\n    \"customer_utterances\": [\"string\"]\n  },\n  \"script_adherence\": {\n    \"mandatory_questions\": [\n      {\n        \"question_id\": \"string\",\n        \"category\": \"string\",\n        \"question_text\": \"string\",\n        \"asked\": \"boolean\",\n        \"matching_utterance\": \"string or null\",\n        \"coaching_tip\": \"string or null\",\n        \"status\": \"Compliant or Non-Compliant\"\n      }\n    ],\n    \"compliance_score\": \"integer (0-100)\",\n    \"status\": \"Compliant or Non-Compliant\"\n  },\n  \"mandatory_disclosures\": {\n    \"disclosures\": [\n      {\n        \"disclosure_id\": \"string\",\n        \"category\": \"string\",\n        \"disclosure_text\": \"string\",\n        \"made\": \"boolean\",\n        \"matching_utterance\": \"string or null\",\n        \"coaching_tip\": \"string or null\",\n        \"status\": \"Compliant or Non-Compliant\"\n      }\n    ],\n    \"compliance_score\": \"integer (0-100)\",\n    \"status\": \"Compliant or Non-Compliant\"\n  },\n  \"forbidden_phrases\": {\n    \"phrases\": [\n      {\n        \"phrase_id\": \"string\",\n        \"phrase_text\": \"string\",\n        \"severity\": \"Critical or High or Medium\",\n        \"detected\": \"boolean\",\n        \"instances\": \"integer\",\n        \"detected_utterances\": [\n          {\n            \"utterance\": \"string\",\n            \"line_number\": \"integer\",\n            \"exact_phrase_used\": \"string\"\n          }\n        ],\n        \"status\": \"Compliant or Non-Compliant\"\n      }\n    ],\n    \"critical_violations\": \"integer\",\n    \"high_violations\": \"integer\",\n    \"compliance_score\": \"integer (0-100)\",\n    \"status\": \"Compliant or Non-Compliant\"\n  },\n  \"overall_audit_summary\": {\n    \"audit_timestamp\": \"string (ISO 8601)\",\n    \"total_utterances_analyzed\": \"integer\",\n    \"script_adherence_score\": \"integer (0-100)\",\n    \"disclosures_score\": \"integer (0-100)\",\n    \"forbidden_phrases_score\": \"integer (0-100)\",\n    \"overall_compliance_status\": \"Compliant or Non-Compliant\",\n    \"overall_compliance_percentage\": \"integer (0-100)\",\n    \"audit_notes\": \"string\"\n  }\n}"
            },
            {
                "role": "user",
                "content": f"{conversation_json}",
            }
        ],
        max_completion_tokens=16384,
        model="gpt-5-nano",

    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content
def customer_confirmation (conversation_json: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a TVR (Tele-Verification of Records) data extraction and validation engine. Your task is to extract and validate customer confirmations of loan and financial parameters from call transcripts.\n\nEXTRACTION RULES:\n- Extract loan-related confirmations DIRECTLY from customer utterances ONLY\n- Ignore agent statements unless the customer explicitly repeats or confirms them\n- Capture EXACT values as stated by customer: do not infer, calculate, or transform values\n- Extract customer values exactly as spoken (e.g., \"ten thousands per month\", \"three years total\", \"February 15th\")\n- Support language variations (English, Hindi, Hinglish, mixed language)\n- Do not assume confirmation unless explicit acknowledgment is present\n- A customer confirmation is valid only when customer directly states, agrees, affirms, or repeats the parameter\n\nCONFIRMATION STATUS DEFINITIONS:\n- Confirmed: Customer explicitly confirms, repeats, or affirms the parameter with clear acknowledgment (e.g., \"Yes\", \"Correct\", \"That's right\", \"Agreed\", \"15000 per month\")\n- Partial: Customer confirms some aspects but disputes or provides alternative information for other aspects; incomplete or ambiguous confirmation\n- Not Confirmed: Customer denies, contradicts, or provides significantly different value; customer remains silent on parameter; parameter not discussed\n\nVALIDATION PROCESS:\n1. Identify all customer utterances related to loan parameters (loan amount, monthly EMI, tenure, repayment date, interest rate, loan purpose, etc.)\n2. Extract exact customer values as stated, preserving language and phrasing variations\n3. Compare system values with customer confirmed values\n4. Determine confirmation status based on explicit customer acknowledgment\n5. Identify closest matching utterance that supports the confirmation status\n6. Flag any discrepancies between system values and customer confirmed values\n7. Flag any missing confirmations (parameters not discussed)\n\nLOAN PARAMETERS TO EXTRACT (if present in transcript):\n- Loan Amount: Principal loan amount in currency\n- Monthly EMI (Equated Monthly Installment): Monthly payment amount\n- Loan Tenure: Loan duration (months or years)\n- Repayment Date: First payment date or subsequent repayment dates\n- Interest Rate: Applicable interest rate percentage\n- Loan Purpose: Purpose of loan (e.g., tractor purchase, equipment)\n- Outstanding Balance: Remaining loan balance\n- Prepayment Terms: Early repayment options or penalties\n- Grace Period: Any moratorium or grace period details\n\nDISCREPANCY FLAGS:\n- Value Mismatch: Customer value significantly differs from system value\n- Language Variation: Customer expresses value in different format (e.g., \"ten thousands\" vs \"10000\")\n- Ambiguity: Customer statement is unclear or contains conditional language\n- Missing Confirmation: Parameter not discussed or confirmed\n- Contradictory Statement: Customer provides conflicting information\n\nOUTPUT REQUIREMENTS:\n- Return ONLY valid raw JSON\n- No markdown formatting, code blocks, or backticks\n- Strictly follow the output schema without adding, renaming, omitting, or reordering fields\n- Include no explanations or extra text outside JSON\n\nOUTPUT JSON SCHEMA:\n{\n  \"tvr_metadata\": {\n    \"call_type\": \"string (TVR call identifier)\",\n    \"extraction_timestamp\": \"string (ISO 8601 format)\",\n    \"total_parameters_in_system\": \"integer (count of system parameters)\",\n    \"total_parameters_discussed\": \"integer (count of customer utterances related to parameters)\"\n  },\n  \"customer_confirmations\": [\n    {\n      \"parameter_id\": \"string (unique identifier)\",\n      \"parameter_name\": \"string (Loan Amount, Monthly EMI, Loan Tenure, Repayment Date, Interest Rate, etc.)\",\n      \"parameter_category\": \"string (Financial, Schedule, Terms, Other)\",\n      \"system_value\": \"string or null (value from system records)\",\n      \"system_value_unit\": \"string (Currency, Months, Years, Date Format, Percentage, etc.)\",\n      \"customer_confirmed_value\": \"string (exact value as stated by customer)\",\n      \"customer_value_unit\": \"string (as expressed by customer)\",\n      \"matching_utterance\": \"string (exact customer utterance supporting confirmation)\",\n      \"confirmation_status\": \"Confirmed or Partial or Not Confirmed\",\n      \"discrepancy_flag\": \"string or null (Value Mismatch, Language Variation, Ambiguity, Missing Confirmation, Contradictory Statement, null if no discrepancy)\",\n      \"discrepancy_description\": \"string or null (detailed explanation if discrepancy exists)\",\n      \"language_variant\": \"string or null (en, hi, hi-en, mixed, etc. if customer used language variation)\"\n    }\n  ],\n  \"confirmation_summary\": {\n    \"total_parameters_extracted\": \"integer (total loan parameters discussed)\",\n    \"confirmed_count\": \"integer (parameters with Confirmed status)\",\n    \"partial_count\": \"integer (parameters with Partial status)\",\n    \"not_confirmed_count\": \"integer (parameters with Not Confirmed status)\",\n    \"confirmation_rate_percentage\": \"integer (0-100, confirmed / total * 100, rounded)\",\n    \"discrepancies_found\": \"integer (count of parameters with discrepancy flags)\",\n    \"critical_discrepancies\": [\n      {\n        \"parameter_name\": \"string\",\n        \"system_value\": \"string\",\n        \"customer_value\": \"string\",\n        \"discrepancy_type\": \"string\"\n      }\n    ]\n  },\n  \"overall_tvr_status\": {\n    \"verification_status\": \"Fully Verified or Partially Verified or Not Verified\",\n    \"verification_notes\": \"string (summary of verification results, discrepancies, and action items)\",\n    \"requires_escalation\": \"boolean (true if critical discrepancies found)\",\n    \"escalation_reason\": \"string or null (reason for escalation if applicable)\",\n    \"recommended_actions\": [\"string (list of follow-up actions, clarifications, or corrections needed)\"]\n  }\n}"
            },
            {
                "role": "user",
                "content": f"{conversation_json}",
            }
        ],
        max_completion_tokens=16384,
        model="gpt-5-nano",

    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

# if __name__ == "__main__":    
#     call_summary_and_desposition(sample_conversation)
#     detect_sentiment_emotion(sample_conversation)
#     script_adherence_and_compliance(sample_conversation)
#     customer_confirmation(sample_conversation)
