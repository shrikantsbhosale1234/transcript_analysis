from fastapi import FastAPI
import uvicorn
from azure import client, detect_sentiment_emotion, call_summary_and_desposition, script_adherence_and_compliance, customer_confirmation
app = FastAPI()
@app.post("/analyze_conversation/")
async def analyze_conversation(sample_conversation: str):
    """
    Analyze the given conversation JSON string using various Azure OpenAI functions.
    """
    return {
        "call_summary_and_disposition": call_summary_and_desposition(sample_conversation),
        "sentiment_emotion": detect_sentiment_emotion(sample_conversation),
        "script_adherence_and_compliance": script_adherence_and_compliance(sample_conversation),
        "customer_confirmation": customer_confirmation(sample_conversation)
    }
if __name__ == "__main__":
    #sample_conversation = "{\n  \"language\": \"en\",\n  \"agent_utterances\": [\n    \"Do you own a New Holland tractor?\",\n    \"How long has it been since your tractor was serviced?\",\n    \"Would you recommend New Holland tractors to your friends or relatives?\",\n    \"How was your overall experience with the New Holland brand?\",\n    \"On a scale of 0 to 10, how would you rate the New Holland brand?\",\n    \"What did you like most about New Holland?\",\n    \"How was your experience with the dealer (communication and work style)?\",\n    \"On a scale of 0 to 10, how would you rate the dealer/service agency?\",\n    \"Did you face any difficulty while booking the service?\",\n    \"On a scale of 0 to 10, how would you rate the service booking process?\",\n    \"Was your problem resolved in a single interaction, or did you have to explain it multiple times?\",\n    \"How was the mechanic’s behavior and communication during service?\",\n    \"On a scale of 0 to 10, how would you rate the mechanic?\",\n    \"After service, how was the tractor’s performance and service quality?\",\n    \"On a scale of 0 to 10, how would you rate the service quality?\",\n    \"Was the service completed within the promised time?\",\n    \"On a scale of 0 to 10, how would you rate service timeliness?\",\n    \"Are you satisfied with the service cost and value for money?\",\n    \"On a scale of 0 to 10, how would you rate service cost/value?\",\n    \"How was your tractor handover experience after service completion?\",\n    \"On a scale of 0 to 10, how would you rate the handover experience?\",\n    \"Did the agency call you after service completion to confirm the tractor was working properly?\",\n    \"Are you satisfied with your current New Holland dealer/agency?\",\n    \"Do you know anyone planning to buy a New Holland tractor?\"\n  ],\n  \"customer_utterances\": [\n    \"Yes\",\n    \"About 25 days ago (approximately one month)\",\n    \"Yes\",\n    \"Very good / Excellent\",\n    \"9 / 10\",\n    \"The brand itself\",\n    \"Very good\",\n    \"9 / 10\",\n    \"No\",\n    \"9 / 10\",\n    \"Resolved in a single interaction\",\n    \"Very good\",\n    \"9 / 10\",\n    \"Very good\",\n    \"9 / 10\",\n    \"Yes\",\n    \"10 / 10\",\n    \"Yes\",\n    \"10 / 10\",\n    \"Good\",\n    \"8 / 10\",\n    \"Yes\",\n    \"Yes\",\n    \"No, not at the moment\"\n  ]\n}"
    # Load or define your sample conversation JSON string here
    # Example: sample_conversation = '{"conversation": [...]}'
    uvicorn.run(app, host="127.0.0.1", port=10000)

