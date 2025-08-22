from typing import Dict, List
from hallucination_agent import HallucinationDetector
import google.generativeai as genai

# ---------------- CONFIGURE GEMINI ----------------
genai.configure(api_key="AIzaSyDG0qdzmCJ7N7xRB--ZrQTRp8oraWiAS-w")
model = genai.GenerativeModel("gemini-2.5-flash")

def hallucination_check(user_query: str, retrievals: List[Dict], detector: HallucinationDetector) -> Dict:
    if not retrievals:
        return {"status": "failed", "text": user_query}
    # Use friend's hallucination detection
    return detector.detect_hallucination(
        retrievals[0]["text"], 
        retrievals[0].get("citation", ""), 
        retrievals[0].get("score", 0.0)
    )

def relation_validation(user_query: str, entities: List[str], retrievals: List[Dict]) -> Dict:
    if not retrievals or retrievals[0]["text"].strip() == "":
        return {"status": "invalid", "attempts": 1, "validated_text": "No retrieval found."}

    retrieved_text = retrievals[0]["text"]

    # Step 1: Medical check
    medical_prompt = f"""
    Determine if this statement is medically related:
    Query: {user_query}
    Retrieval: {retrieved_text}
    Reply ONLY "MEDICAL" or "NOT MEDICAL".
    """
    medical_response = model.generate_content(medical_prompt).text.strip()

    if medical_response != "MEDICAL":
        return {"status": "invalid", "attempts": 1, "validated_text": "Text is not medically related."}

    # Step 2: Contradiction check
    contradiction_prompt = f"""
    You are a medical contradiction detector.
    Compare the user query and retrieval result.

    User Query: "{user_query}"
    Retrieval Result: "{retrieved_text}"

    If they contradict each other, reply ONLY "CONTRADICTION".
    If they are consistent, reply ONLY "CONSISTENT".
    """
    contradiction_response = model.generate_content(contradiction_prompt).text.strip()

    if contradiction_response == "CONSISTENT":
        return {"status": "valid", "attempts": 1, "validated_text": retrieved_text}
    else:
        return {"status": "invalid", "attempts": 2, "validated_text": "Retrieved text contradicts the query."}

def domain_reasoning(user_query: str, retrievals: List[Dict], relation_result: Dict) -> Dict:
    if relation_result["status"] != "valid":
        return {"status": "NON_MEDICAL", "reason": f"Query not medically validated: {relation_result['validated_text']}"}

    text_to_check = user_query + " " + (retrievals[0]["text"] if retrievals else "")
    prompt = f"""
    Determine if the following is medically related:
    Text: "{text_to_check}"
    Reply ONLY "MEDICAL" or "NON_MEDICAL" and give a short reason.
    """
    response = model.generate_content(prompt).text.strip()
    if "MEDICAL" in response.upper():
        return {"status": "MEDICAL", "reason": response}
    else:
        return {"status": "NON_MEDICAL", "reason": response}

def context_agent(user_query: str, text_output: Dict, detector: HallucinationDetector) -> Dict:
    """
    Main context agent function that processes inputs from previous teams
    
    Args:
        user_query: The original user query string
        previous_team_output: Output from the previous team (should contain curated_retrievals and entities)
        detector: HallucinationDetector instance
    
    Returns:
        Dict containing validation results and next steps
    """
    retrievals = text_output.get("curated_retrievals", [])
    entities = text_output.get("entities", [])

    hallucination_result = hallucination_check(user_query, retrievals, detector)
    relation_result = relation_validation(user_query, entities, retrievals)
    domain_result = domain_reasoning(user_query, retrievals, relation_result)

    final_output = {
        "query": user_query,
        "previous_team_output": text_output,
        "hallucination_check": hallucination_result,
        "relation_validation": relation_result,
        "domain_reasoning": domain_result,
        "next_step": "Final Response" if domain_result["status"] == "MEDICAL" else "Flag for Review"
    }
    return final_output