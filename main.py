import json
from context_agent import context_agent
from hallucination_agent import HallucinationDetector

def process_user_query(user_query: str, text_output: dict) -> dict:
    """
    Main function to process user query with previous team's output
    
    Args:
        user_query: The original user query
        previous_team_output: Output from the previous team
    
    Returns:
        Processed results with validation checks
    """
    detector = HallucinationDetector()
    return context_agent(user_query, text_output, detector)

if __name__ == "__main__":
    # Example usage - this would come from previous teams in real scenario
    user_query = "What's the dosage of Humira?"
    
    # This is the output from the previous team that you'll receive as input
    text_output = {
        "intents": ["dosage"],
        "entities": ["Humira"],
        "raw_query": "What's the dosage of Humira?",
        "curated_retrievals": [
            {
                "text": "Humira recommended dosage is 40mg every other week, subcutaneous injection.",
                "citation": "humira.pdf (Section: Dosing, Page: 35)",
                "score": 0.496084011663871
            }
        ]
    }

    # Process the inputs
    result = process_user_query(user_query, text_output)
    
    # Output the results
    print(json.dumps(result, indent=2))