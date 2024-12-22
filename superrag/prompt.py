GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

# Default language and delimiters
PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Process tickers and entity types
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

# Entity extraction prompt template
PROMPTS["entity_extraction"] = """
- Goal -
Given a text document potentially relevant to this activity and a list of entity types, identify all entities of those types and all relationships between them.
Use {language} for the output language.

- Steps -
1. Identify all entities. For each entity, extract the following information:
- entity_name: Name of the entity, using the language of the input text (capitalize if in English).
- entity_type: One of the following types: [{entity_types}]
- entity_description: A detailed description of the entity's attributes and activities.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in Step 1, identify all pairs of (source_entity, target_entity) that are *clearly related*.
   For each pair, extract the following information:
   - source_entity: Name of the source entity, as identified in Step 1.
   - target_entity: Name of the target entity, as identified in Step 1.
   - relationship_description: Explanation of the relationship between the source and target entities.
   - relationship_strength: A numeric score indicating the strength of the relationship.
   - relationship_keywords: Key terms that summarize the nature of the relationship (concepts/themes).
   Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words summarizing the main ideas or themes of the entire text.
   Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return the output in {language} as a single list of all entities and relationships identified in Steps 1 and 2, using **{record_delimiter}** as the delimiter.

5. When finished, output {completion_delimiter}

######################
- Examples -
######################
{examples}

#############################
- Real Data -
#############################
Entity_types: {entity_types}
Text: {input_text}
#############################
Output:
"""

# Entity extraction examples
PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and observes the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################""",
]

# Continue extraction of missing entities prompt
PROMPTS["entiti_continue_extraction"] = """Many entities were missed in the last extraction. Add them below using the same format:"""

# If there's a loop in entity extraction
PROMPTS["entiti_if_loop_extraction"] = """It appears some entities may have still been missed. Answer YES | NO if there are still entities that need to be added.""" 

# Fail response for unanswerable queries
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

# RAG response to summarize data from context tables
PROMPTS["rag_response"] = """--- Role ---

You are a helpful ai assistant responding to questions about data in the tables provided.

--- Goal ---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables, incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

--- Target response length and format ---

{response_type}

--- Data tables ---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

# Keywords extraction prompt
PROMPTS["keywords_extraction"] = """--- Role ---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.
Use {language} as output language.

--- Goal ---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

--- Instructions ---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
- Examples -
######################
{examples}

#############################
- Real Data -
#############################
Query: {query}
#############################
Output:
"""

# Naive RAG response for document-based questions
PROMPTS["naive_rag_response"] = """--- Role ---

You are a helpful assistant responding to questions about documents provided.

--- Goal ---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables, incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

--- Target response length and format ---

{response_type}

--- Documents ---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""
