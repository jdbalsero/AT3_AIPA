# class rag_assistant #
# atributes: model parameteres (model, temperature, max completition tokens, system_config)
# methods:
# - generates response (user_prompt, context)
# - evaluate user prompt (if the question has any relation to the topic)
# - is a valid question (evaluate whether the question is related to GHG topic) TBD
from os import getenv
from groq import AsyncGroq
import spacy
import os
from groq import Groq
from spacy import load
from spacy.matcher import PhraseMatcher


class RAGAssistant:

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.5,
        max_completion_tokens: int = 800,
    ):
        self.model, self.temp, self.max_tokens = (
            model,
            temperature,
            max_completion_tokens,
        )
        self.disclaimer = (
            "\n\n**Disclaimer:** Be mindful that this is an AI assistant. "
            "Please consult with a professional before proceeding."
        )
        # self.system_config = """You are a digital consultant specializing in Australia's evolving greenhouse gas (GHG) emission regulations.
        # Your task is to help companies navigate the complexities of compliance, accurate emission calculations, and industry-specific scope definitions.
        # Ensure the response is practical, actionable, and aligned with the most recent regulatory updates.
        # If the answer is not available or unclear, state that you do not know.
        # """

        self.system_config = """You are a digital GHG emissions consultant focused on Australian companies operating under Australia's evolving climate disclosure regulations, effective from 2025. All companies you assist are based in Australia.
            Your core role is to guide companies through:
            - Regulatory compliance under Australian laws (e.g., Treasury Act 2024, ASRS, NGER Scheme)
            - Emission calculation practices across Scope 1, Scope 2, and Scope 3
            - Disclosure structure aligned with ASRS (Governance, Strategy, Risk, Metrics & Targets)
            - Industry-specific guidance and emission sources

            Your responses must:
            - Be practical, accurate, and tailored to the company’s context
            - Default to Australian regulatory requirements
            - Reference other frameworks (e.g., U.S. EPA, ISO 14064, GHG Protocol, ESRS, API Compendium) **only if explicitly requested**
            - Indicate if data is insufficient or unclear — do not guess
            - Answer concisely but contextually. Include all relevant information from the context without omitting or summarizing key points. Do not exclude details simply for brevity; instead, express them using clear and efficient language. Your response should be short, but not at the cost of completeness or nuance.

            Your goal is to act as a trustworthy, regulation-aware emissions advisor grounded in Australia’s 2025 climate reporting framework.
            """

        # configuration of the system role
        self.conversation = [{"role": "system", "content": self.system_config}]
        # define legal entities for detection of delicate enquiries
        self.legal_entities = ["LAW", "NORP", "ORG", "GPE"]
        self.financial_entities = ["MONEY", "ORG", "PERCENT", "CARDINAL", "PRODUCT"]
        # Define additional legal and financial keywords
        self.legal_terms = [
            "lawsuit",
            "attorney",
            "plaintiff",
            "defendant",
            "malpractice",
            "contract",
            "liability",
            "sue",
            "court",
            "judge",
            "compliance",
            "regulation",
            "policy",
            "statute",
        ]
        self.financial_terms = [
            "investment",
            "stocks",
            "bond",
            "revenue",
            "profit",
            "bankruptcy",
            "tax",
            "audit",
            "loan",
            "mortgage",
        ]
        # nlp model financial and legal topic detections
        self.nlp = load("en_core_web_md")

        # define GHG keywords
        self.ghg_keywords = [
            "ghg",
            "greenhouse",
            "emission",
            "emissions",
            "carbon",
            "sustainability",
            "climate",
            "regulation",
            "regulatory",
            "compliance",
            "scope",
            "gas",
            "reporting",
            "mitigation",
            "policy",
            "energy",
        ]

    def is_legal_or_financial(self, sample_text: str) -> bool:
        """
        takes any text and detects if the text is related to finance or law using a pretrained model
        this might generate issues if the model is not downloaded
        """
        doc = self.nlp(sample_text)

        matcher = PhraseMatcher(vocab=self.nlp.vocab, attr="lower")
        # add terms to the matcher
        pattern = [self.nlp(term) for term in self.legal_terms + self.financial_terms]
        matcher.add("legal_or_financial", pattern)
        flag = False
        for ent in doc.ents:
            if (
                ent.label_ in self.legal_entities
                or ent.label_ in self.financial_entities
            ):
                flag = True
        matches = matcher(doc)
        if matches:
            flag = True
        return flag

    # def is_related_to_ghg(
    #     self,
    #     user_prompt : str
    # ) -> bool:
    #     """
    #     check if the user prompt is related to GHG regulations
    #     """
    #     for key_word in self.ghg_keywords:
    #         return any(keyword in user_prompt.lower() for keyword in self.ghg_keywords)

    def is_related_to_ghg(self, user_prompt: str) -> str:
        """
        check if the user prompt is related to GHG regulations
        """
        system_prompt_second_model = """
        
        You are are reviewing the context of a Green House and Gas or Environmental Sustainability Governance conversation
        You are given a question and you need to determine if the question is related to Green House and Gas Emissions Regulations or Environmental Sustainability Governance.
        If the question is related to Green House and Gas Emissions Regulations or Environmental Sustainability Governance, you need to return True.
        If the question is not related to Green House and Gas Emissions Regulations or Environmental Sustainability Governance, you need to return False.
        
        If the question is a greeting, a thank you, or a goodbye,s return True
        REMEMBER: You are an advisor specialized in greenhouse gas (GHG) emissions. Your role is to help users understand concepts, policies, impacts, metrics, and strategies related to the reduction, measurement, and management of greenhouse gas emissions.

        Your knowledge is strictly limited to the topic of GHG emissions. You are not allowed to generate code, write scripts, perform general technical calculations, answer unrelated questions (such as health, travel, recipes, general math, or any other field), or act as a general virtual assistant.

        If a user asks a question outside your area of expertise or requests programming, calculations, or other types of technical assistance not directly related to GHG emissions, you must kindly respond False as you cannot help with that and remind them that your purpose is to serve as a GHG advisor.
        
        Limit your answer to True or False. NOTHING ELSE.
        """

        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            client_2 = Groq(api_key=getenv("GROQ_API_KEY"))
            messages_temp = self.conversation.copy()
            messages_temp = messages_temp[-3:]
            messages_temp.append(
                {"role": "system", "content": system_prompt_second_model}
            )
            messages_temp.append(
                {
                    "role": "assistant",
                    "content": "Please answer False or True to the next prompt: ",
                }
            )
            messages_temp.append({"role": "user", "content": user_prompt})
            response = client_2.chat.completions.create(
                messages=messages_temp,
                model="llama3-70b-8192",
                temperature=0.5,
                max_completion_tokens=100,
            )

            result = response.choices[0].message.content.strip()

            # Check if the result is exactly 'True' or 'False'
            if result == "True" or result == "False":
                return result

            attempt += 1
            print(f"Attempt {attempt}: Invalid response '{result}'. Retrying...")

        # If we've exhausted all attempts, return 'False' as a safe default
        print("Maximum attempts reached. Defaulting to 'False'")
        return "False"

    async def generate_response(self, user_prompt: str, context: str = None):

        # check if the user prompt is related to GHG topic
        is_related = self.is_related_to_ghg(user_prompt)
        if is_related != "True":
            return "This digital consultant specializes in Australian GHG emission regulations. Please rephrase your question to focus on topics such as compliance, emission calculations, or scope definitions related to GHG emissions."

        client = AsyncGroq(api_key=getenv("GROQ_API_KEY"))
        # initialize the conversation
        self.conversation.append(
            # configuration of the response
            {
                "role": "assistant",
                "content": f"\n\nUse the following context to provide tailored, concise, and accurate guidance.'{context}'",
            }
        )
        self.conversation.append(
            # adding the query from the user
            {"role": "user", "content": user_prompt}
        )

        messages_temp = self.conversation.copy()
        messages_system = list(filter(lambda l: l.get('role') == "system", messages_temp))
        messages_no_system = list(filter(lambda l: l.get('role') != "system", messages_temp))
        messages_no_system = messages_no_system[-3:]
        # generating the response
        response = await client.chat.completions.create(
            messages=messages_system + messages_no_system,
            model=self.model,
            temperature=self.temp
        )
        # retreiving the output
        ai_ouput = response.choices[0].message.content

        # check if the content el related to legal or financial terms
        if self.is_legal_or_financial(user_prompt):
            ai_ouput += self.disclaimer
        # add to the existing memory of the conversation
        self.conversation.append({"role": "assistant", "content": ai_ouput})
        return ai_ouput

    def set_context_form(self, json_data, files_context=None):
        content_prompt = f"""For the subsequent queries of the conversation, please add to your context the following information
                 provided by the user to provide better guidance based on company details and requirements.
                 Company Data:{json_data}"""
         
        if files_context != None:
             content_prompt += f"""\n These are additional documents uploaded by the company to obtain tailored guidance.
                 Documents Information: {files_context}"""
             
        self.conversation.append(
            {
                "role": "system",
                "content": content_prompt,
            }
        )
