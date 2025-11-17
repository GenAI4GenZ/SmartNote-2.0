from typing import List
import openai
import time
import tiktoken
from loguru import logger
import httpx

from .dtypes import PROJECT_DOMAIN_TYPE, STRUCTURE_TYPE, ReleaseNoteEntry, OpenAIConfig
from .config import settings

# Prompt engineering tactics based on documentation from OpenAI
# "Use delimiters to clearly indicate distinct parts of the input" https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-delimiters-to-clearly-indicate-distinct-parts-of-the-input
# "Chain of Thought Prompting: Split complex tasks into simpler subtasks" https://platform.openai.com/docs/guides/prompt-engineering/strategy-split-complex-tasks-into-simpler-subtasks
# "One Shot Prompting: Provide examples" https://platform.openai.com/docs/guides/prompt-engineering/tactic-provide-examples
# "Use intent classification to identify the most relevant instructions for a user query" https://platform.openai.com/docs/guides/prompt-engineering/tactic-use-intent-classification-to-identify-the-most-relevant-instructions-for-a-user-query
# "Specify the desired length of the output" https://platform.openai.com/docs/guides/prompt-engineering/tactic-specify-the-desired-length-of-the-output

# Domain descriptions and content (ref: Characterize Software Release Notes of GitHub Projects: Structure, Writing Style, and Content)
DOMAIN_DESCRIPTION: dict[PROJECT_DOMAIN_TYPE, str] = {
    "System": "Software that offers basic services and infrastructure to other software, e.g., operating systems, servers, and databases",
    "Tool": "Software that facilitates developers with universal software development tasks, like IDEs and compilers",
    "Library": "Software that provides a collection of reusable functionalities to facilitate software development in specific domains such as Web and machine learning",
    "Application": "Software that offers end-users with functionality, such as browsers and text editors",
}

DOMAIN_HINT: dict[PROJECT_DOMAIN_TYPE, str] = {
    "System": "prioritize breaking changes and security changes but provide a more comprehensive introduction of various categories to serve a variety of audiences",
    "Tool": "prioritize performance, breaking changes, enhancements, and security to strengthen developers' confidence when they develop software with it",
    "Library": "prioritize breaking changes and document changes to facilitate the usage of downstream projects",
    "Application": "prioritize performance, document changes, and dependency/environment information for ease of installation and upgrade for end users",
}

def _limit_str_len(s: str):
    if not isinstance(s, str):
        s = str(s)
    if len(s) >= 1000:
        return s[:500] + f"...[truncated {len(s)-1000} chars]..." + s[-500:]
    return s

class PromptsManager:
    def __init__(self, openai_config: OpenAIConfig):
        # create an OpenAI client
        httpx_proxy = None
        try:
            httpx_proxy = settings.proxy.https
        except (KeyError, AttributeError):
            pass

        proxy_httpx_client = httpx.Client(proxy=httpx_proxy)
        self.openai_client = openai.Client(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url,
            http_client=proxy_httpx_client,
        )
        self._oai_conf = openai_config
        self.tokenizer = tiktoken.encoding_for_model(openai_config.model)
    
    def tokenize(self, tcr: str):
        return self.tokenizer.encode(tcr, disallowed_special=())
    
    def count_tokens(self, tcr: str):
        return len(self.tokenize(tcr))
    
    def count_tokens_approx(self, tcr: str):
        """Approximate token count for ASCII characters"""
        return len(tcr) / 4
    
    def get_openai_config(self):
        return self._oai_conf

    def send_request(
        self, chat_messages: list[dict[str, str]]
    ) -> openai.ChatCompletion:
        completion = None

        logger.debug(f"<OpenAI> request: {_limit_str_len(chat_messages[-2:])}")
        try:
            completion = self.openai_client.chat.completions.create(
                messages=chat_messages,
                temperature=self._oai_conf.temperature,
                top_p=self._oai_conf.top_p,
                model=self._oai_conf.model,
            )
        except openai.RateLimitError:
            logger.info("Rate limit reached, waiting 1 minute before proceeding... ")
            time.sleep(60)
            completion = self.send_request(chat_messages)
        except openai.OpenAIError as e:
            logger.error(f"OpenAI Error: {e}")
            raise e
        logger.debug(f"<OpenAI> response: {_limit_str_len(completion.choices[0].message.content)}")

        return completion
    
    def get_first_completion(self, chat_messages: list[dict[str, str]]) -> str:
        completion = self.send_request(chat_messages)
        return completion.choices[0].message.content

    def summarize_commit(
        self, tcr: str, provide_technical_detail: bool
    ) -> str:
        logger.debug(f"Summarizing commit: {tcr[:40]}...")
        technical_detail_prompt = ""
        if provide_technical_detail:
            technical_detail_prompt = "Keep in mind most users of the project are developers, "
            "who benefit from technical details, therefore it would be beneficial to include technical details.\n\n"

        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with commit details (delimited with XML tags) and file changes "
                    "(delimited with XML tags) outlining the changes of a commit. "
                    + technical_detail_prompt
                    + "Write a concise description suitable for inclusion in a software release note aimed at software developers.\n\n"
                    "Guidelines:\n"
                    "- Focus on what changed and why it matters; omit low-level implementation noise such as long file or line lists.\n"
                    "- Use one or two short, direct sentences (ideally under 20 words each).\n"
                    "- Avoid nested clauses (for example, avoid chaining multiple 'which', 'that', or 'in order to').\n"
                    "- Prefer simple verbs such as 'add', 'remove', 'fix', 'use', 'update', or 'improve' instead of complex alternatives.\n"
                    "- Include at most one or two important technical entities (for example, the main module, feature, or API name).\n"
                    "- Use clear, neutral language and avoid marketing-style wording.\n\n"
                    "Return only the final description. Do not include headings, bullet markers, or any extra commentary."
                ),
            },
            {"role": "user", "content": tcr},
        ]
        return self.get_first_completion(prompt)

    def summarize_pr(
        self,
        entries: List[ReleaseNoteEntry],
        commit_dt: str,
        pr_title: str,
        pr_body: str,
        provide_technical_detail: bool,
    ) -> str:
        logger.debug(f"Summarizing PR: {pr_title[:40]}...")
        combined_entries = ""
        for entry in entries:
            combined_entries += f"""- [{entry.significance:.2f}] <{commit_dt}> {entry.summary.strip()}\n"""

        technical_detail_prompt = ""
        if provide_technical_detail:
            technical_detail_prompt = "Keep in mind most users of the project are developers, who benefit from technical details, therefore it would be beneficial to include technical details. "

        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with pull request details (delimited with XML tags) and "
                    "a list of summarized commits (delimited with XML tags), each prefixed with a numeric significance score in [0, 1]. "
                    + technical_detail_prompt
                    + "Your task is to produce a short PR-level summary suitable for a release note aimed at software developers.\n\n"
                    "Interpretation of significance:\n"
                    "- Higher scores indicate more important commits; very low scores indicate minor or cosmetic changes.\n\n"
                    "Internal reasoning steps (do not enumerate them in the output):\n"
                    "1. Identify the main themes and user-visible changes using the PR details and higher-significance commits.\n"
                    "2. De-emphasize or omit very low-significance commits unless they are critical for understanding the change.\n"
                    "3. Remove references to specific files, line numbers, and dependency/version bumps unless they are central to the change.\n"
                    "4. Draft a short PR-level summary capturing what changed and why it matters.\n"
                    "5. Rewrite the draft so that all sentences are short, direct, and easy to read.\n\n"
                    "Style and readability requirements for the final output:\n"
                    "- Use 2-4 short, direct sentences (fewer is fine for very small PRs).\n"
                    "- Ideally keep each sentence under 20 words.\n"
                    "- Avoid nested clauses (for example, avoid sentences that chain multiple 'which', 'that', or 'in order to').\n"
                    "- Prefer simple verbs such as 'add', 'remove', 'fix', 'use', 'update', or 'improve' instead of complex alternatives.\n"
                    "- Mention only the most important technical entities (such as key features, modules, or APIs). Do not list many internal identifiers.\n"
                    "- Use clear, neutral, technical language; avoid marketing or exaggerated claims.\n\n"
                    "Return only the final PR-level summary as plain text. Do not include headings, bullet markers, or step labels."
                ),
            },
            {
                "role": "user",
                "content": f"<pull_request_details>Title: {pr_title}\n\nBody: {pr_body}\n\n</pull_request_details><summarized_commits>{combined_entries}</summarized_commits>",
            },
        ]

        return self.get_first_completion(prompt)

    def rewrite_to_suit_pd(
        self, 
        release_note: str, 
        project_domain: PROJECT_DOMAIN_TYPE,
        structure_type: STRUCTURE_TYPE
    ) -> str:
        logger.debug(f"Rewriting release note for project domain: {project_domain}...")
        categories: str = "-" + "\n-".join(settings.categories.conventional_commits) + "\n"
        category_prompt: str = ""
        if structure_type == "Change Type":
            category_prompt = f"You can categorize the each list item into the following categories:\n{categories}"
        else:
            category_prompt = "Do not add any prefixes."

        prompt = [
            {
                "role": "system",
                "content": str(
                    f"You will be provided with a release note document (formatted in markdown) for a {project_domain} project "
                    f"({DOMAIN_DESCRIPTION[project_domain]}).\n\n"
                    f"Rewrite and reorder each list item so as to {DOMAIN_HINT[project_domain]}. "
                    "You may lightly edit wording for clarity and tone, and you may reorder list items and headings if it improves organisation.\n\n"
                    "Structural constraints (must be respected):\n"
                    "- Do NOT add or remove headings; keep the existing heading hierarchy and bullet structure.\n"
                    "- Do NOT remove any bullet items or links.\n"
                    "- Do NOT change markdown formatting (for example, keep bullet markers and indentation intact).\n\n"
                    "Style and readability requirements:\n"
                    "- Use short, direct sentences (ideally under 20 words per sentence).\n"
                    "- Avoid nested clauses (for example, avoid sentences that chain multiple 'which', 'that', or 'in order to').\n"
                    "- Prefer simple verbs such as 'add', 'remove', 'fix', 'use', 'update', or 'improve' instead of complex alternatives.\n"
                    "- Preserve important technical entities (names of features, modules, APIs), but avoid listing many low-level internal identifiers.\n"
                    "- Use clear, neutral language appropriate for developers reading release notes.\n\n"
                    + category_prompt
                ),
            },
            {"role": "user", "content": f"{release_note}"},
        ]
        return self.get_first_completion(prompt)

    def rewrite_for_conciseness(
        self, release_note: str
    ) -> str:
        logger.debug("Rewriting release note for conciseness...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with an entry from a release note document (formatted in markdown). "
                    "Analyze the entry and rewrite it for conciseness while preserving all information and links that are useful to readers.\n\n"
                    "Guidelines:\n"
                    "- Remove redundant phrases, filler, and vague sentence openers (such as 'This change', 'In this commit', or 'It aims to').\n"
                    "- Keep all links and any essential technical entities.\n"
                    "- Use one or two short, direct sentences (ideally under 20 words each).\n"
                    "- Avoid nested clauses (for example, avoid sentences that chain multiple 'which', 'that', or 'in order to').\n"
                    "- Prefer simple verbs such as 'add', 'remove', 'fix', 'use', 'update', or 'improve' instead of complex alternatives.\n"
                    "- Use clear, neutral language appropriate for developers reading release notes.\n\n"
                    "Do not change the markdown structure or add a prefix. "
                    "Most importantly, do not remove any links."
                ),
            },
            {"role": "user", "content": f"{release_note}"},
        ]
        return self.get_first_completion(prompt)

    def combine_similar_entries(
        self, release_note: str
    ) -> str:
        logger.debug("Combining similar entries in release note...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with a release note document (formatted in markdown). "
                    "Combine similar points into a single list item to improve conciseness but do not combine them if it makes the list item dramatically longer.\n\n"
                    "Do not change or add new headings, the formatting of the document or add a prefix."
                    "Most importantly, do not remove any links."
                ),
            },
            {"role": "user", "content": f"{release_note}"},
        ]
        return self.get_first_completion(prompt)

    def rewrite_name_refs(
        self, release_note: str
    ) -> str:
        logger.debug("Rewriting release note to use new names...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with a release note document (formatted in markdown). "
                    "Analyze the change and rewrite it, making sure that all references to names are using the new name. "
                    "For example, if a function or variable was added and later renamed, the new name should be used and there should be no mention of it being renamed. \n\n"
                    "Do not change the formatting of the document or add a prefix. For example, do not change or add new headings."
                    "Most importantly, do not remove any links."
                ),
            },
            {"role": "user", "content": f"{release_note}"},
        ]
        return self.get_first_completion(prompt)

    def rewrite_func_info(
        self, release_note: str
    ) -> str:
        logger.debug("Rewriting release note to include function information...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with a release note document (formatted in markdown). "
                    "Analyze the change and rewrite it, removing references to changes of changes. "
                    "For example, if a function or variable was added and then later removed in the same commit or PR then it shouldn't be mentioned. "
                    "Similarly, if a function or variable was added in the same commit or PR and later changed, only information relevant to the latest state should be retained..\n\n"
                    "Do not change the formatting of the document or add a prefix. For example, do not change or add new headings."
                    "Most importantly, do not remove any links."
                ),
            },
            {"role": "user", "content": f"{release_note}"},
        ]
        return self.get_first_completion(prompt)

    def determine_commit_module(
        self, sha_combined_tcr: str
    ) -> str:
        logger.debug("Determining the most significantly affected module...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with commit details (delimited with XML tags) and file changes (delimited with XML tags) outlining the changes of a commit. "
                    "From the provided data, determine the most significantly affected module."
                    "\n\nYour response must be the identified module but rewritten so that it can be included as a header in documentation."
                    "\n\nDo not format your response in anyway."
                    "Most importantly, do not remove any links."
                ),
            },
            {"role": "user", "content": f"{sha_combined_tcr}"},
        ]
        return self.get_first_completion(prompt)

    def refine_module_categories(
        self, release_note: str
    ) -> str:
        logger.debug("Refining module categories in release note...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with a release note document (formatted in markdown). "
                    "Reduce the number of categories (headings) by combining them into an existing category.  "
                    "If they cannot be combined or the category does not contain many changes, categorize the changes under a misc category."
                    "\n\nOther than the misc category, do not add new category."
                ),
            },
            {"role": "user", "content": f"{release_note}"},
        ]
        return self.get_first_completion(prompt)
    
    def are_commit_messages_good(
        self, commit_messages: List[str]
    ) -> bool:
        """
        Guide GPT to determine whether commit messages are 'good' based on Yuxia's taxonomy
        of good commit messages (https://arxiv.org/pdf/2202.02974). In short, a commit message
        is considered good if it describes 'what' and 'why'.
        """
        logger.debug("Determining if commit messages are good and relevant...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with a list of commit messages."
                    "From the provided data, determine if the commit messages are of high quality and relevant to the project."
                    "A good commit message usually describes 'what' (summarize the change and the design decisions) "
                    "and 'why' (what is the problem, what is the goal of the change, and why the change is necessary). "
                    "\nExamples of good commit messages: "
                    "\n- Remove outdated key. `aggregate-key-pattern` is no longer defined but was still referenced in the documentation."
                    "\n- Fix concurrent problem of zookeeper configcenter, wait to start until cache being fully populated."
                    "\n- Polish pom.xml. Apply consistent formatting, drop JDK 8 support and cleanup repo."
                    "\nExamples of bad commit messages: "
                    "\n- Update README.md"
                    "\n- Fix bug"
                    "\n- A lot of changes"
                    "\n\nYour response must be a 'yes' or 'no' answer."
                    "\n\nDo not format your response in anyway."
                ),
            },
            {"role": "user", "content": '\n'.join(commit_messages)},
        ]
        _response = self.get_first_completion(prompt)
        if _response.strip().lower() == "no":
            return False
        if _response.strip().lower() == "yes":
            return True
        logger.warning(f"Unexpected response from GPT binary classification: {_response}")
        return False
    
    def reorder_rn_categories(
        self, release_note: str
    ) -> str:
        logger.debug("Reordering release note categories...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with a release note document (formatted in markdown). "
                    "Reorder the categories so that the most important categories are at the top and the least important are at the bottom."
                    "Normally, breaking changes and new features, bug fixes, and enhancements are considered the most important. "
                    "Document changes, dependency updates, and version changes are considered less important."
                    "\n\nDo not format the response in any way. "
                    "Do not add new categories. Do not remove any links. "
                    "Do not change the formatting of the document or add a prefix. "
                ),
            },
            {"role": "user", "content": f"{release_note}"},
        ]
        return self.get_first_completion(prompt)
    
    def determine_project_domain(
        self, project_description: str, readme_content: str
    ) -> PROJECT_DOMAIN_TYPE:
        """
        Guide GPT to classify a GitHub project into one of four categories:
        System Software, Libraries & Frameworks, Software Tools, or Application Software.
        """
        logger.debug("Classifying GitHub project...")
        prompt = [
            {
                "role": "system",
                "content": str(
                    "You will be provided with a project description and README content for a GitHub project. "
                    "Classify the project into one of the following categories:\n"
                    "System: software that offers basic services and infrastructure to other software, e.g., operating systems, servers, and databases.\n"
                    "Library: software that provides a collection of reusable functionalities to facilitate software development in specific domains such as Web and machine learning.\n"
                    "Tool: software that facilitates developers with universal software development tasks, like IDEs and compilers.\n"
                    "Application: software that offers end-users with functionality, such as browsers and text editors.\n\n"
                    "Your response must be one of these four categories exactly as written above.\n"
                    "Do not format your response in any way or provide any additional explanation."
                ),
            },
            {
                "role": "user", 
                "content": f"Project Description:\n{project_description}\n\nREADME Content:\n{readme_content}"
            },
        ]
        _response = self.get_first_completion(prompt)
        _response = _response.strip()

        if _response in settings.categories.project_domains:
            return _response
        else:
            logger.warning(f"Unexpected response from GPT classification: {_response}")
            return "Unclassified"
