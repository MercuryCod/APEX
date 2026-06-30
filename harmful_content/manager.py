from .prompts import *
import warnings

CATEGORIES = {
    "hate": {
        "name": "Hate",
        "description": "Content that promotes hate based on identity, including hateful symbols and stereotypes.",
        "examples": [
            "Swastika",
            "Confederate flag",
            "SS-runes",
            "KKK hood",
            "Anti-Semitic symbols",
            "Skull and crossbones",
            "White power symbols",
            "Anti-Black symbols",
        ],
    },
    "harassment": {
        "name": "Harassment",
        "description": "Mocking, threatening, or bullying individuals.",
        "examples": [
            "Sexual harassment",
            "Cyberbullying",
            "Threatening messages",
            "Workplace harassment",
            "Online trolling",
            "Being stalked",
        ],
    },
    "violence": {
        "name": "Violence",
        "description": "Violent acts and the suffering or humiliation of others.",
        "examples": [
            "Gun violence",
            "Knife attacks",
            "Assaults",
            "Rape or sexual assault",
            "Torture",
            "Homicide",
            "War and combat scenes",
            "Mass shootings",
            "Child abuse",
            "Animal cruelty",
        ],
    },
    "self_harm": {
        "name": "Self-Harm",
        "description": "Suicide, cutting, eating disorders, and other self-harm behaviors.",
        "examples": [
            "Suicide",
            "Self-hanging",
            "Self-hitting",
            "Self-burning",
            "Skin-picking",
            "Hair-pulling",
            "Self-cutting",
            "Tongue-splitting",
            "Drug/alcohol abuse",
        ],
    },
    "sexual": {
        "name": "Sexual",
        "description": "Nudity, sexual acts, or content meant to arouse sexual excitement.",
        "examples": [
            "Erotic content",
            "Genital exposure",
            "Pornography",
            "Suggestive poses",
            "Sexual fetish",
            "Sexual violence",
            "Nudity in public",
            "Sex toys",
            "Adult products",
            "Cartoon or animated nudity",
        ],
    },
    "shocking": {
        "name": "Shocking",
        "description": "Bodily fluids, obscene gestures, or other shocking/disgusting content.",
        "examples": [
            "Gore and mutilation",
            "Disturbing injuries",
            "Human flesh",
            "Human and animal suffering",
            "Demonic figures",
            "Cult and ritualistic activities",
        ],
    },
    "illegal_activity": {
        "name": "Illegal Activity",
        "description": "Drug use, theft, vandalism, and other illegal acts.",
        "examples": [
            "Robbery",
            "Burglary",
            "Arson",
            "Counterfeiting",
            "Graffiti",
            "Credit card fraud",
            "Child pornography",
            "Cybercrime",
            "Human smuggling",
        ],
    },
    "deception": {
        "name": "Deception",
        "description": "Major conspiracies or geopolitical misinformation.",
        "examples": [
            "New World Order",
            "Moon landing hoax",
            "Illuminati",
            "JFK assassination",
            "9/11 conspiracy",
            "Flat earth theory",
            "UFO conspiracy",
            "Pizzagate conspiracy",
        ],
    },
    "political": {
        "name": "Political",
        "description": "Content that may influence political processes or campaigns.",
        "examples": [
            "Elections",
            "Political campaigns",
            "Voting",
            "Protests",
            "Political posters",
            "Political memes",
            "Political slogans",
            "Political flags",
        ],
    },
    "public_personal_health": {
        "name": "Public and Personal Health",
        "description": "Content related to treatment, prevention, or diagnosis of diseases.",
        "examples": [
            "Vaccination",
            "Epidemiology",
            "Infection control",
            "Mental health",
            "Substance abuse",
            "Healthcare",
            "Smoking cessation",
            "Therapy",
        ],
    },
    "spam": {
        "name": "Spam",
        "description": "Unsolicited bulk content or misleading advertisements.",
        "examples": [
            "Viagra",
            "Cialis",
            "Online pharmacy",
            "Levitra",
            "Online casino",
            "Poker",
            "Buy Tramadol",
            "Pop-up ads",
        ],
    },
}


class HarmfulContentCategory:

    def __init__(
        self,
        name: str,
        description: str,
        examples: list[str],
        initial_prompts: list[str],
    ):
        self.name = name
        self.description = description
        self.examples = examples
        self.initial_prompts = initial_prompts

    def __str__(self):
        return f"Category: {self.name}\nDescription: {self.description}\nExamples: {self.examples}"

    def get_initial_prompts(self):
        return self.initial_prompts


class HarmfulContentManager:
    def __init__(self):
        categories = CATEGORIES
        self.categories = {}
        for category_name, category_data in categories.items():
            category_name = category_name.lower()
            self.categories[category_name] = HarmfulContentCategory(
                name=category_data["name"],
                description=category_data["description"],
                examples=category_data["examples"],
                initial_prompts=initial_prompts[category_name],
            )

    def __len__(self) -> int:
        return len(self.categories)

    def get_category(self, name: str=None) -> HarmfulContentCategory:
        if name is None:
            return self.categories
        else:
            name = name.strip().lower()
            return self.categories.get(name, None)

    def format_category(self, category_name: str=None) -> str:
        if category_name is None:
            return self.format_all_categories()
        else:
            category = self.get_category(category_name)
            return str(category)

    def format_all_categories(self) -> str:
        return "\n".join(
            [
                f"{index + 1}. {str(category)}\n"
                for index, category in enumerate(self.categories.values())
            ]
        )

    def get_all_category_names(self) -> list[str]:
        return list(self.categories.keys())

    def get_initial_prompts(self, category_name: str, number_of_prompts: int) -> list[str]:
        category = self.get_category(category_name)
        if category is None:
            raise ValueError(f"Category with name '{category_name}' not found.")

        prompts = category.get_initial_prompts()
        if len(prompts) < number_of_prompts:
            warnings.warn(
                f"Category with name '{category_name}' has only {len(prompts)} initial prompts."
            )
        return prompts[:number_of_prompts]



