"""
CineBot - A Movie & TV Recommender Chatbot

NLP techniques:
  - Tokenisation, stopword removal, stemming  (NLTK)
  - Sentiment analysis                        (NLTK VADER)
  - TF-IDF retrieval + cosine similarity      (scikit-learn)
  - Intent classification (TF-IDF + LinearSVC)(scikit-learn)
  - Fine-tuned causal LM                      (Qwen3.5-0.8B + LoRA)
"""

import os
import random
import difflib

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np

import torch
from chatbot_base import ChatbotBase

nltk.download('punkt',         quiet=True)
nltk.download('punkt_tab',     quiet=True)
nltk.download('stopwords',     quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Maps user words directly to genre labels
from data import GENRE_MAP, MEDIA, INTENT_TEXTS, INTENT_LABELS, INTENT_NAMES



class CineBot(ChatbotBase):

    def __init__(self):
        super().__init__(name="CineBot")
        del self.conversation_is_active  
        self._active      = True
        self.seen_titles  = set()
        self.last_matches = []
        self.last_pick    = None

        self.stemmer    = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.sia        = SentimentIntensityAnalyzer()

        self.tfidf        = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform([m["description"] for m in MEDIA])

        self.intent_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf',   LinearSVC()),
        ])
        
        self.intent_classifier.fit(INTENT_TEXTS, INTENT_LABELS)

        self.llm_model     = None
        self.llm_tokenizer = None
        self._load_finetuned_model()

    # ── LLM (Ollama) ─────────────────────────────────────────────────────────

    def _load_finetuned_model(self):
        """Load the fine-tuned HuggingFace model from ./cinebot-model if it exists."""
        model_path = "./cinebot-model"

        if not os.path.exists(model_path):
            print("(No fine-tuned model found -- run finetune.py first)\n")
            self.llm_model = None
            self.llm_tokenizer = None
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            print("Loading fine-tuned CineBot model...")
            base = "Qwen/Qwen3.5-0.8B"
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

            self.llm_model.eval()
            print("Fine-tuned model loaded.\n")
        
        except Exception as e:
            print(f"(Could not load model: {e})\n")
            self.llm_model = None
            self.llm_tokenizer = None

    def generate_llm_response(self, title, genres):
        """Generate a short hook using the fine-tuned local model"""
        if not self.llm_model or not self.llm_tokenizer:
            return None
        
        try:
            import torch

            prompt = (
                "### User: Write a short, one-sentence recommendation for the Sci-Fi film The Matrix.\n"
                "### CineBot: Get ready for an incredible sci-fi experience!\n"
                f"### User: Write a short, one-sentence recommendation for the {genres} {title}.\n"
                "### CineBot:"
            )

            inputs = self.llm_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(self.llm_model.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs, 
                    max_new_tokens=60,
                    temperature=0.3,
                    do_sample=True, 
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            reply = self.llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            stop_markers = [
                "### User:", "User:", 
                "Question:", "assistant", 
                "<think>", "Thinking Process:", 
                "\n"
            ]

            for marker in stop_markers:
                if marker in reply:
                    reply = reply.split(marker)[0].strip()

            reply = reply.replace("#", "").replace("*", "").strip()

            #print(f"\n[DEBUG] Raw LLM Reply: '{reply}'")

            if reply and 15 < len(reply) < 300:
                return reply
            return None
        except Exception as e:
            print(f"\n[DEBUG] LLM Error: {e}")            
            return None

    # ── NLP pipeline ─────────────────────────────────────────────────────────

    def classify_intent(self, text):
        return INTENT_NAMES.get(int(self.intent_classifier.predict([text])[0]), "recommend")

    def analyse_sentiment(self, text):
        score = self.sia.polarity_scores(text)['compound']
        return 'positive' if score >= 0.3 else 'negative' if score <= -0.3 else 'neutral'

    def fuzzy_expand_query(self, query):
        """Expands genre words and fixes typos before TF-IDF search."""
        synonyms = {
            "horror": "horror scary terrifying", "comedy": "comedy funny laugh hilarious",
            "action": "action fight adrenaline",  "thriller": "thriller suspense tense",
            "drama": "drama emotional serious",    "romance": "romance love romantic",
            "sci-fi": "sci-fi space future",       "documentary": "documentary real factual",
            "animation": "animation cartoon anime","adventure": "adventure journey quest",
            "fantasy": "fantasy magical",          "mystery": "mystery detective clue",
        }
        
        known = list(synonyms.keys()) + ["funny", "scary", "emotional", "dark", "classic"]
        expanded = " ".join(synonyms.get(w, w) for w in query.lower().split())
        corrected = []
        
        for word in expanded.split():
            match = difflib.get_close_matches(word, known, n=1, cutoff=0.75)
            corrected.append(match[0] if match else word)
        return " ".join(corrected)

    def tfidf_match(self, query, top_n=10):
        query_vec = self.tfidf.transform([self.fuzzy_expand_query(query)])
        sims      = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx   = np.argsort(sims)[::-1][:top_n]
        
        return [(MEDIA[i], float(sims[i])) for i in top_idx if sims[i] > 0]

    # ── Base method overrides ─────────────────────────────────────────────────

    def greeting(self):
        print("\nCineBot: Hey! I'm CineBot, your personal movie & TV recommender!")
        print("Tell me what you're in the mood for -- a genre, a feeling, a decade.")
        print("Type 'help' to see all commands, or 'quit' to exit.\n")

    def farewell(self):
        print("\nCineBot: Enjoy whatever you watch! See you next time.\n")

    def conversation_is_active(self):
        return self._active

    def receive_input(self):
        return input("You: ").strip()

    def process_input(self, user_input):
        if not user_input:
            return {"raw": "", "intent": "recommend", "sentiment": "neutral",
                    "tfidf_matches": [], "decades": [], "media_type": None, "genres": []}

        lower   = user_input.lower()

        if lower.strip() in ["help", "commands", "options", "what can you do", "?"]:
            return {"raw": lower, "intent": "help", "sentiment": "neutral",
                    "tfidf_matches": [], "decades": [], "media_type": None, "genres": []}

        if any(p in lower for p in ["tell me more", "why should i", "why watch", "more about", "what's it about", "tell me about"]):
            return {"raw": lower, "intent": "blurb", "sentiment": "neutral",
                    "tfidf_matches": [], "decades": [], "media_type": None, "genres": []}
       
        tokens  = word_tokenize(lower)
        stemmed = [self.stemmer.stem(t) for t in tokens if t.isalpha() and t not in self.stop_words]

        genres  = list({g for w in lower.split() if w in GENRE_MAP for g in GENRE_MAP[w]})

        decades = [d for d, kws in {
            "1980s": ["80s", "eighties"], "1990s": ["90s", "nineties"],
            "2000s": ["2000s", "noughties"], "2010s": ["2010s"],
            "2020s": ["2020s", "latest", "recent", "new"],
        }.items() if any(kw in lower for kw in kws)]

        media_type = (
            "movie" if any(w in stemmed for w in ["film", "movi", "cinema"]) else
            "tv"    if any(w in stemmed for w in ["show", "seri", "tv", "episod", "season", "bing"]) else
            None
        )

        return {
            "raw":           lower,
            "intent":        self.classify_intent(user_input),
            "sentiment":     self.analyse_sentiment(user_input),
            "tfidf_matches": self.tfidf_match(lower),
            "genres":        genres,
            "decades":       decades,
            "media_type":    media_type,
        }

    def generate_response(self, processed_input):
        intent = processed_input.get("intent", "recommend")

        if intent == "farewell":
            self._active = False
            return None
        
    # ────── Blurb Intent ─────────────────────────────────────────────────

        if intent == "blurb":
            if not self.last_pick:
                return "CineBot: I haven't recommended anything yet -- tell me a genre first!"
            
            title  = self.last_pick["title"]
            genres = ", ".join(self.last_pick["genres"]).title()

            #desc_keywords = self.last_pick.get("description", "Great Themes").title()

            full_plot = self.last_pick.get("description", "A great watch.")
            #formatted_desc = desc_keywords.replace(" ", ", ")

            blurb  = self.generate_llm_response(title, genres)

            if blurb:
                return f"CineBot: {blurb}\n\nThe plot: {full_plot}"
            
            return f"CineBot: {title} is a {genres} worth watching!\n\nThe plot: {full_plot}"
        
        if intent == "help":
            return (
                "\nCineBot: Here's what you can say:\n"
                "\n  GENRES     -- horror, comedy, action, thriller, drama, romance,"
                "\n                 sci-fi, documentary, animation, adventure, fantasy, mystery"
                "\n  DECADES    -- 80s, 90s, 2000s, 2010s, 2020s"
                "\n  TYPE       -- film / movie,  show / tv / series"
                "\n  COMBOS     -- 'scary 80s movie', 'funny tv show', 'romantic drama'"
                "\n  more       -- get another recommendation"
                "\n  tell me more -- get a short hook about the last recommendation"
                "\n  bye / quit -- exit CineBot"
                "\n\nExamples: 'horror', 'funny 90s film', 'something dark and intense'"
            )

        if intent == "frustrated":
            return "\nCineBot: Sorry! Tell me a genre or mood and I'll find something better."

        if intent == "positive_feedback":
            return "\nCineBot: Glad you liked it! What else are you in the mood for?"

        if intent == "more":
            return self._recommend_from(self.last_matches)

        genres     = processed_input.get("genres", [])
        decades    = processed_input.get("decades", [])
        media_type = processed_input.get("media_type")
        matches    = processed_input.get("tfidf_matches", [])
        raw        = processed_input.get("raw", "")

        if not raw:
            return "CineBot: What are you in the mood for?"

        # Genre filter takes priority; fall back to TF-IDF results
        if genres:
            candidates = [m for m in MEDIA if any(g in m["genres"] for g in genres)]
        
        else:
            candidates = [m for m, _ in matches] if matches else list(MEDIA)

        if decades:
            candidates = [m for m in candidates if m["decade"] in decades]
        
        if media_type:
            candidates = [m for m in candidates if m["type"] == media_type]

        # Final fallback -- relax filters one at a time
        if not candidates and decades:
            candidates = [m for m in MEDIA if any(g in m["genres"] for g in genres)] if genres else list(MEDIA)
        if not candidates:
            candidates = list(MEDIA)

        self.last_matches = candidates
        return self._recommend_from(candidates)

    def _recommend_from(self, candidates):
        unseen = [m for m in candidates if m["title"] not in self.seen_titles]
        
        if not unseen:
            for m in candidates:
                self.seen_titles.discard(m["title"])
            unseen = candidates
        
        if not unseen:
            return "\nCineBot: Nothing matched that -- try a different genre or mood!"

        pick   = random.choice(unseen)
        self.seen_titles.add(pick["title"])
        self.last_pick = pick
        label  = "Movie" if pick["type"] == "movie" else "TV Show"
        genres = ", ".join(pick["genres"]).title()
        result = f"\nCineBot: I think you'd enjoy...\n  {label}: \"{pick['title']}\"\n  Genres: {genres}  |  Era: {pick['decade']}"
        result += "\n\nWant another? Say 'more', ask 'tell me more', or tell me something different!"
        
        return result

    def respond(self, out_message=None):
        if isinstance(out_message, str):
            print(out_message)

        processed = self.process_input(self.receive_input())
        return self.generate_response(processed)