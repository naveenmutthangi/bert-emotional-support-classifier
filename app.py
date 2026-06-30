import os
import json
import math
import random
import re

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Greeting detection
# ---------------------------------------------------------------------------

GREETING_PATTERNS = [
    re.compile(r'^(hi|hey|hello|hiya|howdy|greetings|good\s*(morning|afternoon|evening|day))[!.,?\s]*$', re.I),
    re.compile(r'^(how\s*(are\s*(you|u)|do\s*you\s*do|is\s*it\s*going|goes\s*it|you\s*doing))[!.,?\s]*$', re.I),
    re.compile(r'^(hi|hey|hello)[,\s]+(how\s*(are\s*(you|u)|is\s*it\s*going|you\s*doing))[!.,?\s]*$', re.I),
    re.compile(r'^(sup|wassup|what\'?s\s*up)[!.,?\s]*$', re.I),
    re.compile(r'^(yo|ola|salut)[!.,?\s]*$', re.I),
    re.compile(r'^(hi+|he+y+|ha+llo)[!.,?\s]*$', re.I),
]

GREETING_RESPONSES = [
    "Hello! I'm glad you reached out. What would you like to talk about today?",
    "Hi there! I'm here to listen. What's been on your mind?",
    "Hello! Feel free to share whatever is on your mind — I'm here to help.",
    "Hey! Good to hear from you. What would you like to talk about?",
    "Hello! I'm here for you. What's going on?",
]


def is_greeting(text: str) -> bool:
    t = text.strip()
    return any(p.match(t) for p in GREETING_PATTERNS)


# ---------------------------------------------------------------------------
# Keyword-based BERT simulation
# (Replace classify_message() with real BERT inference once you have weights)
# ---------------------------------------------------------------------------

DIRECTIVE_KEYWORDS = {
    "how": 2.1, "steps": 2.4, "advice": 2.6, "suggest": 2.3, "solution": 2.5,
    "fix": 2.2, "plan": 2.1, "strategy": 2.3, "options": 2.0, "help": 1.4,
    "should": 1.8, "resources": 2.2, "information": 2.0, "recommend": 2.5,
    "what": 1.3, "do": 1.2, "find": 1.6, "ways": 2.0, "tips": 2.4, "guide": 2.3,
    "action": 2.1, "practical": 2.3, "try": 1.5, "tool": 2.0, "approach": 2.0,
    "need": 1.3, "job": 1.6, "career": 1.8, "money": 1.7, "work": 1.5,
    "apply": 2.0, "search": 1.8, "seek": 1.7, "change": 1.5, "improve": 1.9,
    "solve": 2.4, "problem": 1.9, "issue": 1.6, "deal": 1.5, "handle": 1.8,
}

EMOTIONAL_KEYWORDS = {
    "feel": 2.5, "feeling": 2.6, "felt": 2.4, "feelings": 2.5,
    "sad": 2.8, "anxious": 2.9, "anxiety": 2.8, "scared": 2.7, "afraid": 2.7,
    "lonely": 2.9, "alone": 2.5, "depressed": 2.9, "overwhelmed": 2.8,
    "angry": 2.6, "upset": 2.5, "hurt": 2.6, "pain": 2.3, "stressed": 2.7,
    "worried": 2.6, "lost": 2.3, "confused": 2.1, "hopeless": 2.9,
    "exhausted": 2.7, "tired": 2.2, "miserable": 2.9, "heartbroken": 2.9,
    "difficult": 1.8, "hard": 1.6, "tough": 1.7, "cry": 2.7, "crying": 2.7,
    "understand": 1.8, "support": 1.6, "listen": 1.7, "care": 1.7, "empathy": 2.1,
    "emotion": 2.3, "shame": 2.7, "guilt": 2.6, "embarrassed": 2.5, "regret": 2.4,
    "miss": 2.2, "grief": 2.9, "loss": 2.4, "frustrated": 2.5, "disappointed": 2.5,
}

DIRECTIVE_TEMPLATES = [
    "It sounds like you're looking for some direction here. One approach worth considering is to break the situation into smaller, manageable steps — that often makes a complex problem feel more approachable. Have you had a chance to identify which part feels most urgent to tackle first?",
    "That's a real challenge, and there are some practical strategies that tend to help in situations like this. It might be useful to start by mapping out your options clearly, then weigh the pros and cons of each. Would it help to think through what resources or support you have available right now?",
    "From what you're describing, it sounds like gathering more information and making a concrete plan could make a meaningful difference. Many people in similar situations have found it helpful to speak with someone who has specific expertise in this area. Have you explored what professional guidance might be available to you?",
    "One thing that often helps when facing this kind of situation is to focus on what's within your control first. Setting a realistic short-term goal — even a small one — can help build momentum. What's one concrete step you feel confident you could take this week?",
    "It sounds like you have more options than it might feel right now. Sometimes writing down the situation and your possible responses can make the path forward clearer. Would you like to think through the practical options available to you?",
    "There are some effective approaches people use in situations like this. First, it's worth understanding the full picture of what you're dealing with. Then you can look at what resources, skills, or connections might help you address it directly. Where would you like to start?",
]

EMOTIONAL_TEMPLATES = [
    "What you're going through sounds genuinely difficult, and it makes complete sense that you're feeling this way. Your emotions are a natural response to a hard situation, and there's no need to push them aside. I'm here, and I'm listening — please share whatever is on your mind.",
    "I hear you, and I want you to know that what you're feeling is valid. It takes real courage to acknowledge these emotions and reach out. You don't have to face this alone — I'm here with you, and we can go at whatever pace feels right.",
    "That sounds incredibly hard, and I can understand why you'd be feeling overwhelmed right now. Sometimes just putting words to what we're experiencing can bring a little relief. How long have you been carrying this?",
    "Thank you for trusting me with this. What you're describing sounds really painful, and I want you to know I'm taking it seriously. It's okay to feel exactly the way you're feeling — there's no right or wrong way to respond to something this heavy.",
    "I can hear how much this is weighing on you, and I want you to know that your feelings make complete sense given what you're going through. You're not alone in this. Would it help to talk more about what's been the hardest part?",
    "What you're feeling is real and it matters. It sounds like you've been carrying a lot, and I appreciate you sharing that with me. Sometimes the most important thing isn't finding an answer right away — it's just having someone truly listen. I'm here for that.",
]


def softmax(a: float, b: float):
    max_val = max(a, b)
    ea = math.exp(a - max_val)
    eb = math.exp(b - max_val)
    s = ea + eb
    return ea / s, eb / s


def classify_message(text: str) -> dict:
    """
    Simulates binary BERT classification using keyword scoring + softmax.

    To use your actual fine-tuned BERT model instead:
      1. Place your saved model folder inside flask_prototype/model/
      2. Replace this function body with:

         from transformers import BertTokenizer, BertForSequenceClassification
         import torch

         tokenizer = BertTokenizer.from_pretrained('model/')
         model = BertForSequenceClassification.from_pretrained('model/')
         model.eval()

         inputs = tokenizer(text, return_tensors='pt', max_length=128,
                            padding='max_length', truncation=True)
         with torch.no_grad():
             logits = model(**inputs).logits[0].tolist()

         prob_dir, prob_emo = softmax(logits[0], logits[1])
         predicted_class = 'directive_support' if prob_dir > prob_emo else 'emotional_validation'
         confidence = max(prob_dir, prob_emo)

         templates = DIRECTIVE_TEMPLATES if predicted_class == 'directive_support' else EMOTIONAL_TEMPLATES
         return {
             'predicted_class': predicted_class,
             'confidence': confidence,
             'logits': logits,
             'probabilities': [prob_dir, prob_emo],
             'response': random.choice(templates),
         }
    """
    tokens = set(re.sub(r'[^\w\s]', ' ', text.lower()).split())

    dir_score = sum(DIRECTIVE_KEYWORDS.get(t, 0) for t in tokens)
    emo_score = sum(EMOTIONAL_KEYWORDS.get(t, 0) for t in tokens)

    noise = lambda: (random.random() - 0.5) * 0.4
    dir_logit = 1.2 + dir_score * 0.35 + noise()
    emo_logit = 1.2 + emo_score * 0.35 + noise()

    prob_dir, prob_emo = softmax(dir_logit, emo_logit)
    predicted_class = 'directive_support' if prob_dir > prob_emo else 'emotional_validation'
    confidence = max(prob_dir, prob_emo)

    templates = DIRECTIVE_TEMPLATES if predicted_class == 'directive_support' else EMOTIONAL_TEMPLATES
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'logits': [dir_logit, emo_logit],
        'probabilities': [prob_dir, prob_emo],
        'response': random.choice(templates),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()

    if not message:
        return jsonify({'error': 'message is required'}), 400

    if is_greeting(message):
        return jsonify({
            'predicted_class': 'greeting',
            'confidence': 1.0,
            'logits': [0.0, 0.0],
            'probabilities': [0.5, 0.5],
            'response': random.choice(GREETING_RESPONSES),
        })

    return jsonify(classify_message(message))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
