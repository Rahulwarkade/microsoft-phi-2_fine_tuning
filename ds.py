import json
import random
from typing import List, Dict
from datasets import load_dataset
from itertools import cycle

MAX_DATASET_SIZE = 10000
TONEOP_TARGET = 6000  # 60% of 10k
ALPACA_TARGET = 2000  # 20% of 10k
REJECTION_TARGET = 2000  # 20% of 10k

class FocusedToneOpGenerator:
    def __init__(self):
        self.alpaca_data = None
        self.load_external_data()
    
    def load_external_data(self):
        """Load limited Alpaca data (max 2k samples)"""
        try:
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            # Filter for basic communication only
            self.alpaca_data = [
                ex for ex in dataset 
                if any(greet in ex["instruction"].lower() 
                     for greet in ["hi", "hello", "how are", "thank", "bye"])
            ][:2000]
        except Exception as e:
            print(f"Error loading Alpaca dataset: {e}")
            self.alpaca_data = []

    def generate_toneop_samples(self) -> List[Dict]:
        """Generate maximum samples from ToneOp sources"""
        samples = []
        
        # 1. Enhanced chunking and question generation
        for product_type, text in TONEOP_DOCS.items():
            chunks = self.split_into_chunks(text, chunk_size=100)  # Smaller chunks
            for chunk in chunks:
                questions = self.generate_questions_from_chunk(chunk, product_type)
                answers = self.generate_varied_answers(chunk, product_type)
                
                # Create multiple Q&A pairs per chunk
                for q in questions:
                    for a in answers:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPTS[product_type]},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ]
                        samples.append(self.format_chat(messages))
        
        # 2. Multi-turn conversations (10 per product type)
        for _ in range(10):
            samples.extend(self.create_product_conversations("fit"))
            samples.extend(self.create_product_conversations("eats"))
            samples.extend(self.create_product_conversations("care"))
            samples.extend(self.create_product_conversations("founder"))
            samples.extend(self.create_product_conversations("faqs"))
        
        # 3. Edge cases (100 samples)
        samples.extend(self.create_toneop_edge_cases())
        
        return samples[:TONEOP_TARGET]  # Ensure we don't exceed target

    def split_into_chunks(self, text: str, chunk_size=100) -> List[str]:
        """More aggressive chunking to maximize samples"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    def generate_questions_from_chunk(self, chunk: str, product_type: str) -> List[str]:
        """Generate multiple question variations per chunk"""
        questions = []
        keywords = [w for w in chunk.split() if len(w) > 4][:5]  # Get meaningful keywords
        
        base_questions = [
            f"What does ToneOp {product_type} say about {keywords[0]}?",
            f"How does {product_type} handle {keywords[1]}?",
            f"Explain {keywords[2]} in ToneOp {product_type}",
            f"Can you describe {keywords[3]} in {product_type}?",
            f"What should I know about {keywords[4]}?",
            f"Tell me more about {keywords[0]} and {keywords[1]}",
            f"How is {keywords[2]} implemented in {product_type}?",
            f"What are the benefits of {keywords[3]}?",
            f"Does {product_type} offer {keywords[4]}?",
            f"Why is {keywords[0]} important in {product_type}?"
        ]
        return base_questions

    def generate_varied_answers(self, chunk: str, product_type: str) -> List[str]:
        """Generate multiple answer variations from a chunk"""
        sentences = [s.strip() for s in chunk.split('.') if s.strip()]
        answers = []
        
        # Create different length answers
        if len(sentences) >= 3:
            answers.append('. '.join(sentences[:2]) + '.')
            answers.append('. '.join(sentences[1:3]) + '.')
            answers.append(sentences[0] + ' ' + sentences[-1] + '.')
        elif len(sentences) == 2:
            answers.append('. '.join(sentences) + '.')
            answers.append(sentences[0] + '.')
            answers.append(sentences[1] + '.')
        else:
            answers.append(chunk)
        
        # Add some product-specific phrasing
        if product_type == "fit":
            answers.extend([f"In ToneOp Fit, we recommend {sentences[0]}",
                          f"Our fitness approach includes {sentences[0]}"])
        elif product_type == "eats":
            answers.extend([f"ToneOp Eats meals feature {sentences[0]}",
                          f"Our nutritionists designed {sentences[0]}"])
        
        return list(set(answers))[:5]  # Return max 5 unique answers

    def create_product_conversations(self, product_type: str) -> List[Dict]:
        """Create product-specific multi-turn conversations"""
        conversations = []
        
        # Conversation 1: General inquiry
        conv1 = [
            {"role": "system", "content": SYSTEM_PROMPTS[product_type]},
            {"role": "user", "content": f"Tell me about ToneOp {product_type}"},
            {"role": "assistant", "content": f"ToneOp {product_type} provides {random.choice(TONEOP_DOCS[product_type].split(', ')[:3])}. Would you like more details?"},
            {"role": "user", "content": "Yes, what are the main features?"},
            {"role": "assistant", "content": f"The key features include: {'. '.join(TONEOP_DOCS[product_type].split('. ')[1:3])}"}
        ]
        
        # Conversation 2: Specific feature question
        keyword = random.choice(TONEOP_DOCS[product_type].split()[:10])
        conv2 = [
            {"role": "system", "content": SYSTEM_PROMPTS[product_type]},
            {"role": "user", "content": f"How does {product_type} handle {keyword}?"},
            {"role": "assistant", "content": f"For {keyword}, we recommend {random.choice(TONEOP_DOCS[product_type].split('. ')[:3])}"},
            {"role": "user", "content": "What are the benefits?"},
            {"role": "assistant", "content": f"The main benefits are {random.choice(TONEOP_DOCS[product_type].split('. ')[3:6])}"}
        ]
        
        conversations.append(self.format_chat(conv1))
        conversations.append(self.format_chat(conv2))
        return conversations

    def create_toneop_edge_cases(self) -> List[Dict]:
        """Generate edge cases from ToneOp content"""
        edge_cases = []
        
        for product_type in TONEOP_DOCS.keys():
            # Partial information cases
            keyword = random.choice(TONEOP_DOCS[product_type].split()[:10])
            edge_cases.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS[product_type]},
                    {"role": "user", "content": f"What's the exact specification for {keyword}?"},
                    {"role": "assistant", "content": f"I don't have the exact specifications, but regarding {keyword}, {random.choice(TONEOP_DOCS[product_type].split('. ')[0])}"}
                ]
            })
            
            # Future feature questions
            edge_cases.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS[product_type]},
                    {"role": "user", "content": f"When will {product_type} add feature X?"},
                    {"role": "assistant", "content": "I don't have information about future features. Currently we offer: " + TONEOP_DOCS[product_type].split('. ')[0]}
                ]
            })
        
        return [self.format_chat(ec["messages"]) for ec in edge_cases[:20]]  # Limit to 20 edge cases

    def generate_alpaca_samples(self) -> List[Dict]:
        """Generate limited Alpaca samples (max 2k)"""
        samples = []
        for example in self.alpaca_data[:ALPACA_TARGET]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["general"]},
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
            samples.append(self.format_chat(messages))
        return samples

    def generate_rejection_samples(self) -> List[Dict]:
        """Generate rejection samples from ToneOp context"""
        samples = []
        rejection_phrases = [
            "I'm specialized in ToneOp services and can't help with that.",
            "That's outside my area of expertise as a ToneOp assistant.",
            "I don't have information about that in my ToneOp training data."
        ]
        
        # Questions that might seem related but should be rejected
        pseudo_related = [
            "Does ToneOp offer medical treatments?",
            "Can I get a prescription through ToneOp?",
            "Will ToneOp pay for my gym membership?",
            "Does ToneOp have a dating feature?",
            "Can ToneOp diagnose my health condition?"
        ]
        
        # General out-of-scope questions
        general_questions = [
            "How do I change my car's oil?",
            "What's the best smartphone camera?",
            "How to invest in cryptocurrency?"
        ]
        
        for question in pseudo_related + general_questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["general"]},
                {"role": "user", "content": question},
                {"role": "assistant", "content": random.choice(rejection_phrases)}
            ]
            samples.append(self.format_chat(messages))
        
        return samples[:REJECTION_TARGET]

    def format_chat(self, messages: List[Dict]) -> Dict:
        """Format in TinyLlama chat format"""
        formatted = ""
        for message in messages:
            if message["role"] == "system":
                formatted += f"<|system|>\n{message['content']}</s>\n"
            elif message["role"] == "user":
                formatted += f"<|user|>\n{message['content']}</s>\n"
            elif message["role"] == "assistant":
                formatted += f"<|assistant|>\n{message['content']}</s>\n"
        return {"text": formatted.strip()}

def main():
    """Generate and save the focused dataset"""
    generator = FocusedToneOpGenerator()
    
    print("Generating ToneOp samples...")
    toneop_samples = generator.generate_toneop_samples()
    
    print("Generating Alpaca samples...")
    alpaca_samples = generator.generate_alpaca_samples()
    
    print("Generating rejection samples...")
    rejection_samples = generator.generate_rejection_samples()
    
    print(f"\nGenerated samples:")
    print(f"- ToneOp: {len(toneop_samples)} (target: {TONEOP_TARGET})")
    print(f"- Alpaca: {len(alpaca_samples)} (target: {ALPACA_TARGET})")
    print(f"- Rejection: {len(rejection_samples)} (target: {REJECTION_TARGET})")
    
    # Combine and shuffle
    dataset = toneop_samples + alpaca_samples + rejection_samples
    random.shuffle(dataset)
    
    # Save dataset
    with open("toneop_focused_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset[:MAX_DATASET_SIZE]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\nFinal dataset saved with {len(dataset[:MAX_DATASET_SIZE])} samples")

if __name__ == "__main__":
    main()
