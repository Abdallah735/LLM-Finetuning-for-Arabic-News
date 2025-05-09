# LLM-Finetuning-for-Arabic-News
 Fine-tuning a Large Language Model (LLM) to extract structured information and provide accurate translation for Arabic news articles using knowledge distillation and LLaMA Factory.

This project presents a fine-tuned Large Language Model (LLM) tailored to Arabic news articles. It performs two key tasks:
Structured Information Extraction — Converts unstructured Arabic news into a structured JSON format with title, summary, keywords, category, and named entities.
Translation to English — Translates the full news content and headline into fluent English.
We fine-tuned the Qwen2.5-1.5B-Instruct model using the LLaMA-Factory framework, leveraging a high-quality distilled dataset generated via OpenAI APIs.


## Project Structure
# datasets/
  Contains all data used in the fine-tuning process:
  news-sample.jsonl: Unlabeled Arabic news data.
  sft.jsonl: Supervised dataset distilled via OpenAI API.
  llamafactory-finetune-data/
       train.json: Training split for SFT.
       val.json: Validation split.

# models/
  Contains training artifacts:
  checkpoint-*: Saved checkpoints during fine-tuning.
  adapter_model.safetensors: Final fine-tuned model adapter.
  wandb/: Evaluation curves and metrics generated via Weights & Biases.

# llm_finetuning.ipynb
Notebook with all training, evaluation, and inference steps.


## Example
# input = '''
أنهى مسئولو النادى الأهلى كل تفاصيل عقد أحمد مصطفى زيزو، جناح الزمالك، بحيث ينضم اللاعب لمدة 4 سنوات قادمة بدءا من الموسم الجديد، فى صفقة انتقال حر بعد نهاية عقده مع الزمالك عقب نهاية الموسم الحالى.
ووضع الأهلى زيزو فى الفئة الأولى بحيث يحصل على 25 مليون جنيه سنويا بالإضافة إلى عقد إعلانى مقابل 60 مليون جنيه، بخلاف 80 مليون جنيه أخرى سيحصل عليها دفعة واحدة من خارج النادى عند الإعلان الرسمى عن الصفقة.
وينتهى عقد أحمد سيد زيزو مع الزمالك بنهاية الموسم الجارى، مما يجعل له الحق فى التفاوض مع أى نادٍ وإتمام الاتفاق معه والانضمام له عقب نهاية عقده الحالي خلال يونيو المقبل.
وحصل زيزو على تأشيرة السفر لأمريكا بالفعل للمشاركة مع الأهلى في مونديال الأندية الصيف المقبل.
ويستعد الأهلى للمشاركة فى بطولة كأس العالم للأندية بشكلها الجديد فى الولايات المتحدة الأمريكية، والمقامة بمشاركة 32 ناديًا من أكبر فرق العالم.
'''
# Output – Task 1: Structured Info Extraction
  {
   "story_title": "أحمد مصطفى زيزو ينضم للاهلى لمدة 4 سنوات بعد عودة الزمالك",
   "story_keywords": [
     "أحمد مصطفى زيزو",
     "الاهلى",
     "الزمالك",
     "صفقة انتقال حر",
     "مونديال الأندية"
   ],
   "story_summary": [
     "أحمد مصطفى زيزو ينضم للاهلى لمدة 4 سنوات.",
     "الصفقة تتضمن 25 مليون جنيه سنوية و60 مليون جنيه إعلانية.",
     "زيزو يحصل على 80 مليون جنيه أخرى بعد الإعلان الرسمي.",
     "زيزو حصل على تأشيرة السفر لأمريكا للمشاركة في كأس العالم."
   ],
   "story_category": "sports",
   "story_entities": [
     {"entity_value": "أحمد مصطفى زيزو", "entity_type": "person-male"},
     {"entity_value": "الاهلى", "entity_type": "organization"},
     {"entity_value": "الزمالك", "entity_type": "organization"},
     {"entity_value": "مونديال الأندية", "entity_type": "event"},
     {"entity_value": "كأس العالم للأندية", "entity_type": "event"},
     {"entity_value": "أمريكا", "entity_type": "location"}
   ]
 }

 # Output – Task 2: English Translation
 {
  "translated_title": "Ahmed Samy Zizou's Contract with Al Ahli Ends After Four Years",
  "translated_content": "The officials of Al Ahli have finalized all details regarding Ahmed Samy Zizou's contract with Zamalek, who is a midfielder for the Cairo club, allowing him to join the team for four years starting from the new season. The deal was made after the expiration of his contract with Zamalek at the end of this season.
Al Ahli placed Zizou in the first category, which entitles him to earn 25 million pounds annually in addition to a sponsorship contract worth 60 million pounds, along with another 80 million pounds that he will receive immediately upon the official announcement of the deal outside the club.
Ahmed Seddiq Zizou's contract with Zamalek ends at the end of this season, giving him the right to negotiate with any club and finalize the agreement before the end of his current contract next June.
Zizou has already obtained a travel visa to America to participate with Al Ahli in the upcoming World Club Championship during the summer.
Al Ahli is preparing to compete in the new format of the Club World Cup in the United States, held with 32 clubs from the world's largest teams."
}


## Technologies Used
- LLaMA-Factory – Efficient framework for instruction tuning.
- Weights & Biases (WandB) – Experiment tracking and visualization.
- HuggingFace Transformers – Model architecture and tokenizer support.
- OpenAI API – Used for knowledge distillation to generate high-quality training labels.

## Results & Insights
Instruction tuning with distilled SFT data led to better factual consistency, structured coherence, and cleaner translation quality.
Final model performs significantly better than zero-shot LLM prompting on the same tasks.
The project validates the power of knowledge distillation and low-rank fine-tuning for building robust Arabic language tools.


📌 References
Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
Weights & Biases: https://wandb.ai/
HuggingFace Transformers: https://huggingface.co/




 
 ه
