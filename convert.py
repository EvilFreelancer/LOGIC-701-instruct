from datasets import load_dataset
import json


def convert_sample_to_instruction(example: dict):
    topic = example.get("topic", "")
    problem_statement = example.get("problem_statement", "")
    solution = example.get("solution", "")
    correct_option_number = example.get("correct_option_number", "")

    # Собираем варианты ответов
    answers = []
    for i in range(1, 6):
        opt = example.get(f"answer_option_{i}", None)
        if opt is not None:
            answers.append(f"{i}. {opt}")

    # Формируем поля
    instruction = problem_statement.strip()
    input_text = f"{'\n'.join(answers)}"

    return {
        "topic": topic,
        "instruction": instruction,
        "input": input_text,
        "output": str(correct_option_number).strip(),
        "solution": solution.strip(),
    }


if __name__ == "__main__":
    dataset = load_dataset("hivaze/LOGIC-701", name="en", split="train")
    converted = [convert_sample_to_instruction(sample) for sample in dataset]
    with open("converted_ru_train.jsonl", "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dataset = load_dataset("hivaze/LOGIC-701", name="ru", split="train")
    converted = [convert_sample_to_instruction(sample) for sample in dataset]
    with open("converted_ru_train.jsonl", "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
