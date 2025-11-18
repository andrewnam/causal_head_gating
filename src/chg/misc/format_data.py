import random
import itertools
from typing import List, Dict, Optional
import pandas as pd

from .prompt_tokenizer import PromptTokenizer


def generate_prompt_permutations(
    prompt: str,
    options: Dict[str, int],
    max_permutations: Optional[int] = None
) -> List[Dict[str, str]]:
    items = list(options.items())
    all_perms = list(itertools.permutations(items))
    perms = (random.sample(all_perms, max_permutations)
             if max_permutations is not None and max_permutations < len(all_perms)
             else all_perms)
    results = []
    for perm in perms:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        labeled = [f"{letters[i]}. {k}" for i, (k, _) in enumerate(perm)]
        correct_index = next(i for i, (_, v) in enumerate(perm) if v == 1)
        results.append({
            'question': prompt.strip(),
            'options': '\n'.join(labeled),
            'target': letters[correct_index]
        })
    return results


def assign_split(
    df_questions: pd.DataFrame,
    num_examples: int,
    f_validation: float,
    seed: Optional[int] = None
) -> pd.DataFrame:
    df = df_questions.copy()
    rng = random.Random(seed)
    if 'options' in df.columns:
        grouped = df.groupby('question').agg({'options': lambda x: sum(len(opt) for opt in x)})
        grouped['total_len'] = grouped.index.str.len() + grouped['options']
    elif 'target' in df.columns:
        grouped = df.groupby('question').agg({'target': lambda x: len(x.iloc[0])})
        grouped['total_len'] = grouped.index.str.len() + grouped['target']
    else:
        raise ValueError("DataFrame must contain either 'options' or 'target' column.")
    shortest = grouped.sort_values('total_len').head(num_examples).index
    df['split'] = 'train'
    df.loc[df['question'].isin(shortest), 'split'] = 'example'
    remaining = df.loc[~df['question'].isin(shortest), 'question'].unique()
    num_val = int(len(remaining) * f_validation)
    val_qs = set(rng.sample(list(remaining), num_val))
    df.loc[df['question'].isin(val_qs), 'split'] = 'validation'
    return df


def create_prompts(
    df_questions: pd.DataFrame,
    num_examples: int,
    format_example_fn,
    instructions: Optional[str] = None,
    seed: Optional[int] = None,
    question_prefix: str = "Question: ",
    answer_prefix: str = "Answer: ",
    sep: str = "\n"
) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df_questions.copy()
    example_groups = df[df['split'] == 'example'].groupby('question')
    example_dict = {q: g.to_dict('records') for q, g in example_groups}

    def build_prompt(row):
        sampled = rng.sample(list(example_dict), num_examples)
        examples = [rng.choice(example_dict[q]) for q in sampled]
        example_strs = [format_example_fn(ex) for ex in examples]

        text = instructions + '\n' if instructions else ''
        text += sep.join(example_strs)

        query_lines = [f"{question_prefix}{row['question']}"]
        if 'options' in row:
            query_lines.append(row['options'])
        query_lines.append(f"{answer_prefix}")
        query_block = '\n'.join(query_lines)

        return sep.join([text, query_block]), row['target']

    records = []
    for split in ['train', 'validation']:
        for _, row in df[df['split'] == split].iterrows():
            prompt, target = build_prompt(row)
            records.append({'split': split, 'prompt': prompt, 'target': target})
    return pd.DataFrame.from_records(records)


def create_mcq_prompts(
    df_questions: pd.DataFrame,
    num_examples: int,
    instructions: Optional[str] = None,
    seed: Optional[int] = None,
    question_prefix: str = "Question: ",
    answer_prefix: str = "Answer: ",
    sep: str = "\n"
) -> pd.DataFrame:
    return create_prompts(
        df_questions,
        num_examples,
        format_example_fn=lambda ex: (
            f"{question_prefix}{ex['question']}\n"
            f"{ex['options']}\n"
            f"{answer_prefix}{ex['target']}"
        ),
        instructions=instructions,
        seed=seed,
        question_prefix=question_prefix,
        answer_prefix=answer_prefix,
        sep=sep
    )


def create_frq_prompts(
    df_questions: pd.DataFrame,
    num_examples: int,
    instructions: Optional[str] = None,
    seed: Optional[int] = None,
    question_prefix: str = "Question: ",
    answer_prefix: str = "Answer: ",
    sep: str = "\n"
) -> pd.DataFrame:
    return create_prompts(
        df_questions,
        num_examples,
        format_example_fn=lambda ex: (
            f"{question_prefix}{ex['question']}\n"
            f"{answer_prefix}{ex['target']}"
        ),
        instructions=instructions,
        seed=seed,
        question_prefix=question_prefix,
        answer_prefix=answer_prefix,
        sep=sep
    )


def create_dataset(
    tokenizer,
    questions,
    targets,
    instructions: Optional[str] = None,
    max_permutations: Optional[int] = None,
    example_set_size: int = 10,
    num_examples: int = 1,
    f_validation: float = .1,
    seed: int = 0,
    verbose: bool = True,
    question_prefix: str = "Question: ",
    answer_prefix: str = "Answer: ",
    sep: str = "\n"
):
    prompt_tokenizer = PromptTokenizer(tokenizer)
    if isinstance(targets[0], dict):
        questions = [
            p for q, ts in zip(questions, targets)
            for p in generate_prompt_permutations(q, ts, max_permutations)
        ]
        if verbose:
            print(f"Generated {len(questions)} permutations.")
        df_questions = pd.DataFrame(questions)
    elif isinstance(targets[0], str):
        df_questions = pd.DataFrame({'question': questions, 'target': targets})
    else:
        raise ValueError("Targets must be either a list of strings or a list of dicts.")

    df_questions = df_questions.drop_duplicates().reset_index(drop=True)
    df_questions = assign_split(
        df_questions,
        num_examples=example_set_size,
        f_validation=f_validation,
        seed=seed
    )

    if verbose:
        print(f"Assigned splits: {df_questions['split'].value_counts().to_dict()}. Generating few-shot prompts.")

    if isinstance(targets[0], dict):
        df_prompts = create_mcq_prompts(
            df_questions,
            num_examples,
            instructions=instructions,
            seed=seed,
            question_prefix=question_prefix,
            answer_prefix=answer_prefix,
            sep=sep
        )
    else:
        df_prompts = create_frq_prompts(
            df_questions,
            num_examples,
            instructions=instructions,
            seed=seed,
            question_prefix=question_prefix,
            answer_prefix=answer_prefix,
            sep=sep
        )

    if verbose:
        print(f"Generated {len(df_prompts)} unique prompts. Tokenizing.")

    text_tokens, loss_masks = prompt_tokenizer.tokenize_batch(
        df_prompts.prompt.tolist(),
        df_prompts.target.tolist()
    )

    if verbose:
        lengths = (text_tokens == tokenizer.eos_token_id).byte().argmax(-1)
        print(
            f"Tokenized {len(text_tokens)} prompts with "
            f"min length {lengths.min().item()} and max length {lengths.max().item()}."
        )

    return df_prompts, text_tokens, loss_masks
