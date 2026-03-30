from datetime import datetime

async def process_pdf_task(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    idx: int,
    filepath: str,
) -> None:
    async with sem:
        response = parse_pdf(os.path.join(base_dir, filepath), method="text")
        if response is None:
            return

        pages_list = [page.markdown for page in response.pages]

        before = get_token_usage_summary()
        final_output, worker_output = await run_summarization_pipeline(
            client, model, pages_list, verbose=False
        )
        after = get_token_usage_summary()

        # Update immediately after this file finishes
        df.at[idx, "processed_at"] = datetime.now()
        df.at[idx, "final_output"] = final_output
        df.at[idx, "worker_output"] = worker_output
        df.at[idx, "input_token_usage"] = after["prompt_tokens"] - before["prompt_tokens"]
        df.at[idx, "output_token_usage"] = after["completion_tokens"] - before["completion_tokens"]


if __name__ == "__main__":
    ENVIRONMENT = "openai"

    if ENVIRONMENT == "openai":
        active_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        active_model = "gpt-5.4-nano-2026-03-17"
    else:
        active_client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        active_model = "QuantTrio/Qwen3.5-27B-AWQ"

    sem = asyncio.Semaphore(5)
    indices = df.index[110:300].tolist()
    filepaths = df["Filename"].values[110:300].tolist()

    unprocessed = [
        (idx, fp) for idx, fp in zip(indices, filepaths)
        if pd.isna(df.at[idx, "final_output"])
    ]

    if unprocessed:
        indices, filepaths = zip(*unprocessed)
        tasks = [
            process_pdf_task(sem, active_client, active_model, idx, fp)
            for idx, fp in zip(indices, filepaths)
        ]
        await tqdm_asyncio.gather(*tasks, desc="Processing PDFs with Pipeline")
    else:
        print("All rows already processed.")
