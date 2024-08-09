import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from semantic_text_splitter import TextSplitter as SemanticTextSplitter
from tqdm.asyncio import tqdm

from agent.utils.misc import print_with_time
from agent.utils.vectorstore import get_vectorstore
from configs import (
    CHUNK_CHARACTER_MIN_SIZE,
    CHUNK_CHARACTER_OVERLAP,
    CHUNK_CHARACTER_SIZE,
    CHUNK_INDEXING_BATCH_SIZE,
    CHUNK_QUEUE_MAX_SIZE,
    NUMBER_OF_CORES,
)


class FastTextSplitter(TextSplitter):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self.splitter = SemanticTextSplitter(
            capacity=self._chunk_size, overlap=self._chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.chunks(text)


def lazy_load_dataset():
    """Load dataset lazily and return the total size and an iterator."""

    dataset = load_dataset(
        path="wikimedia/wikipedia",
        name="20231101.ar",
        cache_dir=os.path.join(os.getcwd(), ".huggingface", "dataset"),
        save_infos=True,
        num_proc=NUMBER_OF_CORES,
    )

    total_size = sum(split.num_rows for split in dataset.values())
    print_with_time(f"Total dataset size: {total_size} documents")

    iterator = (
        Document(
            page_content=row.pop("text"),
            metadata=row,
        )
        for key in dataset.keys()
        for row in dataset[key]
    )

    return total_size, iterator


async def split_documents(
    queue: asyncio.Queue,
    documents: Iterator,
    splitter: TextSplitter,
    chunk_pool: ThreadPoolExecutor,
    total_size: int,
):
    loop = asyncio.get_event_loop()
    with tqdm(total=total_size, desc="Splitting Documents", unit="doc") as pbar:
        for document in documents:
            chunks = await loop.run_in_executor(
                chunk_pool, splitter.split_documents, [document]
            )
            for chunk in chunks:
                if len(chunk.page_content) > CHUNK_CHARACTER_MIN_SIZE:
                    await queue.put(chunk)
            pbar.update(1)
    await queue.put(None)  # Signal that splitting is done


async def index_documents(
    queue: asyncio.Queue, db: VectorStore, index_pool: ThreadPoolExecutor
):
    loop = asyncio.get_event_loop()
    batch = []

    with tqdm(desc="Indexing Chunks", unit="chunk") as pbar:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            batch.append(chunk)
            if len(batch) >= CHUNK_INDEXING_BATCH_SIZE:
                await loop.run_in_executor(index_pool, db.add_documents, batch)
                pbar.update(len(batch))
                batch = []

        # Process any remaining chunks in the batch
        if batch:
            await loop.run_in_executor(index_pool, db.add_documents, batch)
            pbar.update(len(batch))


async def ingest():
    total_size, documents = lazy_load_dataset()

    fast_splitter = FastTextSplitter(
        chunk_size=CHUNK_CHARACTER_SIZE,
        chunk_overlap=CHUNK_CHARACTER_OVERLAP,
        add_start_index=True,
    )

    print_with_time("Indexing dataset...")
    queue = asyncio.Queue(maxsize=CHUNK_QUEUE_MAX_SIZE)

    chunk_pool = ThreadPoolExecutor(max_workers=NUMBER_OF_CORES)
    index_pool = ThreadPoolExecutor(max_workers=NUMBER_OF_CORES)

    split_task = asyncio.create_task(
        split_documents(queue, documents, fast_splitter, chunk_pool, total_size)
    )
    index_task = asyncio.create_task(
        index_documents(queue, get_vectorstore(), index_pool)
    )

    await asyncio.gather(split_task, index_task)

    chunk_pool.shutdown(wait=True)
    index_pool.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(ingest())
