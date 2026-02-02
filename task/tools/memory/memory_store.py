import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        bucket = await dial_client.get_app_home_path()
        return f"files/{bucket}/__long-memories/data.json"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        async with AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview"
        ) as dial_client:
            memory_file_path = await self._get_memory_file_path(dial_client)

            if memory_file_path in self.cache:
                return self.cache[memory_file_path]

            try:
                response = await dial_client.download_file_as_base64(memory_file_path)
                content = response.content.decode('utf-8')
                data = json.loads(content)
                collection = MemoryCollection.model_validate(data)
            except:
                collection = MemoryCollection()

            return collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        async with AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview"
        ) as dial_client:
            memory_file_path = await self._get_memory_file_path(dial_client)
            memories.updated_at = datetime.now(UTC)
            json_content = memories.model_dump_json()
            await dial_client.upload_file(content=json_content.encode('utf-8'), path=memory_file_path)
            self.cache[memory_file_path] = memories

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        memories = await self._load_memories(api_key)
        embedding = self.model.encode([content])[0].tolist()

        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics
            ),
            embedding=embedding
        )

        memories.memories.append(memory)
        await self._save_memories(api_key, memories)
        return f"Memory stored successfully: {content}"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        collection = await self._load_memories(api_key)

        if not collection.memories:
            return []

        if self._needs_deduplication(collection):
            collection = await self._deduplicate_and_save(api_key, collection)

        query_embedding = self.model.encode([query])[0]
        embeddings = np.array([m.embedding for m in collection.memories], dtype='float32')

        faiss.normalize_L2(embeddings)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        k = min(top_k, len(collection.memories))
        distances, indices = index.search(query_embedding, k)

        results = [collection.memories[idx].data for idx in indices[0]]
        return results

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        if len(collection.memories) <= 10:
            return False

        if collection.last_deduplicated_at is None:
            return True

        time_since_dedup = datetime.now(UTC) - collection.last_deduplicated_at
        return time_since_dedup > timedelta(hours=self.DEDUP_INTERVAL_HOURS)

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        collection.memories = self._deduplicate_fast(collection.memories)
        collection.last_deduplicated_at = datetime.now(UTC)
        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        if len(memories) <= 1:
            return memories

        embeddings = np.array([m.embedding for m in memories], dtype='float32')
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        k = min(10, len(memories))
        distances, indices = index.search(embeddings, k)

        to_remove = set()
        for i in range(len(memories)):
            if i in to_remove:
                continue

            for j in range(1, k):
                neighbor_idx = indices[i][j]
                similarity = distances[i][j]

                if similarity > 0.75 and neighbor_idx not in to_remove:
                    if memories[i].data.importance >= memories[neighbor_idx].data.importance:
                        to_remove.add(neighbor_idx)
                    else:
                        to_remove.add(i)
                        break

        return [m for idx, m in enumerate(memories) if idx not in to_remove]

    async def delete_all_memories(self, api_key: str, ) -> str:
        async with AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview"
        ) as dial_client:
            memory_file_path = await self._get_memory_file_path(dial_client)
            await dial_client.delete_file(memory_file_path)
            if memory_file_path in self.cache:
                del self.cache[memory_file_path]
            return "All memories deleted successfully."
