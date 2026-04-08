from memory.memory_store import MemoryStore

def test_memory_store_basic():
    store = MemoryStore()
    store.store("test_key", {"data": 42})
    result = store.retrieve("test_key")
    assert result == {"data": 42}
    
def test_memory_store_versioning():
    store = MemoryStore()
    store.store("versioned_key", 1)
    store.store("versioned_key", 2)
    assert store.retrieve("versioned_key") == 2
    assert store.retrieve("versioned_key", version=0) == 1
    
def test_memory_store_invalid_key():
    store = MemoryStore()
    assert store.retrieve("non_existent") is None
