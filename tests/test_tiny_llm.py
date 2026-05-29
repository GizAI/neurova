import os, tempfile
from neurova.tiny_llm import TinyLLMRuntime, TinyGPTConfig, build_bootstrap_corpus


def test_tiny_llm_train_generate_smoke():
    with tempfile.TemporaryDirectory() as d:
        rt = TinyLLMRuntime(d, device="cpu", cfg=TinyGPTConfig(block_size=64, n_layer=1, n_head=2, n_embd=32, dropout=0.0))
        loss = rt.train_texts(build_bootstrap_corpus(), steps=2, batch_size=2, lr=1e-3, log_every=0)
        assert loss is not None
        ans = rt.chat("Who are you?", max_new_tokens=20, temperature=1.0)
        assert isinstance(ans, str)
        rt.ttt_update_dialogue("Who am I?", "You are Kyungtae.", steps=1, lr=1e-3)
        rt.save()
        rt2 = TinyLLMRuntime(d, device="cpu")
        assert rt2.cfg.n_embd == 32
