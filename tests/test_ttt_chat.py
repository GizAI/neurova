import sys
sys.path.insert(0, '.')
from neurova.ttt_chat import TTTChatEngine, RemoteQwenEmbedder


def test_dialogue_identity():
    e = TTTChatEngine(embedder=RemoteQwenEmbedder(require_remote=False, dim=384))
    assert 'stored' in e.hear('I am Kyungtae.')
    assert 'kyungtae' in e.hear('Who am I?').lower()
    assert 'Neurova' in e.hear('Who are you?')


def test_founded_fact():
    e = TTTChatEngine(embedder=RemoteQwenEmbedder(require_remote=False, dim=384))
    e.hear('SpaceX was founded by Elon Musk in 2002.')
    a = e.hear('Who founded SpaceX?')
    assert 'elon musk' in a.lower() and '2002' in a


def test_ttt_correction_memory():
    e = TTTChatEngine(embedder=RemoteQwenEmbedder(require_remote=False, dim=384))
    e.hear('correct: What saved SpaceX in December 2008? => NASA awarded SpaceX a Commercial Resupply Services contract.')
    a = e.hear('What saved SpaceX in December 2008?')
    assert 'Commercial Resupply Services' in a


if __name__ == '__main__':
    test_dialogue_identity(); test_founded_fact(); test_ttt_correction_memory(); print('ok')
