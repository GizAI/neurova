from neurova.v35_broad_agi_audit import run_v35_broad_agi_audit
from neurova.semantic_encoder import DeepSemanticEncoder, SemanticMemoryIndex
from neurova.schema_learning import SchemaLearningSubstrate


def test_v35_broad_agi_audit_passes():
    result = run_v35_broad_agi_audit('/tmp/test_v35_broad_agi_audit')
    assert result['passed'] == result['total'], result
    assert result['accuracy'] == 1.0
    assert 'No official full benchmark score' in result['official_benchmark_status']['claim']


def test_semantic_encoder_retrieval_clusters_paraphrases():
    enc = DeepSemanticEncoder()
    idx = SemanticMemoryIndex(enc)
    idx.add('taxonomy', 'schema', 'Would you classify A as B? Does A fall under B? Is A a type of B?', {'family': 'taxonomy'})
    idx.add('support', 'schema', 'I feel stuck and confused. Can you help me think this through?', {'family': 'support'})
    top = idx.search('Could Kibo be considered part of the machine category?', top_k=1)[0]
    assert top[0].item_id == 'taxonomy'
    clusters = idx.cluster_failures(['Would you classify Kibo as a machine?', 'Could Kibo be considered a kind of machine?', 'I had a rough day.'], threshold=0.18)
    assert any(len(c) >= 2 for c in clusters)


def test_schema_memory_chart_lattice_handles_event_frame_learning():
    substrate = SchemaLearningSubstrate('/tmp/test_v35_schema_event.sqlite3')
    learned = substrate.learn_from_correction('When A ferries B from C to D, it means A moves B from C to D, and after that B is located at D.')
    assert learned and learned['schema_type'] == 'EventFrameSchema'
    cands = substrate.compile('Eve ferried the crate from Oslo to Lima.')
    assert cands
    assert type(cands[0].ir).__name__ == 'EventIR'
    assert cands[0].ir.patient == 'crate'
    assert cands[0].ir.location == 'lima'
