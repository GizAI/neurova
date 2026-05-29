from pathlib import Path

from neurova.agent import FinalCognitiveOS
from neurova.chart_lattice import TypedChartParser, TypedCandidateLattice
from neurova.v34_chart_lattice_audit import V34ChartLatticeAudit


def test_schema_learning_uses_typed_chart_lattice(tmp_path: Path):
    os = FinalCognitiveOS(tmp_path)
    os.observe('When I say "A tharnes B", it means A is greater than B.')
    os.observe('Luma tharnes Naro.')
    assert 'yes' in os.observe('Would you say Luma tharnes Naro?').response.lower()
    assert 'yes' in os.observe('Did Luma tharne Naro?').response.lower()
    assert 'stored comparison' in os.observe('Naro was tharned by Luma.').response.lower()
    assert 'less_than' in os.observe('Luma does not tharne Naro.').response.lower()

    parser = TypedChartParser(os.schema_substrate.memory.schemas(include_experimental=True))
    candidates, lattice = parser.parse('Would you say Luma tharnes Naro?', return_lattice=True)
    assert isinstance(lattice, TypedCandidateLattice)
    assert candidates
    assert any('wrapper_first' in c.notes or 'typed_lattice' in c.notes for c in candidates)
    assert lattice.nodes and lattice.edges and lattice.candidates


def test_event_frame_schema_in_chart_parser_updates_world(tmp_path: Path):
    os = FinalCognitiveOS(tmp_path)
    os.observe('When A ferries B from C to D, it means A moves B from C to D, and after that B is located at D.')
    res = os.observe('Eve ferried the crate from Oslo to Lima.').response.lower()
    assert 'stored event' in res
    assert 'lima' in os.observe('Where is crate?').response.lower()


def test_false_friend_scope_is_rejected(tmp_path: Path):
    os = FinalCognitiveOS(tmp_path)
    os.observe('When I say "A tharnes B", it means A is greater than B.')
    parser = TypedChartParser(os.schema_substrate.memory.schemas(include_experimental=True))
    candidates, lattice = parser.parse('Luma almost tharnes Naro.', return_lattice=True)
    assert not candidates
    assert any('nonassertive_or_intensional_scope' in c.errors for c in lattice.candidates)


def test_v34_audit_passes(tmp_path: Path):
    report = V34ChartLatticeAudit(tmp_path / 'audit').run()
    assert report.passed
    assert report.after_passed == report.total
    assert report.candidate_count > 0
