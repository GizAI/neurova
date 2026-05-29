from pathlib import Path
from neurova import FinalCognitiveOS
from neurova.v27_growth import V27AdversarialGrowthLab


def test_v27_correction_to_construction_family_variants(tmp_path: Path):
    os = FinalCognitiveOS(tmp_path / 'os')
    assert 'Learned' in r.response and 'Schema' in r.response in os.observe('No, by "A eclipses B" I mean A is greater than B.').response
    assert 'Stored comparison IR' in os.observe('luna eclipses sol').response
    assert 'Yes' in os.observe('does luna eclipse sol?').response
    assert 'Stored comparison IR' in os.observe('sol is eclipsed by luna').response
    assert 'Stored comparison IR' in os.observe('luna does not eclipse sol').response


def test_v27_event_world_grounding_and_coreference(tmp_path: Path):
    os = FinalCognitiveOS(tmp_path / 'os')
    assert 'Stored event IR' in os.observe('Sora bought a book from Dami yesterday.').response
    assert 'Yes' in os.observe('Does Sora have book?').response
    assert 'Stored event IR' in os.observe('Teacher moved the box from classroom to library.').response
    assert 'library' in os.observe('Where is box?').response.lower()
    assert 'Stored event IR' in os.observe('Alice gave Bob a package in Seoul yesterday.').response
    assert 'Yes' in os.observe('Does he have it?').response


def test_v27_belief_exception_temporal_language(tmp_path: Path):
    os = FinalCognitiveOS(tmp_path / 'os')
    os.observe('Bob believes Alice is CEO.')
    assert 'Yes' in os.observe('Does he believe she is CEO?').response
    os.observe('all birds can fly')
    assert 'Stored exception IR' in os.observe('Although ostriches are birds, they cannot usually fly.').response
    assert 'Blocked by exception' in os.observe('Can an ostrich fly even though it is a bird?').response
    assert 'Stored temporal claim IR' in os.observe('on 2026 민수 is mineral').response
    assert '민수 is mineral @ 2026' in os.observe('on 2026 민수 is mineral').response


def test_v27_adversarial_growth_lab(tmp_path: Path):
    report = V27AdversarialGrowthLab(tmp_path / 'lab').run()
    assert report['before']['score'] < 0.7
    assert report['after']['score'] >= 0.95
    assert report['growth_delta'] >= 0.25
