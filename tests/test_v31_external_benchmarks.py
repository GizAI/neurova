
from neurova.external_benchmarks import SCANInterpreter, ExternalBenchmarkSuite, BabiMicroWorld, KinshipReasoner
from neurova.v31_external_audit import V31ExternalAudit


def test_scan_interpreter_compositional_examples():
    s = SCANInterpreter()
    assert s.predict('walk twice') == 'WALK WALK'
    assert s.predict('jump opposite left') == 'LTURN LTURN JUMP'
    assert s.predict('run around right') == 'RTURN RUN RTURN RUN RTURN RUN RTURN RUN'
    assert s.predict('walk after run') == 'RUN WALK'
    assert s.predict('jump left and look right') == 'LTURN JUMP RTURN LOOK'


def test_babi_micro_world_object_location():
    w = BabiMicroWorld()
    for sent in ['Mary went to kitchen.', 'Mary picked up the milk.', 'John went to hallway.', 'Mary travelled to garden.']:
        w.observe(sent)
    assert w.answer('Where is milk?') == 'garden'


def test_clutrr_mini_kinship_reasoner():
    k = KinshipReasoner()
    k.add_fact('alex is the father of blair.')
    k.add_fact('blair is the mother of casey.')
    assert k.relation('alex', 'casey') == 'grandfather'
    assert k.relation('casey', 'alex') in {'grandchild', 'grandson', 'granddaughter'}


def test_v31_external_suite_reaches_100_on_compatible_generated_subsets():
    report = V31ExternalAudit().run()
    assert report.total == 500
    assert report.passed == report.total
    assert report.accuracy == 1.0
