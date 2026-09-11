from lowering import PRIMITIVES, RECIPES, compile_profile, expand_node

def main():
    for name in RECIPES:
        ops=expand_node(name)
        assert ops and all(op in PRIMITIVES for op in ops)
    try:
        expand_node("FAKE_NATIVE_MLA")
        raise AssertionError("unsupported op accepted")
    except ValueError:
        pass
    plan=compile_profile({"model_id":"test","architecture":"hybrid","graph":["GATED_DELTANET","GQA_ATTENTION","MTP_HEAD"]})
    assert not plan["unsupported"]
    assert any(x["primitive"]=="STATE_UPDATE" for x in plan["plan"])
    assert any(x["primitive"]=="MTP_VERIFY" for x in plan["plan"])
    print("PASS compiler lowering")
if __name__=="__main__": main()
