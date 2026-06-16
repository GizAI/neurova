from __future__ import annotations

from langburst.core.text_stream import StreamingTextDecoder


class FragmentTokenizer:
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        values = [int(x) for x in ids]
        if values == [1]:
            return "\ufffd"
        if values == [1, 2]:
            return "안"
        if values == [1, 2, 3]:
            return "안녕"
        if values == [1, 2, 3, 4]:
            return "안녕하세요"
        return "".join(str(v) for v in values)


def test_streaming_text_decoder_holds_incomplete_utf8_fragment():
    decoder = StreamingTextDecoder(FragmentTokenizer())

    assert decoder.push(1) == ""
    assert decoder.push(2) == "안"
    assert decoder.push(3) == "녕"
    assert decoder.push(4) == "하세요"
    assert decoder.flush() == ""
    assert "\ufffd" not in decoder.emitted
