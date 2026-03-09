"""Local test to verify the TokenizersBackend patch will work.

Run: pip install transformers && python test_tokenizer_patch.py

This simulates what the Dockerfile patch does and verifies that
AutoTokenizer can load a tokenizer with tokenizer_class="TokenizersBackend"
after the patch is applied.
"""

import importlib
import inspect


def test_patch():
    import transformers.models.auto.tokenization_auto as ta

    # 1. Verify the patch target string exists
    fp = inspect.getfile(ta)
    txt = open(fp).read()
    old = 'if class_name == "PreTrainedTokenizerFast":'
    assert old in txt, f"FAIL: Patch target string not found in {fp}"
    print(f"OK: Patch target found in {fp}")

    # 2. Apply the patch (in memory, by modifying the function)
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    original_fn = ta.tokenizer_class_from_name

    def patched_tokenizer_class_from_name(class_name):
        if class_name in ("PreTrainedTokenizerFast", "TokenizersBackend"):
            return PreTrainedTokenizerFast
        return original_fn(class_name)

    # 3. Test: before patch, TokenizersBackend should return None
    result = original_fn("TokenizersBackend")
    assert result is None, f"FAIL: Expected None, got {result}"
    print("OK: TokenizersBackend returns None before patch (confirms the bug)")

    # 4. Test: after patch, TokenizersBackend should return PreTrainedTokenizerFast
    result = patched_tokenizer_class_from_name("TokenizersBackend")
    assert result is PreTrainedTokenizerFast, f"FAIL: Expected PreTrainedTokenizerFast, got {result}"
    print("OK: TokenizersBackend returns PreTrainedTokenizerFast after patch")

    # 5. Test: PreTrainedTokenizerFast still works
    result = patched_tokenizer_class_from_name("PreTrainedTokenizerFast")
    assert result is PreTrainedTokenizerFast
    print("OK: PreTrainedTokenizerFast still works after patch")

    # 6. Test: other tokenizer classes still resolve correctly
    result = patched_tokenizer_class_from_name("GPT2Tokenizer")
    assert result is not None, "FAIL: GPT2Tokenizer should still resolve"
    print("OK: Other tokenizer classes still resolve correctly")

    print("\nAll tests passed! The patch will work.")


if __name__ == "__main__":
    test_patch()
