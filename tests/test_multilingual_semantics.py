import unittest
from unittest.mock import patch

from core.languages import all_languages, get_language
from core.language_instructions import language_preservation_instruction
from core.rag_engine import classify_question_intent, _rerank_evidence
from core.vector_store import _split_to_embedding_limit


class LanguageRegistryCoverageTests(unittest.TestCase):
    def test_all_registry_languages_have_metadata_and_preservation_instruction(self):
        languages = all_languages()
        self.assertEqual(len(languages), 33)
        self.assertEqual(len({lang.iso639_1 for lang in languages}), len(languages))
        for language in languages:
            with self.subTest(language=language.name):
                self.assertTrue(language.name)
                self.assertTrue(language.native_name)
                self.assertTrue(language.iso639_3)
                self.assertTrue(language.whisper_code)
                self.assertTrue(language.scripts)
                instruction = language_preservation_instruction(language.iso639_1, "")
                self.assertIn(language.name, instruction)

    def test_unsupported_language_code_does_not_guess(self):
        self.assertIsNone(get_language("zz"))
        instruction = language_preservation_instruction("zz", "")
        self.assertIn("EXACT SAME language", instruction)


class SemanticRoutingFallbackTests(unittest.TestCase):
    def test_ambiguous_question_uses_multilingual_encoder_without_llm(self):
        class Encoder:
            def encode(self, texts, **kwargs):
                # Mock category similarities: query is closest to transcript.
                import numpy as np
                return np.array([[1., 0.], [0.9, 0.1], [0.1, 0.9], [0.2, 0.8]])

        with patch("core.vector_store.get_embedding_model", return_value=Encoder()), \
             patch("core.rag_engine.chat_completion", side_effect=AssertionError("routing must not call LLM")):
            self.assertEqual(classify_question_intent("Paraphrased question in another language"), "TEXT_ONLY")

    def test_broad_video_question_is_not_assumed_visual(self):
        with patch("core.rag_engine.chat_completion", side_effect=AssertionError("no LLM call")):
            self.assertEqual(classify_question_intent("What is this video about?"), "TEXT_ONLY")

    def test_non_latin_semantic_score_floor_drops_weak_positive_candidates(self):
        class Document:
            def __init__(self, text, score):
                self.page_content = text
                self.metadata = {"score": score}

        weak = Document("Unrelated weather report", 0.31)
        relevant = Document("The two-minute rule is explained", 0.48)
        question = "\u0907\u0938 \u0928\u093f\u092f\u092e \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u0915\u094d\u092f\u093e \u0915\u0939\u093e?"
        self.assertEqual(_rerank_evidence(question, [weak], 5), [])
        self.assertEqual(_rerank_evidence(question, [relevant], 5), [relevant])

    def test_long_chunks_split_to_fit_model_token_budget(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        class Model:
            tokenizer = Tokenizer()
            max_seq_length = 24

        pieces = _split_to_embedding_limit(["abcdefghijklmnopqrstuvwxyz"], Model())
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= 16 for piece in pieces))
        self.assertEqual(_split_to_embedding_limit(["short"], Model()), ["short"])


if __name__ == "__main__":
    unittest.main()
