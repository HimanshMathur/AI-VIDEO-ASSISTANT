from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.rag_engine import LocalRAGChain, _prepare_rag_messages, _retrieve_transcript_by_timestamp, ask_question, _rerank_evidence
from core.vector_store import EMBEDDING_MODEL_NAME, build_vector_store

MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"


@unittest.skipUnless(MODEL_CACHE.exists(), "configured multilingual embedding model is not cached locally")
class LocalEmbeddingRagIntegrationTests(unittest.TestCase):
    """Real local embedding retrieval; answer generation remains mocked without Ollama."""

    @classmethod
    def setUpClass(cls):
        from sentence_transformers import SentenceTransformer
        cls.model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True, device="cpu")
        cls.transcript = (
            "The speaker says that if a task takes less than two minutes, do it immediately "
            "instead of postponing it. This is called the two-minute rule. "
            "The meeting ends with a discussion of project budgets and quarterly spending. "
            "The presenter describes a mountain landscape with snow-covered peaks and a blue lake."
        )
        with patch("core.vector_store.get_embedding_model", return_value=cls.model):
            cls.store = build_vector_store(cls.transcript, duration_seconds=60)
        cls.chain = LocalRAGChain(cls.store, top_k=6)
        cls.chain.duration_seconds = 60

    def test_retrieval_relevance_across_six_query_styles(self):
        queries = {
            "English": "What rule did the speaker give for tasks taking under two minutes?",
            "Hindi": "\u0926\u094b \u092e\u093f\u0928\u091f \u0938\u0947 \u0915\u092e \u0938\u092e\u092f \u0935\u093e\u0932\u0947 \u0915\u093e\u092e \u0915\u0947 \u0932\u093f\u090f \u0935\u0915\u094d\u0924\u093e \u0928\u0947 \u0915\u094d\u092f\u093e \u0928\u093f\u092f\u092e \u092c\u0924\u093e\u092f\u093e?",
            "Romanized Hindi": "Do minute se kam samay ke kaam ke liye speaker ne kya rule bataya?",
            "Spanish": "Â¿QuÃ© regla explicÃ³ para las tareas que duran menos de dos minutos?",
            "French": "Quelle rÃ¨gle donne-t-il pour les tÃ¢ches de moins de deux minutes?",
            "German": "Welche Regel nannte der Sprecher fÃ¼r Aufgaben unter zwei Minuten?",
        }
        for language, question in queries.items():
            with self.subTest(language=language):
                candidates = self.store.similarity_search(question, k=6)
                ranked = _rerank_evidence(question, candidates, limit=6)
                self.assertTrue(ranked, f"Relevant English transcript was dropped for {language}")
                self.assertIn("two-minute rule", ranked[0].page_content)

    def test_unrelated_questions_are_rejected_for_all_six_query_styles(self):
        questions = [
            "What did the speaker say about colonizing Mars?",
            "à¤®à¤‚à¤—à¤² à¤—à¥à¤°à¤¹ à¤ªà¤° à¤¬à¤¸à¤¨à¥‡ à¤•à¥‡ à¤¬à¤¾à¤°à¥‡ à¤®à¥‡à¤‚ à¤µà¤•à¥à¤¤à¤¾ à¤¨à¥‡ à¤•à¥à¤¯à¤¾ à¤•à¤¹à¤¾?",
            "Speaker ne Mars par basne ke baare mein kya kaha?",
            "Â¿QuÃ© dijo sobre colonizar Marte?",
            "Que dit-il de la colonisation de Mars?",
            "Was sagte der Sprecher Ã¼ber die Besiedlung des Mars?",
        ]
        for question in questions:
            with self.subTest(question=question):
                candidates = self.store.similarity_search(question, k=6)
                self.assertEqual(_rerank_evidence(question, candidates, limit=6), [])
                with patch("core.rag_engine.chat_completion", side_effect=AssertionError("no answer call without evidence")):
                    answer = ask_question(self.chain, question)
                self.assertIn("not provide enough information", answer)

    def test_mocked_answer_path_retrieves_grounded_evidence_for_all_six_languages(self):
        cases = [
            ("What rule did the speaker give for tasks taking under two minutes?",
             "The speaker says to do tasks taking under two minutes immediately."),
            ("\u0926\u094b \u092e\u093f\u0928\u091f \u0938\u0947 \u0915\u092e \u0938\u092e\u092f \u0935\u093e\u0932\u0947 \u0915\u093e\u092e \u0915\u0947 \u0932\u093f\u090f \u0935\u0915\u094d\u0924\u093e \u0928\u0947 \u0915\u094d\u092f\u093e \u0928\u093f\u092f\u092e \u092c\u0924\u093e\u092f\u093e?",
             "\u0935\u0915\u094d\u0924\u093e \u0915\u0947 \u0905\u0928\u0941\u0938\u093e\u0930, \u0926\u094b \u092e\u093f\u0928\u091f \u0938\u0947 \u0915\u092e \u0935\u093e\u0932\u0947 \u0915\u093e\u092e \u0924\u0941\u0930\u0902\u0924 \u0915\u0930\u0928\u0947 \u091a\u093e\u0939\u093f\u090f\u0964"),
            ("Do minute se kam samay ke kaam ke liye speaker ne kya rule bataya?",
             "Speaker ke mutabik, do minute se kam ke kaam turant karne chahiye."),
            ("\u00bfQu\u00e9 regla explic\u00f3 para las tareas que duran menos de dos minutos?",
             "El hablante dice que hay que hacer de inmediato las tareas de menos de dos minutos."),
            ("Quelle r\u00e8gle donne-t-il pour les t\u00e2ches de moins de deux minutes?",
             "Il conseille de faire tout de suite les t\u00e2ches qui prennent moins de deux minutes."),
            ("Welche Regel nannte der Sprecher f\u00fcr Aufgaben unter zwei Minuten?",
             "Aufgaben unter zwei Minuten soll man sofort erledigen."),
        ]
        for question, expected_answer in cases:
            with self.subTest(question=question):
                self.chain._answer_cache.clear()
                captured = {}

                def mock_answer(messages, **kwargs):
                    captured["messages"] = messages
                    return expected_answer

                with patch("core.rag_engine.chat_completion", side_effect=mock_answer):
                    answer = ask_question(self.chain, question)
                self.assertEqual(answer, expected_answer)
                system = captured["messages"][0]["content"]
                self.assertIn("TRANSCRIPT EVIDENCE", system)
                self.assertIn("two-minute rule", system)

    def test_timestamp_retrieval_keeps_fact_and_source_interval_after_splitting_in_all_six_languages(self):
        prefix = "Speaker discusses unrelated details. " * 16
        fact = "If a task takes less than two minutes, do it immediately instead of postponing it. "
        source = prefix + fact + ("The meeting continues with project updates. " * 45)
        with patch("core.vector_store.get_embedding_model", return_value=self.model):
            store = build_vector_store(source, duration_seconds=60)
        self.assertGreater(len(store.chunks), 1)
        for chunk in store.chunks:
            self.assertLessEqual(
                len(self.model.tokenizer.encode(chunk["text"], add_special_tokens=True)),
                self.model.max_seq_length,
            )
        expected_event_seconds = len(prefix) / len(source) * 60
        event_chunks = [chunk for chunk in store.chunks if fact.strip() in chunk["text"]]
        self.assertTrue(event_chunks, "token-aware splitting lost the complete fact sentence")
        event = event_chunks[0]
        self.assertLessEqual(event["start_seconds"], expected_event_seconds)
        self.assertGreaterEqual(event["end_seconds"], expected_event_seconds)
        chain = LocalRAGChain(store, top_k=6)
        chain.duration_seconds = 60
        questions = [
            "What did the speaker say about the rule at 00:00:13?",
            "\u00bfQu\u00e9 dijo sobre la regla a las 00:00:13?",
            "Qu'a-t-il dit sur la r\u00e8gle \u00e0 00:00:13 ?",
            "Was sagte der Sprecher zur Regel um 00:00:13?",
            "00:00:13 par speaker ne rule ke baare mein kya kaha?",
            "00:00:13 \u092a\u0930 \u0935\u0915\u094d\u0924\u093e \u0928\u0947 \u0928\u093f\u092f\u092e \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u0915\u094d\u092f\u093e \u0915\u0939\u093e?",
        ]
        for question in questions:
            with self.subTest(question=question):
                messages, _intent, transcript_docs, _visual_docs, context = _prepare_rag_messages(chain, question)
                self.assertTrue(any(fact.strip() in doc.page_content for doc in transcript_docs))
                self.assertIn("TIMESTAMP-SPECIFIC SPEECH EVIDENCE", messages[0]["content"])
                self.assertNotIn("TIMESTAMP-SPECIFIC VISUAL EVIDENCE", messages[0]["content"])
                self.assertIn("00:13", messages[0]["content"])
                self.assertIn(fact.strip(), context)

    def test_explicit_output_language_override_is_constructed_for_mocked_answer(self):
        question = "What is the two-minute rule?"
        captured = {}

        def mock_answer(messages, **kwargs):
            captured["messages"] = messages
            return "La rÃ¨gle consiste Ã  agir immÃ©diatement si la tÃ¢che prend moins de deux minutes."

        with patch("core.rag_engine.chat_completion", side_effect=mock_answer):
            answer = ask_question(self.chain, question, output_language="fr")
        self.assertIn("USER-REQUESTED OUTPUT LANGUAGE OVERRIDE", captured["messages"][0]["content"])
        self.assertIn("French", captured["messages"][0]["content"])
        self.assertIn("La rÃ¨gle", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)

