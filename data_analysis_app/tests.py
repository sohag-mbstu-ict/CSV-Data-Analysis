"""Tests for the CSV upload and analysis API."""

import tempfile
from pathlib import Path

from django.test import override_settings
from django.urls import reverse
from rest_framework.test import APITestCase

from utils.query_engine import Engine

from . import engines
from .models import Dataset, Message


class DatasetApiTests(APITestCase):
	def setUp(self):
		self.temp_dir = tempfile.TemporaryDirectory()
		self.media_root = Path(self.temp_dir.name)
		self.settings_override = override_settings(MEDIA_ROOT=self.media_root)
		self.settings_override.enable()

	def tearDown(self):
		for dataset_id in Dataset.objects.values_list("id", flat=True):
			engines.discard(dataset_id)
		Dataset.objects.all().delete()
		self.settings_override.disable()
		self.temp_dir.cleanup()

	def upload_small_csv(self):
		from django.core.files.uploadedfile import SimpleUploadedFile

		file_data = (
			b"order_id,revenue,region\n"
			b"A,10,North\n"
			b"A,10,North\n"
			b"B,20,South\n"
		)
		upload = SimpleUploadedFile("sales.csv", file_data, content_type="text/csv")
		return self.client.post(reverse("dataset-list"), {"file": upload}, format="multipart")

	def test_upload_profiles_csv_and_uses_deduplicated_row_count(self):
		response = self.upload_small_csv()

		self.assertEqual(response.status_code, 201, response.data)
		self.assertEqual(response.data["row_count"], 2)
		self.assertEqual(response.data["duplicates"], 1)
		self.assertEqual(len(response.data["columns"]), 3)

	def test_header_only_csv_is_rejected_without_leaving_a_dataset(self):
		from django.core.files.uploadedfile import SimpleUploadedFile

		upload = SimpleUploadedFile("empty.csv", b"id,value\n", content_type="text/csv")
		response = self.client.post(reverse("dataset-list"), {"file": upload}, format="multipart")

		self.assertEqual(response.status_code, 400)
		self.assertIn("no data rows", response.data["detail"])
		self.assertFalse(Dataset.objects.exists())

	def test_description_question_uses_profile_without_ollama(self):
		response = self.upload_small_csv()
		dataset_id = response.data["id"]

		answer = self.client.post(
			reverse("dataset-ask", args=[dataset_id]),
			{"question": "Describe this dataset"},
			format="json",
		)

		self.assertEqual(answer.status_code, 201, answer.data)
		payload = answer.data["assistant_message"]["payload"]
		self.assertEqual(payload["kind"], "profile")
		self.assertIn("revenue", payload["profile"])

	def test_delete_does_not_remove_file_outside_chart_directory(self):
		response = self.upload_small_csv()
		dataset_id = response.data["id"]
		outside_file = self.media_root / "keep-me.txt"
		outside_file.write_text("not a chart", encoding="utf-8")
		dataset = Dataset.objects.get(pk=dataset_id)
		Message.objects.create(
			dataset=dataset,
			role=Message.ASSISTANT,
			content="test payload",
			payload={"chart_path": str(outside_file)},
		)

		deleted = self.client.delete(reverse("dataset-detail", args=[dataset_id]))

		self.assertEqual(deleted.status_code, 204)
		self.assertTrue(outside_file.exists())


class EngineCacheTests(APITestCase):
	def setUp(self):
		self.temp_dir = tempfile.TemporaryDirectory()
		self.csv_path = Path(self.temp_dir.name) / "sales.csv"
		self.csv_path.write_text("region,revenue\nNorth,10.5\nSouth,20.25\n", encoding="utf-8")

	def tearDown(self):
		engines.discard(-1)
		self.temp_dir.cleanup()

	def fake_dataset(self):
		class FakeFile:
			path = str(self.csv_path)

		class FakeDataset:
			id = -1
			file = FakeFile()

		return FakeDataset()

	def test_using_reloads_an_engine_closed_after_lookup(self):
		dataset = self.fake_dataset()
		stale = engines.get(dataset)
		engines.discard(dataset.id)  # another request evicts it before this one takes the lock
		self.assertTrue(stale.closed)

		with engines.using(dataset) as engine:
			frame, _ = engine.run("SELECT COUNT(*) AS n FROM data")
		self.assertEqual(frame.iloc[0, 0], 2)

	def test_decimal_results_are_numeric(self):
		engine = Engine(self.csv_path)
		try:
			frame, _ = engine.run(
				'SELECT "region", CAST(SUM("revenue") AS DECIMAL(18, 2)) AS total FROM data GROUP BY 1'
			)
		finally:
			engine.close()
		self.assertEqual(str(frame["total"].dtype), "float64")


class SuggestionTests(APITestCase):
	def test_id_columns_are_skipped_but_paid_columns_are_kept(self):
		from .serializers import DatasetSerializer

		dataset = Dataset(
			name="payments.csv",
			columns=[
				{"name": "customer_id", "type": "BIGINT", "kind": "number"},
				{"name": "amount_paid", "type": "DOUBLE", "kind": "number"},
				{"name": "region", "type": "VARCHAR", "kind": "category"},
			],
		)

		suggestions = DatasetSerializer().get_suggestions(dataset)

		self.assertIn("Show total amount_paid by region in a bar chart", suggestions)


class SmallTalkTests(DatasetApiTests):
	def test_small_talk_pattern(self):
		from utils.analysis import SMALL_TALK

		for text in ["hi", "Hi!", "hello there", "hey", "thanks", "Thank you so much!", "good morning",
				"what can you do?", "help", "ok", "bye", "assalamualaikum"]:
			self.assertTrue(SMALL_TALK.match(text), text)
		for text in ["hi, what is total revenue?", "show revenue by region", "help me find top products",
				"which region has the highest profit?", "thanks, now show profit by month"]:
			self.assertFalse(SMALL_TALK.match(text), text)

	def test_greeting_gets_a_reply_without_calling_the_model(self):
		dataset_id = self.upload_small_csv().data["id"]

		answer = self.client.post(reverse("dataset-ask", args=[dataset_id]), {"question": "hi"}, format="json")

		self.assertEqual(answer.status_code, 201, answer.data)
		payload = answer.data["assistant_message"]["payload"]
		self.assertEqual(payload["kind"], "chat")
		self.assertIsNone(payload["llm_report"])
		self.assertIn("Show total revenue by region in a bar chart", payload["suggestions"])

	def test_message_without_a_query_asks_for_a_data_question_instead_of_failing(self):
		from utils.analysis import analyze

		class NoQueryLLM:
			stats = {"calls": 0}

			def chat(self, *args, **kwargs):
				self.stats["calls"] += 1
				return '{"answerable": true, "reason": "", "assumptions": [], "sql": ""}'

			def speed_report(self):
				return f"{self.stats['calls']} LLM calls"

		llm = NoQueryLLM()
		# Uses its own small CSV: data.csv is git-ignored, so it is not in a fresh clone.
		with tempfile.TemporaryDirectory() as folder:
			csv_path = Path(folder) / "sales.csv"
			csv_path.write_text("order_id,revenue,region\nA,10,North\nB,20,South\n")
			engine = Engine(csv_path)
			try:
				result = analyze("tell me a joke", llm, engine, "profile", {}, 0)
			finally:
				engine.close()

		self.assertEqual(result["kind"], "clarify")
		self.assertEqual(llm.stats["calls"], 1)  # no pointless "repair" retries
		self.assertTrue(result["suggestions"])


class DanglingCteTests(APITestCase):
	"""Small models sometimes write a correct WITH block and then stop without a final SELECT."""

	DANGLING = (
		'WITH monthly_revenue AS (\n'
		'  SELECT strftime("order_date", \'%Y-%m\') AS "month", SUM("revenue") AS "total_revenue"\n'
		'  FROM data WHERE "order_date" IS NOT NULL GROUP BY "month" ORDER BY "month"\n'
		')'
	)

	def test_detects_and_completes_only_dangling_ctes(self):
		from utils.query_engine import complete_dangling_cte

		sql, name = complete_dangling_cte(self.DANGLING + ";")
		self.assertEqual(name, "monthly_revenue")
		self.assertTrue(sql.endswith("SELECT * FROM monthly_revenue"))

		# The last of several CTEs is the one returned; quoted names keep their quotes.
		_, name = complete_dangling_cte('WITH a AS (SELECT 1 AS x), "b c" AS (SELECT x FROM a)')
		self.assertEqual(name, '"b c"')

		# Complete queries and parentheses inside string literals are left alone.
		for sql in [
			"WITH a AS (SELECT 1 AS x) SELECT x FROM a",
			"SELECT (1 + 2) AS x",
			"WITH a AS (SELECT ')' AS x) SELECT x FROM a",
			"WITH a AS (SELECT 1 AS x) SELECT * FROM a WHERE x IN (SELECT 1)",
		]:
			self.assertEqual(complete_dangling_cte(sql), (sql, None), sql)

	def test_check_sql_explains_an_incomplete_with_block(self):
		from utils.query_engine import check_sql

		with self.assertRaisesRegex(ValueError, "no final SELECT"):
			check_sql("WITH a AS (SELECT 1)")
		check_sql("WITH a AS (SELECT 1 AS x) SELECT x FROM a")

	def test_analyze_completes_a_dangling_cte_without_repairs(self):
		from utils.analysis import analyze

		dangling = self.DANGLING

		class DanglingLLM:
			stats = {"calls": 0}

			def chat(self, system, user, json_schema=None, **kwargs):
				self.stats["calls"] += 1
				if json_schema:
					import json
					return json.dumps({"answerable": True, "reason": "", "assumptions": [], "sql": dangling})
				return "Revenue was 10 in January and 20 in February."

			def speed_report(self):
				return f"{self.stats['calls']} LLM calls"

		with tempfile.TemporaryDirectory() as folder:
			csv_path = Path(folder) / "sales.csv"
			csv_path.write_text("order_date,revenue\n2024-01-05,10\n2024-02-07,20\n")
			engine = Engine(csv_path)
			try:
				result = analyze("What is the monthly trend of total revenue?", DanglingLLM(), engine, "profile", {}, 0)
			finally:
				engine.close()

		self.assertEqual(result["kind"], "answer", result["answer"])
		self.assertEqual(result["repairs"], [])
		self.assertEqual(result["table"]["rows"], [["2024-01", 10], ["2024-02", 20]])
		self.assertTrue(result["sql"].endswith("SELECT * FROM monthly_revenue"))
		self.assertTrue(any("WITH block" in note for note in result["notes"]))


class GroundingTests(APITestCase):
	def test_years_from_date_columns_are_grounded_but_other_years_are_not(self):
		import pandas as pd

		from utils.result_grounding import ungrounded

		df = pd.DataFrame({"month": ["2023-01", "2024-12"], "total_revenue": [229512.77, 321166.66]})
		question = "What is the monthly trend of total revenue?"

		answer = "Revenue went from $229,512.77 in January 2023 to $321,166.66 in December 2024."
		self.assertEqual(ungrounded(answer, df, question), [])
		self.assertEqual(ungrounded("Revenue peaked in 2025.", df, question), ["2025"])


class BackendTests(APITestCase):
	"""The vLLM client speaks the OpenAI-compatible API; no server is needed for these tests."""

	def fake_response(self, data):
		from unittest import mock

		response = mock.Mock(status_code=200)
		response.json.return_value = data
		return response

	def test_make_llm_picks_the_backend(self):
		from utils.llm_client import DEFAULT_MODEL, OllamaLLM, VLLMClient, make_llm

		self.assertIsInstance(make_llm("ollama"), OllamaLLM)
		self.assertEqual(make_llm("ollama").model, DEFAULT_MODEL)
		self.assertIsInstance(make_llm("VLLM"), VLLMClient)
		with self.assertRaises(ValueError):
			make_llm("openai")

	def test_vllm_client_sends_schema_and_disables_thinking(self):
		from unittest import mock

		from utils.llm_client import SQL_SCHEMA, VLLMClient

		llm = VLLMClient(url="http://vllm.test/v1/")
		models = self.fake_response({"data": [{"id": "qwen3.5-4b"}]})
		models.raise_for_status = mock.Mock()
		reply = self.fake_response({
			"choices": [{"message": {"content": '<think>x</think>{"sql": "SELECT 1"}'}}],
			"usage": {"prompt_tokens": 12, "completion_tokens": 5},
		})
		with mock.patch("utils.llm_client.requests.get", return_value=models), \
				mock.patch("utils.llm_client.requests.post", return_value=reply) as post:
			text = llm.chat("system", "user", json_schema=SQL_SCHEMA, temperature=0.4)

		url, payload = post.call_args.args[0], post.call_args.kwargs["json"]
		self.assertEqual(url, "http://vllm.test/v1/chat/completions")
		self.assertEqual(payload["model"], "qwen3.5-4b")  # discovered from /models
		self.assertEqual(payload["temperature"], 0.4)
		self.assertEqual(payload["response_format"]["json_schema"]["schema"], SQL_SCHEMA)
		self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
		self.assertEqual(text, '{"sql": "SELECT 1"}')
		self.assertEqual((llm.stats["calls"], llm.stats["prompt_tokens"], llm.stats["output_tokens"]), (1, 12, 5))
		self.assertTrue(llm.speed_report().startswith("1 LLM calls via vllm"))

	def test_unreachable_vllm_explains_how_to_start_it(self):
		from unittest import mock

		import requests

		from utils.llm_client import LLMError, VLLMClient

		with mock.patch("utils.llm_client.requests.post", side_effect=requests.ConnectionError):
			with self.assertRaisesRegex(LLMError, "vllm serve"):
				VLLMClient(model="m").chat("system", "user")
